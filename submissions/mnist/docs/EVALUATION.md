# Reproducing the CROSS / TPU evaluation

Every number in `RESULTS.md` comes from the commands below. Timings are what
they took on the reference platform; substitute your own paths freely.

**Reference platform.** One Google Cloud TPU v6e VM (`v6e-8`): 8 × v6e chips,
31.2 GiB HBM each, 180 vCPU host, 1.4 TiB RAM. Python 3.13, JAX 0.11.2 with
`jax[tpu]`, CROSS (`jaxite_word`) 3.0.0, PyTorch 2.14 CPU, OpenFHE 1.3.1.

A TPU is not required for everything: step 2 verifies the submission on CPU
alone, and step 6 re-derives the op breakdown from a committed trace.

---

## 0. Prerequisites

```console
# CROSS, which is used in place (it is not packaged)
git clone https://github.com/EfficientPPML/CROSS.git ~/CROSS
export CROSS_ROOT=~/CROSS          # optional: see note below

# this repository
git clone https://github.com/fhe-benchmarking/ml-inference.git
cd ml-inference
```

`CROSS_ROOT` is optional. The submission looks for `jaxite_word` under
`$CROSS_ROOT`, then on `PYTHONPATH`, then in `../CROSS`,
`third_party/cross` and `~/CROSS`, and names the paths it searched if it
finds nothing.

Python environment:

```console
conda create -y --name jaxite python=3.13 && conda activate jaxite
pip install -U "jax[tpu]" absl-py numpy
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

CPU-only PyTorch is enough — torch is used for the model definition and
tracing, never for the encrypted evaluation.

---

## 1. Build the submission  *(seconds)*

```console
./scripts/build_task.sh ./submissions/mnist
```

CROSS is a JAX library, so nothing is compiled. This checks the runtime is
present, confirms the trained weights exist, and marks the stage executables
executable. The expensive artifact — the compiled `Mapping` with its keys,
BSGS diagonals and encoded constants — is built later by benchmark stage 3,
where the harness measures it.

`submissions/mnist/model/he_mlp_weights.pth` is committed. To retrain from scratch (~40 min on
CPU) delete it and rerun, or:

```console
python3 submissions/mnist/model/train_he_mlp.py --epochs 30
python3 submissions/mnist/model/export_weights.py   # folds BatchNorm, re-checks
```

---

## 2. Self-test — no TPU required  *(~90 s)*

```console
python3 submissions/mnist/src/selftest.py
```

Ten checks covering the accelerator-independent half of the submission:

```
  ok    CROSS library is importable
  ok    model weights load and the BatchNorm fold is exact  (fold deviation 2.38e-07)
  ok    client and server derive the same ring  (degree=32768 slots=16384 depth=5 security=128-bit)
  ok    manifest mismatch is refused
  ok    encrypt -> file -> decrypt round-trips with scale intact  (max error 3.1e-13)
  ok    a corrupt ciphertext is rejected by name
  ok    wrong-shape ciphertext is rejected
  ok    atomic write leaves no partial file on failure
  ok    paths do not create directories when merely read
  ok    processes that write under run/ create it first

all 10 checks passed
```

If this passes, the submission is intact; only the accelerator half is
unverified.

---

## 3. Run the benchmark on TPU

The harness is used unmodified; it picks up `submissions/mnist/<stage>.py`
through its own `run_exe_or_python` helper.

```console
python3 harness/run_submission.py 0 --seed 3 --num_runs 3   # single (1)
python3 harness/run_submission.py 1 --seed 3 --num_runs 3   # small  (100)
python3 harness/run_submission.py 2 --seed 3 --num_runs 3   # medium (1000)
```

The `Mapping` is built once per invocation and reused across all three runs, so
`--num_runs 3` costs roughly one build plus three cheap inference passes.

Knobs (all optional):

| variable | default | meaning |
|---|---|---|
| `CROSS_DEVICE_COUNT` | 8 | TPU chips; also the global ciphertext batch |
| `CROSS_BSGS_JOBS` | 1 | workers encoding BSGS diagonals (offline only) |
| `CROSS_STARTUP_TIMEOUT` | 5400 | seconds to wait for stage 3 |
| `CROSS_REQUEST_TIMEOUT` | 7200 | seconds to wait for stage 7 |
| `CROSS_DAEMON_IDLE_TIMEOUT` | 3600 | idle seconds before the server exits |

**Expected output**, small instance:

```
[server] CROSS Mapping resident on 8 TPU chip(s), global_batch=8
[server] ring degree=32768 slots=16384 depth=5 security=128-bit
[server] setup: pack 0.81s, keys 0.09s, mapping 466.18s, warmup 3.67s
[server] rotation keys: 134, live memory estimate: 3.17 GiB
   3: Server: (Encrypted) model preprocessing completed (elapsed: ~475-485s)
[server] encrypted compute: 0.307s for 15 inference(s) -> 20.4 ms each, 48.9 inferences/s
   7: Server: Encrypted ML Inference computation completed (elapsed: ~0.5s)
[harness] Encrypted Model Accuracy: 0.9333 (14/15 correct)
[harness] Harness Model Accuracy:   0.9333 (14/15 correct)
[total latency] ~508s
```

Roughly 8 minutes per invocation, almost all of it stage 3 building the
`Mapping` once. Stage 7 — the encrypted inference itself — is well under a
second. The single instance prints `PASS (expected=7, got=7)` instead of an
accuracy.

**Release the TPU when done** (the server stays resident by design so repeated
runs do not rebuild):

```console
python3 submissions/mnist/server_stop.py
```

---

## 4. The TPU scaling sweep  *(~35 min)*

```console
python3 submissions/mnist/src/profile_tpu.py --devices 1,2,4,8 --local-batch 1
```

Builds one `Mapping` per configuration and times 20 steady-state executes
after 3 warmups, at 1, 2, 4 and 8 chips, checking every configuration against
the cleartext model before timing it. It also captures a `jax.profiler` trace
of the 8-chip run and prints the op breakdown. Reproduces the `RESULTS.md`
section 2 table:

```
chips  lbatch  gbatch  execute ms  ms/infer   infer/s  speedup
    1       1       1       147.7    147.71      6.77    1.00x
    2       1       2       151.2     75.59     13.23    1.95x
    4       1       4       152.0     38.01     26.31    3.89x
    8       1       8       153.9     19.24     51.97    7.68x
```

Writes `results/tpu_profile/profile_results.json`. Sweep a subset with
`CROSS_SWEEP_DEVICES=1,8`.

> A per-chip batch above 1 aborts the XLA TPU fusion emitter on this stack
> (reproduced twice; evidence in `results/tpu_profile/crash/`), so the sweep
> does not attempt it. `CROSS_SWEEP_LOCAL_BATCH=2` tries anyway on a stack
> where it is fixed.

---

## 5. Where the time goes  *(seconds, no TPU)*

```console
python3 submissions/mnist/src/trace_breakdown.py \
    submissions/mnist/docs/results/tpu_profile/traces/8chip_lb1 --iterations 3
```

Groups device time by XLA's `hlo_category` across all 8 device lanes and
normalizes to per-chip-per-execute. Add `--keyword-report` to also run CROSS's
own `profile_analysis/analyze_trace_json.py`, and `--device N` for one chip.

```
   1782.40   51.5%   310992  loop fusion          <- elementwise modular arithmetic
   1056.84   30.5%    19056  custom fusion        <- NTT / basis conversion matmuls
    320.50    9.3%   217947  data formatting
    252.80    7.3%    87192  convolution fusion
   3462.85  100.0%           total (leaf nodes)

normalized: 3462.8 ms over 8 lane(s) x 3 iteration(s) = 144.3 ms per chip per execute
```

144.3 ms of device time against 153.9 ms measured end to end — the difference
is host dispatch. Traces are large and therefore not committed; a committed
`results/tpu_profile/op_breakdown_8chip.txt` holds this output, and step 4
regenerates the trace.

---

## 6. The OpenFHE CPU baseline, for comparison  *(~25 min)*

This submission replaces `submissions/mnist/` and the two `scripts/` helpers,
so running the upstream OpenFHE reference again means restoring them from the
upstream branch in a separate worktree:

```console
git worktree add /tmp/ml-inference-upstream origin/main
cd /tmp/ml-inference-upstream
./scripts/get_openfhe.sh                                    # ~10 min, needs cmake
python3 harness/run_submission.py 0 --seed 3 --num_runs 3
python3 harness/run_submission.py 1 --seed 3 --num_runs 3
```

Expect ~8 s per encrypted inference on a comparable CPU. The reference numbers
quoted in `RESULTS.md` were measured this way on the same host, on the earlier
harness revision whose small instance was 15 samples rather than 100 — so
compare the **per-inference** figures, not the per-instance totals.

> **Measure on an idle host.** The same binary measured 142 s, 70 s and 8 s
> per inference on this machine depending on what else was running; a
> 243-thread training job was enough to inflate it 18×. Check `uptime` first.
> Only idle-host numbers are reported in `RESULTS.md`.

---

## 7. Reading the committed results

```
submissions/mnist/docs/results/
├─ cross_tpu/{single,small}/   harness measurements + per-stage timings + run logs
├─ baseline_openfhe/{single,small}/   the same, for the reference submission
└─ tpu_profile/
   ├─ profile_results.json     the sweep: latency, throughput, memory, error
   ├─ op_breakdown_8chip.txt   section 5 output
   ├─ sweep.log                raw sweep log
   └─ crash/                   the XLA abort, twice, with stack
```

`python3 submissions/mnist/src/report.py --sizes 0,1` prints a consolidated
view; `--markdown` emits the sweep as a table.

---

## Troubleshooting

**`Unable to initialize backend 'tpu' ... Device or resource busy`**
Another process holds the chips — a TPU admits exactly one. Stage 3 stops any
resident CROSS server itself; for anything else, find it with
`pgrep -af libtpu` or `ls /proc/*/fd | grep vfio`.

**`Internal error when accessing libtpu multi-process lockfile`**
A stale `/tmp/libtpu_lockfile` from a process killed abnormally. Stage 3
clears it; otherwise `rm -f /tmp/libtpu_lockfile` once nothing is running.

**Stage 3 exits with "CROSS server exited ... before becoming ready"**
The daemon died during the build. Its last 25 log lines are printed, and the
full log is at `submissions/mnist/run/<instance>/cross_server.log`.

**`Check failed: window.strides[...] == current_scaled_dim_window_bound`**
The XLA fusion-emitter abort, hit only with more than one ciphertext per chip.
Keep `global_batch` equal to the chip count.

**Stage 6 or 8 is slow.** Expected: CROSS's client codec runs on CPU at ring
degree 32768 — ~1.8 s to encrypt and ~0.6 s to decrypt per sample, against
OpenFHE's ~16 ms and ~21 ms at degree 2048. This is the honest weak side of
the deployment and is discussed in `RESULTS.md` caveat 2.

**Out of HBM.** One chip peaks at 9.31 GiB of 31.2 GiB at `global_batch =
chips`. If you lowered `CROSS_DEVICE_COUNT`, the same constants are still
replicated per chip; lower it further or use a larger topology.
