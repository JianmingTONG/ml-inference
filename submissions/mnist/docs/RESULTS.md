# CROSS on TPU v6e — MNIST ML-inference results

**Platform.** One Google Cloud TPU v6e VM: 8 × v6e chips (32 GiB HBM each,
33.5 GB reported limit), 180 vCPU host, 1.4 TiB RAM. JAX 0.11.2, CROSS
(`jaxite_word`) 3.0.0, OpenFHE 1.3.1, torch 2.14 CPU.

**Reproducing these numbers.** See [EVALUATION.md](EVALUATION.md); each
section below names the command that produced it.

**Workload.** The benchmark's MNIST inference, `784 → 128 → 64 → 10`, with the
harness's ReLU replaced by x² (the only substitution CROSS's activation
registry accepts). See `README.md` for the architecture change and the
client/server split.

## Headline

| | result |
|---|---|
| Benchmark verdict | **PASS** — single instance 3/3 runs; small instance 14/15 = 0.9333, identical to the harness plaintext model |
| Encrypted inference, 1 TPU v6e chip | **147.7 ms** |
| Encrypted inference, 8 chips | **19.24 ms** — 51.97 inferences/s |
| Strong scaling, 1 → 8 chips | **7.68×** (near-linear) |
| vs OpenFHE on this host's 180-vCPU CPU | **54×** at batch 1, **413×** at 8-chip throughput |
| Numerical error vs cleartext | max logit deviation **5.2e-12** |
| Security | **128-bit classical** (the reference's parameters are `HEStd_NotSet`) |
| One-off server cost | 467 s Mapping build, 3.17 GiB live, 9.31 GiB HBM of 31.2 |

The speedups come with two conditions that matter: CROSS is running a 16×
larger ring at a real security level, and its CPU-side client codec is far
slower than OpenFHE's, which dominates the single-query end-to-end time.
Section 7 lists these in full.

---

## 1. Cryptographic parameters are not comparable at face value

| | CROSS (this submission) | reference (`../submission`, OpenFHE) |
|---|---|---|
| ring degree | 32768 | 2048 |
| slots | 16384 | 1024 |
| modulus chain | 12 Q + 4 P towers, `dnum=3` | multiplicative depth 9 |
| security | **128-bit classical** | `HEStd_NotSet` |
| emitted depth | 5 | 9 |

`packing.pack` derived the CROSS ring from the program itself; nothing in this
submission chose it. The reference sets `SetSecurityLevel(HEStd_NotSet)` with
`SetRingDim(1 << 11)`, which is not a secure parameter set. **Every latency
number below should be read with this in mind: CROSS is doing 16× the ring
work per ciphertext, at a real 128-bit security level.**

---

## 2. Encrypted inference latency

Steady-state `Mapping.execute`, median of 20 timed iterations after 3 warmups,
each configuration verified against the cleartext model on the same inputs
before timing. One `Mapping` per configuration — device topology and global
batch are static compilation inputs in CROSS.

| chips | ct/chip | batch | execute (ms) | ms/inference | inferences/s | scaling | max logit err |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 147.7 | 147.71 | 6.77 | 1.00× | 2.5e-12 |
| 2 | 1 | 2 | 151.2 | 75.59 | 13.23 | 1.95× | 3.7e-12 |
| 4 | 1 | 4 | 152.0 | 38.01 | 26.31 | 3.89× | 4.1e-12 |
| 8 | 1 | 8 | 153.9 | **19.24** | **51.97** | **7.68×** | 5.2e-12 |

**Strong scaling is near-linear to all eight chips** (7.68× of a possible 8×).
Adding chips costs almost nothing in per-execute latency — 147.7 ms on one chip
to 153.9 ms on eight — because each chip runs an independent ciphertext and the
only growth is dispatch. Execute time is very stable: at 8 chips the 20 samples
span 153.8–154.2 ms, stdev 0.13 ms.

**Per-chip batching could not be measured**: every attempt to compile a
Mapping with more than one ciphertext per chip crashed the XLA TPU compiler
(below). Whether the chips are saturated at one ciphertext each is therefore
open — the ring carries 16384 slots for a model that uses 784, which suggests
headroom, but that is an inference from slot occupancy, not a measurement.

Batch-1 *latency* does not improve with more chips: one query occupies one lane
and the other seven are padding, so a single encrypted inference costs ~148 ms
whether one chip or eight is present.

### Against the OpenFHE CPU reference, same host

| | CROSS / TPU v6e | OpenFHE / CPU (180 vCPU) |
|---|---:|---:|
| per encrypted inference, batch 1 | 147.7 ms (1 chip) | 7950 ms |
| per encrypted inference, best throughput | **19.24 ms** (8 chips) | 7950 ms |
| inferences/s | **51.97** | 0.126 |
| speedup, like-for-like batch 1 | **54×** | 1× |
| speedup, best measured throughput | **413×** | 1× |

Again: the reference is running a ring 16× smaller at `HEStd_NotSet`. This is a
deployed-system comparison, not a kernel-for-kernel one.

### Configurations that do not compile

**Any per-chip batch above 1 aborts the XLA TPU compiler.** Two independent
attempts at `8 chips × 2 ciphertexts` (global batch 16) both died during
Mapping construction, roughly eight minutes into the build:

```
F fusion_emitter.cc:2988] Check failed:
  window.strides[physical_scaled_dim_index] == current_scaled_dim_window_bound
  (1 vs. 4) Scaled dimension stride is not equal to window bound.
*** SIGABRT received ***
```

This is a fatal `CHECK` inside `xla::jellyfish::BitcastInput::TransformStaticWindow`,
reached through the nested fusion emitters for the BSGS giant-step loops. It is
a compiler-side limit, not an out-of-memory condition — the build aborts before
HBM is anywhere near full, and the single-ciphertext-per-chip configuration
peaks at 9.31 GiB of 31.2 GiB.

**So every latency figure in this report is at one ciphertext per chip**, and
`global_batch = number of chips` is the only shape of this Mapping that
compiles on this JAX/libtpu version.

## 3. Where the time goes

`jax.profiler` trace of the 8-chip configuration, 3 executes, all 8 device
lanes. Grouped by XLA's own `hlo_category` rather than by op-name keywords:
CROSS's `profile_analysis/analyze_trace_json.py` classifies by name, which
works when each HLO is one recognizable kernel, but this Mapping compiles into
`loop fusion` / `custom fusion` nodes whose names carry no arithmetic meaning —
under name matching 54% of device time lands in "Other". The `while` nodes are
the BSGS giant-step loops and their duration *contains* their children's, so
they are reported separately rather than double-counted.

| device time | share | count | category |
|---:|---:|---:|---|
| 1782.40 ms | 51.5% | 310992 | loop fusion |
| 1056.84 ms | 30.5% | 19056 | custom fusion |
| 320.50 ms | 9.3% | 217947 | data formatting |
| 252.80 ms | 7.3% | 87192 | convolution fusion |
| 24.21 ms | 0.7% | 17664 | broadcast |
| 19.33 ms | 0.6% | 38112 | dynamic-slice |
| 6.77 ms | 0.2% | — | copy / async / slice / iota / custom-call |
| **3462.85 ms** | **100%** | | **total (leaf nodes)** |
| 4037.69 ms | — | | `while` containers (include the leaves above) |

**This trace validates the wall-clock measurement.** 3462.85 ms over 8 lanes ×
3 executes is **144.3 ms of device time per chip per execute**, against
153.9 ms measured end to end — the ~10 ms difference is host dispatch and the
`block_until_ready` boundary.

Elementwise modular arithmetic (`loop fusion`) is half the time and the NTT /
basis-conversion matmuls (`custom fusion`) are another third; layout work
(`data formatting`) is under 10%. Reproduce with:

```console
python3 submissions/mnist/src/trace_breakdown.py \
    submissions/mnist/docs/results/tpu_profile/traces/8chip_lb1 --iterations 3
```

The saved output, including CROSS's name-keyword view for comparison, is in
`docs/results/tpu_profile/op_breakdown_8chip.txt`.

## 4. End-to-end benchmark stages

The authoritative per-stage record is the committed measurement files, produced
by the unmodified harness with `--num_runs 3`:

```
measurements/single/results-{1,2,3}.json     1 inference
measurements/small/results-{1,2,3}.json    100 inferences
measurements/medium/results-{1,2,3}.json  1000 inferences
```

Each carries the harness `Timing` and `Bandwidth` blocks plus a
`Server Reported` block this submission publishes through the harness's
`io/<size>/server_reported_steps.json` hook, which separates the pure
accelerator time from the stage wall clock that also covers ciphertext file
I/O:

| Server Reported key | meaning |
|---|---|
| `server_mapping_build` | building the CKKS Mapping (offline, once) |
| `server_mapping_warmup` | first execute, so XLA compilation is not billed later |
| `server_tpu_execute` | accelerator time for the whole batch |
| `server_tpu_execute_per_inference` | the headline per-image figure |
| `server_ciphertext_load` / `_store` | ciphertext file I/O inside stage 7 |

**Single instance**, from `measurements/single/results-3.json`:

| | value |
|---|---:|
| Encrypted model preprocessing (stage 3) | 478.17 s *(once)* |
| — of which Mapping build | 460.89 s |
| Input encryption (stage 6) | 4.07 s |
| **Encrypted computation (stage 7)** | **0.264 s** |
| — **TPU execute only** | **0.1537 s** |
| Result decryption (stage 8) | 2.74 s |
| public + evaluation keys | 8.0 MB¹ |
| encrypted input / result | 3.0 MB / 517.5 KB |
| verdict | **PASS** (expected=7, got=7) |

¹ PKE public key only — see README §6.3; evaluation and rotation keys are
regenerated server-side rather than shipped.

### Against an OpenFHE run on the same host

Measured on the earlier harness revision, whose small instance was 15 samples
rather than 100, so compare **per-inference** figures rather than per-instance
totals:

| | CROSS / TPU v6e ×8 | OpenFHE / CPU (180 vCPU) | speedup |
|---|---:|---:|---:|
| per inference, batch 1 | 147.7 ms | 7950 ms | **54×** |
| per inference, 8-chip throughput | **19.24 ms** | 7950 ms | **413×** |
| inferences / second | **51.97** | 0.126 | **413×** |

Read with README §6.1 in mind: the reference ran at `HEStd_NotSet`, degree
2048, against this submission's 128-bit, degree 32768.

## 5. Offline (one-off) server cost

Everything below is paid once, in benchmark stage 3, and is what
`server_preprocess_model` measures.

| | value |
|---|---:|
| Mapping build, 1 / 2 / 4 / 8 chips | 439 / 464 / 465 / 467 s |
| stage 3 wall clock (8 chips, in-harness) | 475–485 s |
| rotation keys | 134 |
| evaluation key | 12.0 MiB |
| rotation keys, total | 1.57 GiB |
| MatVec constants (BSGS diagonals) | 1.56 GiB |
| logical live memory | 3.17 GiB |
| XLA captured constants | 5.44 GB |
| HBM in use, busiest chip | 9.31 GiB of 31.2 GiB |

BSGS plans chosen by Mapping analysis:

| layer | diagonals | giant steps | baby rotations |
|---|---:|---:|---:|
| fc1 (784→128) | 911 | 8 | 127 |
| fc2 (128→64) | 191 | 2 | 127 |
| fc3 (64→10) | 73 | 2 | 72 |

Build time is dominated by XLA compilation, not by diagonal encoding:
`bsgs_n_jobs=16` moved the 8-chip build only from 488 s to 462 s on a 180-core
host. Build time is also essentially flat in device count (439 s at 1 chip to
467 s at 8), which is what one expects when the same program is compiled for a
wider device mesh.

Note the asymmetry this creates for the benchmark's `total latency`: for the
single instance it is ~493 s against the reference's ~19 s, because the harness
charges the whole one-off Mapping build to a single query. Per additional
query that cost is zero — the second and third runs of the single instance
reuse the resident Mapping and spend 0.26 s in stage 7.

## 6. Correctness

Checked three independent ways.

| Check | Result |
|---|---|
| Harness single-instance verdict | **PASS** (expected=7, got=7), 3/3 runs |
| Harness batch-instance quality | encrypted accuracy **identical to the harness plaintext model** on every batch instance measured; see `measurements/*/results-*.json` |
| Decrypted logits vs cleartext torch model | max abs error **2.6e-12** |
| CKKS encrypt→decrypt roundtrip (client only) | max abs error 3.5e-13 |
| BatchNorm fold (training → inference model) | max deviation 5.2e-06 |

The profiler re-checks the logits against the cleartext model on the same
inputs before timing any configuration, so every latency number below comes
from a run that was verified correct first.

The HE-friendly model reaches 97.82% MNIST test accuracy; the harness's own
ReLU reference model, trained by the harness during a run, reaches 97.62%.
Replacing ReLU with x² cost nothing measurable on this task.

## 7. Caveats

Read the numbers above with these in mind.

1. **The two parameter sets are not equivalent.** CROSS runs at degree 32768,
   128-bit classical; the reference runs at degree 2048 with `HEStd_NotSet`,
   which is not a secure parameter set. A speedup figure between them is a
   system comparison, not a like-for-like kernel comparison.

2. **Evaluation keys are derived server-side.** CROSS's `Mapping` owns its
   `CKKSContext` and materializes evaluation and rotation keys from the secret
   key during construction, so this submission hands the server process the key
   pair instead of shipping client-generated evaluation keys. That is a real
   deviation from the benchmark's trust model, and it is why the reported
   "public and evaluation keys" size (8.0 MB) is not comparable to the
   reference's 1.4 GB — the reference ships its evaluation keys, this
   submission regenerates them. The equivalent key volume is reported
   separately from `Mapping.estimate_live_memory()`.

3. **Host contention distorts wall-clock measurements.** Early OpenFHE runs on
   this host measured 142 s and 70 s per inference while a 243-thread torch job
   was resident; the same binary measured 7-9 s once the host was idle. Only
   idle-host measurements are reported. Mapping build times taken while the
   harness was training its own reference model are flagged where they occur.

4. **Slot occupancy is low, and could not be exploited.** The ring gives 16384
   slots and the model uses 784. The ring follows from the emitted depth and
   the security target, not from the model's width. CROSS's public API batches
   across *ciphertexts* (`global_batch`), not across slot ranges within one
   ciphertext, so several images cannot be packed into one ciphertext today;
   and the one way to put more work on a chip that the API does offer — a local
   batch above 1 — crashes the XLA compiler here (section 2). The headroom is
   visible but unreachable through either route on this stack.

5. **`ms/inference` at a global batch above 1 is throughput, not latency.** A
   single query still costs one full `Mapping.execute`; batching amortizes the
   cost across independent requests.
