# CROSS on TPU v6e — MNIST ML-inference submission

Encrypted MNIST inference on **eight Google TPU v6e chips**, implemented with
[CROSS](https://github.com/EfficientPPML/CROSS) (`jaxite_word`) — the JAX CKKS
library from *Leveraging ASIC AI Chips for Homomorphic Encryption* (HPCA'26).

| | |
|---|---|
| Scheme | **CKKS** (RNS-HEIB / HYBRID key switching) |
| Security | **128-bit classical**, ring degree 32768 — justified below |
| Acceleration hardware | **8 × Google TPU v6e** (31.2 GiB HBM per chip), one `v6e-8` VM |
| Submission type | Open source, complete implementation in this directory |
| Encrypted inference | **4.3 ms** per image on 8 chips (224 inferences/s) |
| Ring | N = 16384, 128-bit classical (`CROSS_MODEL=shallow`, default) |

---

## 1. Performance

Steady-state `Mapping.execute`, median of 20 iterations after 3 warmups. Every
configuration is verified against the cleartext model on the same inputs before
it is timed.

| chips | 1 | 2 | 4 | **8** |
|---|---:|---:|---:|---:|
| execute (ms) | 30.9 | 33.0 | 34.3 | 35.7 |
| ms / inference | 30.91 | 16.52 | 8.57 | **4.46** |
| inferences / s | 32.35 | 60.55 | 116.65 | **224.01** |
| scaling | 1.00× | 1.87× | 3.61× | **6.92×** |

The deeper `CROSS_MODEL=deep` variant (784-128-64-10, N = 32768) is 4.3×
slower per inference and 0.008 more accurate at the medium instance; see
[docs/PARAMETER_STUDY.md](docs/PARAMETER_STUDY.md) for the full comparison and
for why Lattica-ai's N = 4096 is not reachable from CROSS at any depth.

Strong scaling is near-linear to all eight chips. Per-execute latency barely
moves from 1 to 8 chips because each chip evaluates an independent ciphertext
and only dispatch grows; the 20 samples at 8 chips span 153.8–154.2 ms
(stdev 0.13 ms).

`docs/RESULTS.md` has the full measurements, the per-stage comparison against
an OpenFHE run on the same host, and the profiler breakdown of where the
accelerator time goes.

---

## 2. Why this is 128-bit secure

**The ring is derived, not chosen.** Nothing in this submission picks a
security parameter. `packing.pack` computes the program's slot demand and the
multiplicative depth it actually emits, then asks CROSS's `he_params` for the
smallest supported ring meeting a **128-bit classical** target. The submission
then samples its key pair at *that* ring's own error width. The resulting
parameter set is:

| parameter | value |
|---|---|
| scheme | CKKS |
| ring degree `N` | **16384** (`shallow`); 32768 (`deep`) |
| slots | 8192 = `N/2` (`shallow`); 16384 (`deep`) |
| ciphertext modulus `Q` | 8 towers, **log2(Q) = 241.0** (`shallow`) |
| key-switching modulus `P` | 4 towers, **log2(QP) = 365.0** (`shallow`) |
| key-switching | HYBRID, `dnum = 3` |
| secret distribution | ternary uniform |
| error distribution | discrete Gaussian, `sigma = 3.19` |
| multiplicative depth used | 3 (`shallow`); 5 (`deep`) |
| claimed security | **128-bit classical** |

**Why the claim holds.** The security of RLWE at a given dimension is bounded
by the total modulus the adversary sees, which for HYBRID key switching is
`QP`, not `Q` alone. Here:

```
shallow:  N = 16384,  log2(Q) = 241.0,  log2(QP) = 365.0,  ternary,  sigma = 3.19
deep:     N = 32768,  log2(Q) = 361.0,  log2(QP) = 485.0,  ternary,  sigma = 3.19
```

The Homomorphic Encryption Standard's `HEStd_128_classic` table permits
`log2(Q) ≤ 438` at `N = 16384` and `≤ 881` at `N = 32768` for a uniform ternary
secret. `shallow` uses **365 of 438 bits (a 73-bit margin)** and `deep`
**485 of 881 (a 396-bit margin)**, so both sit inside the 128-bit region, and likewise for the Albrecht et al. LWE
estimator at `n = 32768`. Reproduce the numbers with:

```console
python3 -c "
import sys, math; sys.path.insert(0,'submissions/mnist/src')
import cross_task as ct
rc = ct.build_client_packed().ring_config
Q = math.prod(rc.q_towers); P = math.prod(rc.p_towers)
print('N', rc.degree, 'log2(Q) %.1f' % math.log2(Q), 'log2(QP) %.1f' % math.log2(Q*P),
      'sigma', rc.sigma, rc.target)"
``` The submission asserts the target at runtime rather than
trusting a comment: `ring_config.target.bits` is checked to be `>= 128` by
`src/selftest.py`, and the ring, tower counts and depth are re-verified on both
the client and the server before any ciphertext is produced
(`cross_task.verify_against_manifest`).

**CKKS-specific caveat.** CKKS is an approximate scheme, and the Li–Micciancio
passive attack applies to any deployment that hands decrypted approximate
results to a party that does not hold the secret key. Here decryption happens
only on the client, which already holds the plaintext input, so the attack
surface does not arise in this benchmark's threat model. A deployment that
returned decrypted values to the evaluator would need noise flooding.

**What is NOT claimed.** See the deviation in §6: evaluation and rotation keys
are generated inside the server process from the secret key, rather than
shipped by the client. That is a trust-model deviation, not a parameter-level
weakness — the parameters above are unaffected.

---

## 3. How the solution works

```
shallow (default):  784 ──Linear(784,50)──► x² ──Linear(50,10)──► 10 logits
deep:               784 ──Linear(784,128)──► x² ──Linear(128,64)──► x² ──Linear(64,10)──► 10 logits
```

`shallow` mirrors the topology Lattica-ai's submission uses. It emits depth 3,
which is what places it on `N = 16384`; the harness topology emits depth 5 and
needs `N = 32768`. Under CROSS the architecture *is* the parameter choice.

One ciphertext carries one image, packed into the 16384-slot vector. Each
`Linear` is a CKKS matrix–vector product evaluated with the baby-step/giant-step
(Halevi–Shoup) diagonal method; CROSS's `Mapping` chooses the BSGS split per
layer from the diagonals that layer actually has:

| layer | diagonals | giant steps | baby rotations |
|---|---:|---:|---:|
| fc1 (784→128) | 911 | 8 | 127 |
| fc2 (128→64) | 191 | 2 | 127 |
| fc3 (64→10) | 73 | 2 | 72 |

Each square is one ciphertext–ciphertext multiplication with relinearization
and rescale. Total multiplicative depth 5, no bootstrapping.

The pipeline is the ordinary CROSS deployment path — the model enters once as a
`torch.nn.Module` and is lowered by the library, not by hand:

```
torch.nn.Module ──nn.vectorize──► VectorizedProgram ──packing.pack──► PP-op DAG + RingConfig ──Mapping──► compiled TPU executable
```

### Model architecture change: ReLU → x²

The harness model (`harness/mnist/model.py`) is `784 → 128 → 64 → 10` with
ReLU. **The only change is ReLU → x².** Topology, layer widths and the
input/output contract are unchanged.

CKKS evaluates polynomials, not ReLU, so every FHE submission substitutes
something. CROSS's activation registry accepts only a bare `x**(2**k)` squaring
chain: reaching any other exponent needs one operand mod-switched down to meet
the other, and the only level-changing primitive is Rescale, which also divides
the scale.

A bare square is hard to train directly, so `model/he_mlp.py` carries a
`BatchNorm1d` before each square while the statistics are still moving, and
`to_inference_model` folds each one into the preceding `Linear` at export.
`BN(Wx+b)` is itself affine, so the fold is an exact algebraic rewrite;
`model/export_weights.py` re-checks it and refuses to write a model whose
function moved by more than `1e-4` (observed: 2.5e-07).

Accuracy on the MNIST test set: **97.39%** (`shallow`) and **97.82%**
(`deep`), against **97.62%** for the harness's own ReLU reference model. On the
benchmark's own samples the encrypted `shallow` model scores 0.96 (small) and
0.976 (medium) — ahead of Lattica-ai's 0.94 / 0.972, and just under the harness
plaintext model's 0.97 / 0.982. `deep` scores 0.99 / 0.984, the highest on the
board. See [docs/PARAMETER_STUDY.md](docs/PARAMETER_STUDY.md).

### Pre- and post-processing outside the encryption (documented per the rules)

- **Before encryption (client, stage 5):** the MNIST normalization
  `(x - 0.1307) / 0.3081`. The harness exports raw `ToTensor` pixels in
  `[0, 1]`, and the harness's own predictor applies exactly this transform in
  `harness/mnist/test.py`. It is affine and free in the clear, so it does not
  spend ciphertext depth.
- **After decryption (client, stage 9):** `argmax` over the ten decrypted
  logits to produce a label.

**The entire model — both matrix products and both activations — is evaluated
on encrypted data.** Nothing of the network is computed in the clear.

---

## 4. Running it

See **[docs/EVALUATION.md](docs/EVALUATION.md)** for the full guide with
expected output, timings and troubleshooting. Short form:

```console
# 1. CROSS, which is used in place (not packaged)
git clone https://github.com/EfficientPPML/CROSS.git ~/CROSS
export CROSS_ROOT=~/CROSS        # optional; ../CROSS and ~/CROSS are searched too

# 2. Python environment
conda create -y --name jaxite python=3.13 && conda activate jaxite
pip install -U "jax[tpu]" absl-py numpy
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. Verify without an accelerator (12 checks, ~90 s)
python3 submissions/mnist/src/selftest.py

# 4. Run the benchmark
python3 harness/run_submission.py 0 --seed 3 --num_runs 3   # single
python3 harness/run_submission.py 1 --seed 3 --num_runs 3   # small  (100)
python3 harness/run_submission.py 2 --seed 3 --num_runs 3   # medium (1000)

# 5. Release the TPU (the server stays resident so repeat runs do not rebuild)
python3 submissions/mnist/server_stop.py
```

| variable | default | meaning |
|---|---|---|
| `CROSS_ROOT` | autodetect | CROSS checkout providing `jaxite_word` |
| `CROSS_DEVICE_COUNT` | 8 | TPU chips; also the global ciphertext batch |
| `CROSS_BSGS_JOBS` | 1 | workers encoding BSGS diagonals (offline only) |
| `CROSS_STARTUP_TIMEOUT` | 5400 | seconds to wait for stage 3 |
| `CROSS_REQUEST_TIMEOUT` | 14400 | seconds to wait for stage 7 |
| `CROSS_DAEMON_IDLE_TIMEOUT` | 3600 | idle seconds before the server exits |

### Hardware

A Google Cloud TPU v6e VM. The measurements here are from a `v6e-8`
(8 chips). It runs on fewer chips by setting `CROSS_DEVICE_COUNT`; the global
ciphertext batch always equals the chip count (see §6, point 4).

No remote backend is involved: the accelerator is local to the machine running
the harness, and the whole implementation is in this directory.

---

## 5. Repository changes

- `submissions/mnist/` — replaced with this implementation.
- `scripts/get_openfhe.sh`, `scripts/build_task.sh` — replaced, as the
  submission instructions anticipate. CROSS links no OpenFHE and compiles
  nothing, so the OpenFHE fetch and the CMake/LibTorch build are not needed.
  Restore both from upstream `main` to run the reference submission again.
- **`harness/` is untouched.**

---

## 6. Honest limitations

1. **The comparison against the OpenFHE reference is not like-for-like.** That
   reference runs at `SetSecurityLevel(HEStd_NotSet)` with
   `SetRingDim(1 << 11)`, which is not a secure parameter set. This submission
   does 16× the ring work per ciphertext at a real 128-bit level. Any speedup
   quoted against it is a deployed-system comparison, not kernel-for-kernel.
2. **The acceleration is server-side only.** CROSS's CPU-side client codec
   encrypts at ~1.8 s and decrypts at ~0.6 s for a single sample (~0.38 s and
   ~0.13 s per sample amortized over a batch, as JIT warmup is paid once)
   against OpenFHE's ~16 ms and ~21 ms. Part is the 16× larger ring; the rest
   is that CROSS's codec is JAX on CPU and is not an optimized client path.
   For a single-image instance this dominates the end-to-end time.
3. **Evaluation keys are derived server-side — a trust-model deviation.**
   CROSS's `Mapping` owns its `CKKSContext` and materializes evaluation and
   rotation keys from the secret key during construction, so this submission
   hands the server process the key pair instead of shipping client-generated
   evaluation keys. It is forced by the current CROSS API, and it is why the
   reported public-key size (8.0 MB, the PKE public key alone) is not
   comparable to a submission that ships ~1.4 GB of evaluation keys. The
   equivalent key volume here is 12.0 MiB of evaluation key plus 1.57 GiB of
   rotation keys, reported by `Mapping.estimate_live_memory()`.
4. **`global_batch` must equal the chip count.** More than one ciphertext per
   chip aborts the XLA TPU fusion emitter on this stack
   (`Check failed: window.strides[...] == current_scaled_dim_window_bound`),
   reproduced twice. So the ring's 16384 slots against a model using 784 is
   visible headroom that neither slot-packing nor per-chip batching reaches
   here.
5. **Offline cost is large but paid once.** Materializing the `Mapping` — BSGS
   diagonals, keys, constants, XLA compilation — takes ~470 s and is charged to
   stage 3. Per additional query it is zero.

---

## 7. Layout

```
submissions/mnist/
├─ README.md                    this file
├─ client_*.py, server_*.py     the harness stages
├─ src/
│  ├─ cross_task.py             shared: packing, keys, ciphertext I/O, paths
│  ├─ server_daemon.py          the resident TPU server
│  ├─ selftest.py               12 checks, no accelerator required
│  └─ profile_tpu.py, trace_breakdown.py, report.py
├─ model/                       HE-friendly model, training and export
└─ docs/
   ├─ EVALUATION.md             step-by-step reproduction
   └─ RESULTS.md                full measurements
```

## License

Apache v2, as the harness. CROSS is MIT.
