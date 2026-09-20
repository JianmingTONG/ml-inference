# Can CROSS run at Lattica-ai's N = 4096?

**Yes — at depth 1.** CROSS runs on ring degree 4096 at 128-bit classical
security, verified on TPU v6e, at **1.127 ms per inference on 8 chips**. What
does not fit at that ring is a *multi-layer* network: only a single linear
layer. This note records how to get there and what it costs.

Two earlier revisions of this note claimed N = 4096 was unreachable. That was
wrong, and the error is instructive: the first test only varied
`composite_degree` (which CROSS rejects) and the second only varied
`scaling_mod_size`. Neither touched `register_word_size`, which is what sets
the auxiliary modulus `P`.

## What Lattica-ai uses

From `submission_remote/mlp/submission_params_and_model/README_Lattica_submission.md`
in [Lattica-ai/ml-inference](https://github.com/Lattica-ai/ml-inference):

```python
"full_q_list_precision": ((61,), (45,)),   # modulus chain: ~106 bits total
"n": 2 ** 12,                              # ring degree 4096
"err_std": 3.19,
"sk_hw": 0,                                # uniform ternary secret
"g_base_bits": 4,                          # gadget decomposition in the eval key
"pt_scale": 2 ** 20,
```

106 bits at N = 4096, against the 109-bit ceiling the Homomorphic Encryption
Standard gives for a uniform ternary secret at 128-bit classical security. A
3-bit margin, and no separate `P`: with gadget-decomposition key switching the
security modulus is just `Q`.

## How to reach N = 4096

Three settings have to move together, all through the public
`packing.PackingPolicy` -- no library change is required:

```python
packing.PackingPolicy(
    register_word_size=19,   # => aux_mod_size ~20 bits instead of 31
    scaling_mod_size=34,     # smallest the N=4096 prime search sustains
    first_mod_size=35,
)
# on a depth-1 program (one matvec, num_q = 2)
```

which yields:

| | value |
|---|---|
| ring degree | **4096** |
| Q towers | 4 — `(188417, 40961, 65537, 114689)`, 16–18 bits |
| P towers | 2 × ~20 bits |
| **log2(QP)** | **100.1** (ceiling at N=4096 is 109) |
| `he_params.qp_is_secure` | **True**, 128-bit classical |

`register_word_size` was the missing lever. `aux_mod_size` is defined as
`register_word_size - 1`, so at the default 32 the P towers are 31 bits each
and alone push QP to 127.7 — past the ceiling — no matter how small the Q
primes get. Dropping it to 19 puts P at ~20 bits and brings QP to 100.1.

## Why only depth 1 fits there

CROSS targets the TPU, whose integer path is 32-bit. Every native modulus must
stay below `2^31`, so one 60-bit CKKS scale is stored as **two ~30-bit RNS
primes**, and `composite_degree = 2` is a hard invariant — `he_params` raises
`"CROSS secure profiles require composite_degree=2"` for anything else.

That fixes the cost of a level at ~60 bits and makes the depth ladder coarse.
Measured by packing real models through `packing.pack`:

| program | depth | `N` | Q towers | log2(QP) | ceiling at that `N` |
|---|---:|---:|---:|---:|---:|
| one matvec, nothing else | 1 | 8192 | 4 | 183 | 218 |
| matvec + square | 2 | 16384 | 6 | 243 | 438 |
| **784-50-10, one square** (Lattica topology) | 3 | **16384** | 8 | 365 | 438 |
| **784-128-64-10, two squares** (harness topology) | 5 | **32768** | 12 | 485 | 881 |

At the *default* 32-bit register word, even a single matvec needs 183 bits —
74 bits past the 109-bit ceiling. That is what the earlier revisions measured
and over-generalized from.

That table used the default 60-bit scale, so the obvious next question is
whether a smaller scale gets there. It does not. Sweeping the *entire*
specification space that `he_params.generate_ring_config` accepts —
`num_q` 1..4, `scaling_mod_size` 16..60, every valid `dnum`, and slot demands
from 16 to 784 — the only degree at or below 8192 it will ever emit is **8192**:

```
degrees reachable at <=8192: [8192]

     N  num_q  smod  dnum  slots_req   Qp   Pp  log2(QP)
  8192      2    36     2         16    4    2     129.3   <- smallest ring CROSS can emit
```

Asking for anything smaller returns

```
no tabulated degree in (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
satisfies p at 128-bit classical security
```

The binding constraints compound: `composite_degree = 2` forces at least four
Q primes plus two P primes for even one level, and an NTT at degree `N` needs
primes congruent to 1 mod `2N`, which puts a floor under how small those six
primes can be. Six primes at that floor is ~129 bits — still 20 bits past the
109-bit ceiling at N = 4096, with nothing left to trim.

With `register_word_size = 19` the picture changes: depth 1 fits (100.1 bits),
depth 2 does not. At depth 2 the Q chain alone is ~100 bits, leaving nothing
for P under a 109-bit ceiling; the N=4096 prime search then fails outright
("Underflow in PreviousPrime") when asked for a smaller scale.

So the boundary at Lattica's ring is **depth 1 for CROSS, depth 3 for
Lattica** -- one linear layer versus a two-layer network with a square.
CROSS stores one logical scale as two RNS primes (`composite_degree = 2`) and
needs NTT primes congruent to 1 mod 2N, so depth 1 already costs four Q
primes. Lattica needs two primes for their whole chain because 61- and 45-bit
moduli are native 64-bit arithmetic. That is the price of 32-bit lanes.

Lattica fits 106 bits because 61- and 45-bit moduli are native 64-bit
arithmetic on a GPU. That is the trade: CROSS accepts a larger ring to get
32-bit lanes the TPU's matrix unit can actually use.

Lowering `scaling_mod_size` does not help either, because `composite_degree=1`
is rejected outright; and relaxing that guard would change the scale semantics
the evaluator depends on (`_default_matvec_plaintext_scale` treats CD1
differently), so it is not a parameter to turn casually.

## The full ladder, measured

| ring | depth | architecture that fits | test accuracy | ms/inference, 8 chips | `CROSS_MODEL` |
|---:|---:|---|---:|---:|---|
| **4096** | 1 | `Linear(784,10)` | 92.33% | **1.127** | `linear` |
| 8192 | 2 | `x² → Linear(784,10)` | 91.6% | not measured | — |
| **16384** | 3 | **784-50-10, one square** | **97.39%** | **4.46** | `shallow` (default) |
| 32768 | 5 | 784-128-64-10, two squares | 97.82% | 19.24 | `deep` |

Measured on 8 × TPU v6e, median of 20 iterations, each configuration checked
against the cleartext model first. Raw JSON under `results/tpu_profile*/`.

N = 8192 is genuinely reachable — `square → Linear` packs there at
`scaling_mod_size` 40 or 50, with 6 Q towers and log2(QP) = 182. But depth 2
cannot express `Linear → Square → Linear`: the only classifier that fits is a
linear layer on squared pixels, which tops out at **91.6%** after 12 epochs.
That is roughly six points below every entry on the leaderboard (Lattica-ai
0.972, OpenFHE reference 0.974 at the medium instance), so it trades far more
accuracy than the ring is worth.

**N = 16384 is therefore the smallest ring that still runs a real two-layer
network**, which is why this submission stops there.

## What is reachable, and what it buys

The lever CROSS does respond to is **program depth**, which means the
architecture is the parameter choice. Adopting Lattica's shallower topology —
`784 → 50 → 10` with a single square — halves the ring:

| | deep (`CROSS_MODEL=deep`) | shallow (`CROSS_MODEL=shallow`) |
|---|---:|---:|
| topology | 784-128-64-10, 2 squares | 784-50-10, 1 square |
| depth | 5 | 3 |
| ring degree | 32768 | **16384** |
| Q towers | 12 | **8** |
| log2(QP) | 485 | 365 |
| ciphertext | 3.0 MiB | **1.0 MiB** |
| PP-ops | 8 | 5 |
| MNIST test accuracy | 97.82% | 97.39% |

Both are 128-bit classical. Select the architecture with `CROSS_MODEL`.

## Measured effect of the ring reduction

Steady-state `Mapping.execute`, median of 20 iterations after 3 warmups, every
configuration checked against the cleartext model before timing. Raw JSON in
`results/tpu_profile_shallow/profile_results.json` and
`results/tpu_profile/profile_results.json`.

| chips | deep, N=32768 | shallow, N=16384 | speedup |
|---:|---:|---:|---:|
| 1 | 147.71 ms | **30.91 ms** | 4.78× |
| 2 | 75.59 ms | **16.52 ms** | 4.58× |
| 4 | 38.01 ms | **8.57 ms** | 4.44× |
| 8 | 19.24 ms | **4.46 ms** | **4.31×** |

At 8 chips that is **224.0 inferences/s**, against 51.97 for the deep model.
Secondary effects, all in the same direction:

| | deep | shallow |
|---|---:|---:|
| Mapping build | 439–467 s | **122–131 s** |
| rotation keys | 134 | **76** |
| HBM, busiest chip | 9.31 GiB | **1.70 GiB** |
| ciphertext | 3.0 MiB | **1.0 MiB** |
| max logit error vs cleartext | 5.2e-12 | 2.1e-12 |
| MNIST test accuracy | 97.82% | 97.39% |

Halving the ring roughly quartered the per-inference cost: the ring is 2×
smaller *and* carries 8 Q towers instead of 12, and the program emits 5 PP-ops
instead of 8. The cost is 0.43 points of plaintext accuracy.

## Where that leaves the comparison with Lattica-ai

Per-image accelerator time, from the published leaderboard and from the runs
above:

| | per image |
|---|---:|
| Lattica-ai, medium instance (1000 images) | 0.215 ms |
| Lattica-ai, small instance (100 images) | 2.16 ms |
| **CROSS/TPU shallow, 8 chips, batch 8** | **4.46 ms** |
| CROSS/TPU deep, 8 chips, batch 8 | 19.24 ms |
| Reference OpenFHE (CPU), medium | 6390 ms |

The shallow model closes most of the gap at comparable batch: 2.1× behind
Lattica's small instance, down from 9.2×. The remaining gap at their medium
instance is not ring size but **packing**: Lattica puts the batch on the ring
axis (784 ciphertexts each holding up to 512 images), so their per-image cost
keeps falling with batch size, while CROSS carries one image per ciphertext and
stays flat. Their small and medium rows report the same 98 MB of input and the
same ~215 ms of compute for 10× the images, which is that effect.


## End-to-end re-evaluation, unmodified harness, `--num_runs 3`

Both architectures measured the same way, on the same 8 × TPU v6e. `deep` rows
are preserved under `results/measurements_deep/`; the committed
`measurements/` are the `shallow` runs.

| instance | | deep N=32768 | shallow N=16384 | change |
|---|---|---:|---:|---|
| single | TPU per inference | 153.7 ms | **34.4 ms** | 4.5× faster |
| | harness total | 500.7 s | **158.0 s** | 3.2× faster |
| | verdict | PASS | **PASS** | — |
| small (100) | TPU per inference | 19.9 ms | **4.5 ms** | 4.4× faster |
| | stage 7 | 2.86 s | **1.04 s** | |
| | harness total | 552.3 s | **188.7 s** | 2.9× faster |
| | encrypted accuracy | 0.99 | **0.96** | −0.03 |
| medium (1000) | TPU per inference | 19.1 ms | **4.3 ms** | 4.4× faster |
| | stage 7 | 26.72 s | **9.32 s** | |
| | harness total | 917.0 s | **331.7 s** | 2.8× faster |
| | encrypted accuracy | 0.984 | **0.976** | −0.008 |
| | keys / input / result | 8.0 M / 2.9 G / 501.5 M | **3.0 M / 1001.6 M / 251.5 M** | ~3× smaller |

The harness plaintext model scores 0.97 (small) and 0.982 (medium) on the same
samples, so `shallow` sits just below it where `deep` sat just above.

### Which to submit

| | deep | shallow |
|---|---|---|
| per-inference latency | 19.1 ms | **4.3 ms** |
| accuracy, medium | **0.984** | 0.976 |
| accuracy vs Lattica-ai (0.972) | +0.012 | +0.004 |
| accuracy vs OpenFHE reference (0.974) | +0.010 | +0.002 |

`shallow` is 4.4× faster and still ahead of both other leaderboard entries on
accuracy; `deep` is the most accurate submission on the board. Select with
`CROSS_MODEL=deep` or `CROSS_MODEL=shallow`; the committed measurements are
`shallow`.


## N = 4096 measured on TPU v6e

`CROSS_MODEL=linear`, 128-bit classical, log2(QP) = 100.1:

| chips | build | execute | ms/inference | inferences/s |
|---:|---:|---:|---:|---:|
| 1 | 26 s | 7.36 ms | 7.358 | 135.9 |
| 2 | 29 s | 8.00 ms | 4.000 | 250.0 |
| 4 | 29 s | 8.03 ms | 2.009 | 497.9 |
| **8** | 29 s | 9.02 ms | **1.127** | **887.4** |

HBM on the busiest chip is 0.299 GiB and there are 56 rotation keys, against
1.70 GiB / 76 keys at N = 16384 and 9.31 GiB / 134 at N = 32768. The Mapping
builds in 26–29 s rather than 122–131 s.

**Precision degrades, and it is visible.** Max absolute error on the decrypted
logits is **~2e-03**, against 2.1e-12 at N = 16384 — a 100-bit modulus with an
~18-bit scale has far less room. Labels still matched the cleartext model on
every configuration measured (1/1, 2/2, 4/4, 8/8), which is what argmax
classification needs, but this parameter set has little margin and should not
be assumed safe for a task that consumes the logit values themselves.

### Against Lattica-ai at the same ring

| | per image | model | MNIST accuracy |
|---|---:|---|---:|
| **CROSS/TPU, 8 × v6e, N=4096** | **1.127 ms** | `Linear(784,10)` | 92.33% |
| Lattica-ai, small instance, N=4096 | 2.16 ms | 784-50-10 + square | ~0.94–0.97 |
| Lattica-ai, medium instance, N=4096 | 0.215 ms | same | same |

At the same ring degree CROSS on eight TPU v6e chips is **1.9× faster per
image than Lattica's small-instance figure**, and 5.2× slower than their
medium-instance figure, which amortizes over batch-axis packing CROSS does not
have. The comparison is not like-for-like on capability: they run a two-layer
network at that ring and CROSS runs a linear classifier, which is five
accuracy points worse. Reaching Lattica's ring *and* their model capability
would need 64-bit moduli in CROSS, not a parameter change.


### N = 4096 end to end, unmodified harness, `--num_runs 3`

Saved under `results/measurements_4096/`. The committed `measurements/` remain
the `shallow` runs, which is what this submission claims.

| instance | TPU / inference | stage 7 | harness total | encrypted accuracy | keys | input |
|---|---:|---:|---:|---:|---:|---:|
| single | 9.4 ms | 0.108 s | 53.6 s | PASS | 388.1 K | 133.5 K |
| small (100) | **1.2 ms** | 0.531 s | 67.8 s | 0.88 | 388.1 K | 12.7 M |
| medium (1000) | **1.1 ms** | 4.52 s | 101.9 s | 0.92 | 388.1 K | 126.5 M |

Key material drops to 388 KB and the medium instance's ciphertext upload from
1001.6 MB to 126.5 MB. Harness total for the medium instance falls from
331.7 s (`shallow`) to 101.9 s.

The accuracy is the catch: **0.88 and 0.92**, against the harness plaintext
model's 0.97 and 0.982, and against Lattica-ai's 0.94 and 0.972 *at the same
ring*. A linear classifier is all that fits at N = 4096 under 32-bit lanes.

## Which variant to submit

| | `linear` N=4096 | `shallow` N=16384 | `deep` N=32768 |
|---|---:|---:|---:|
| TPU per inference, 8 chips | **1.1 ms** | 4.3 ms | 19.1 ms |
| encrypted accuracy, medium | 0.92 | **0.976** | **0.984** |
| harness total, medium | **101.9 s** | 331.7 s | 917.0 s |
| vs Lattica-ai accuracy (0.972) | −0.052 | +0.004 | +0.012 |
| vs OpenFHE reference (0.974) | −0.054 | +0.002 | +0.010 |

`linear` is the fastest and matches Lattica's ring exactly, but it is the only
variant that would rank *below* both existing leaderboard entries on accuracy.
`shallow` stays the submission: it is 4.4× faster than the original and still
the most accurate entry bar `deep`.

---

## Can the other variants use a smaller ring?

Two questions, answered by sweeping `scaling_mod_size` and `register_word_size`
through `he_params.generate_ring_config` (the cheap part that picks the ring —
calling `packing.pack` per configuration does full BSGS planning and is far too
slow for a sweep).

### shallow: 16384 → 8192? **No.**

| `rws` | `smod` | N | Qp | Pp | log2(QP) |
|---:|---:|---:|---:|---:|---:|
| 32 | 60 | 16384 | 8 | 4 | 365.0 |
| 32 | 44 | 16384 | 8 | 3 | 267.7 |
| 32 | 42 | 16384 | 8 | 3 | **256.2** ← floor |
| 32 | 40 | — | | | `ParameterGenerationError` |
| 24 | 42 | 16384 | 8 | 4 | 254.8 |
| 24 | 40 | — | | | `ParameterGenerationError` |

The ceiling at N = 8192 is 218 bits and the floor reachable here is **254.8** —
a 37-bit gap. Below `scaling_mod_size = 42` the composite-prime search fails
outright at every `register_word_size` tried (32, 28, 24), so there is nothing
left to trim. Depth 3 costs 8 Q towers, and 8 towers will not fit under 218
bits at the prime sizes N = 8192 admits.

### deep: 32768 → 16384? **Yes.**

`scaling_mod_size = 44` instead of 60 is enough on its own — `register_word_size`
does not need to move:

| | deep | **deep16k** |
|---|---:|---:|
| ring degree | 32768 | **16384** |
| Q / P towers | 12 / 4 | 12 / 3 |
| log2(QP) | 485.0 | **353.3** (ceiling 438) |
| TPU per inference, 8 chips | 19.243 ms | **7.216 ms** |
| inferences/s | 52.0 | **138.6** |
| max logit error | 5.2e-12 | 3.0e-07 |
| **encrypted accuracy, medium** | **0.984** | **0.984** |
| stage 7, medium | 26.72 s | **12.10 s** |
| harness total, medium | 917.0 s | **427.1 s** |

**2.7× faster with identical accuracy**, on the same weights. The logit error
rises from 5.2e-12 to 3.0e-07 — still five orders of magnitude better than the
N = 4096 configuration, and labels matched cleartext on every run. `deep16k`
strictly dominates `deep`; there is no reason to run the 32768 ring.

### A smaller free win: shallow42

`shallow` at `scaling_mod_size = 42` stays on N = 16384 but drops from 4 P
towers to 3, log2(QP) 365 → 256: **4.349 ms** against 4.464, a 2.5% gain at the
same accuracy.

### Updated ladder

| variant | ring | log2(QP) | ms/inference | accuracy (medium) |
|---|---:|---:|---:|---:|
| `linear` | 4096 | 100.1 | **1.127** | 0.92 |
| `shallow42` | 16384 | 256.2 | 4.349 | 0.976 |
| `shallow` | 16384 | 365.0 | 4.464 | 0.976 |
| `deep16k` | 16384 | 353.3 | 7.216 | **0.984** |
| `deep` | 32768 | 485.0 | 19.243 | 0.984 | *(superseded by `deep16k`)* |

Against the leaderboard, `deep16k` is the most accurate entry (0.984, versus
Lattica-ai 0.972 and the OpenFHE reference 0.974) and now costs only 1.7× the
latency of `shallow` rather than 4.3×.
