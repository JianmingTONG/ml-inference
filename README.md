# FHE Benchmarking Suite - ML Inference

> ## This fork: CROSS on Google TPU v6e
>
> This fork replaces `submissions/mnist/` with a **CKKS submission running on
> eight Google TPU v6e chips**, implemented with
> [CROSS](https://github.com/EfficientPPML/CROSS) (`jaxite_word`) from
> *Leveraging ASIC AI Chips for Homomorphic Encryption* (HPCA'26).
>
> | | |
> |---|---|
> | **Encrypted inference** | **4.3 ms per image** on 8 × TPU v6e (224 inferences/s) |
> | **Scaling** | 6.92× from 1 to 8 chips |
> | **Accuracy** | 0.976 at the medium instance, ahead of both other leaderboard entries (0.972, 0.974) |
> | **Security** | **128-bit classical**, CKKS at ring degree 16384 (`log2(QP) = 365` of the 438 bits the HE Standard permits at N=16384) |
> | **Acceleration hardware** | one Google Cloud `v6e-8` VM: 8 × TPU v6e, 31.2 GiB HBM each |
> | **Submission type** | open source; complete implementation in `submissions/mnist/` |
>
> **Read [`submissions/mnist/README.md`](submissions/mnist/README.md)** for the
> scheme and parameters, the security justification, the one model-architecture
> change (ReLU → x²), the documented pre/post-processing, and the honest
> limitations. Reproduction steps are in
> [`submissions/mnist/docs/EVALUATION.md`](submissions/mnist/docs/EVALUATION.md);
> full measurements in
> [`submissions/mnist/docs/RESULTS.md`](submissions/mnist/docs/RESULTS.md).
>
> Measurements committed under `measurements/{single,small,medium}/` were
> produced by the **unmodified harness** with `--num_runs 3`.
>
> Changed in this fork: `submissions/mnist/` (replaced),
> `scripts/build_task.sh` and `scripts/get_openfhe.sh` (replaced — CROSS links
> no OpenFHE and compiles nothing). **`harness/` is untouched.**
>
> Quick start:
> ```console
> git clone https://github.com/EfficientPPML/CROSS.git ~/CROSS
> conda create -y --name jaxite python=3.13 && conda activate jaxite
> pip install -U "jax[tpu]" absl-py numpy
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
>
> python3 submissions/mnist/src/selftest.py                  # 12 checks, no TPU needed
> python3 harness/run_submission.py 1 --seed 3 --num_runs 3   # small instance
> python3 submissions/mnist/server_stop.py                    # release the TPU
> ```

This repository contains the harness for the ML-inference workload of the FHE benchmarking suite of [HomomorphicEncryption.org](https://www.HomomorphicEncryption.org).
The harness currently supports mnist model benchmarking as specified in `harness/mnist` directory.
The `main` branch contains a reference implementation of this workload, under the `submissions` subdirectory.
The harness also supports an optional *remote backend execution mode* under the `submission_remote` subdirectory, where the homomorphic evaluation is executed on a remote backend.

Submitters should clone this repository and add their content as a subdirectory within the `submissions` directory, or replaced the content of `submission_remote` subdirectory by their own implementation.
They also may need to changes or replace the script `scripts/build_task.sh` to account for dependencies and build environment for their submission.
Submitters are expected to document any changes made to the model architecture `harness/mnist/mnist.py` in the `submissions/[--model]/README.md` file. Submitters have the option to generate an `io/server_reported_steps.json` file, which contains fine grained metrics reported by the server in addition to the metrics reported by the harness.

## Execution Modes

The ML Inference benchmark supports two execution modes:

### Local Execution (Default)

All steps are executed on a single machine:
- Cryptographic context setup and model preprocessing
- Key generation
- Input preprocessing and encryption
- Homomorphic inference
- Decryption and postprocessing

This corresponds to every reference submission in `submissions/`.

### Remote Backend Execution (Optional)

Some FHE deployments separate client-side and server-side responsibilities.  
In this mode:

- **Client-side (local):**
  - Key generation
  - Input preprocessing and encryption
  - Decryption and postprocessing

- **Server-side (remote):**
  - Cryptographic context setup and model preprocessing
  - Homomorphic inference

This execution mode is enabled by passing the `--remote` flag to the harness.

## Running the ML-inference workload

#### Dependencies
- Python 3.12+
- The build environment for local execution depends on OpenFHE being installed as specified in `scripts/get_openfhe.sh` and `submission/CMakeLists.txt`. See https://github.com/openfheorg/openfhe-development#installation.
- The build environment for remote-backend execution depends on lattica-query being installed as specified in `submission_remote/requirements.txt`. See https://platformdocs.lattica.ai/how-to-guides/client-installation/how-to-install-query-client. Should be installed on a `linux_x86_64` machine.

#### Execution
To run the workload, clone and install dependencies:
```console
git clone https://github.com/fhe-benchmarking/ml-inference.git
cd ml-inference

python -m venv bmenv
source ./bmenv/bin/activate
pip install -r requirements.txt

python3 harness/run_submission.py -h  # Information about command-line options
```

The harness script `harness/run_submission.py` will attempt to build the submission itself, if it is not already built. If already built, it will use the same project without re-building it (unless the code has changed). An example run is provided below.


```console
$ python3 harness/run_submission.py -h
usage: run_submission.py [-h] [--num_runs NUM_RUNS] [--seed SEED] [--clrtxt CLRTXT] [--remote] {0,1,2,3} [--dataset] {mnist} [--model] {mlp}

Run ML Inference FHE benchmark.

positional arguments:
  {0,1,2,3}            Instance size (0-single/1-small/2-medium/3-large)

options:
  -h, --help           show this help message and exit
  --num_runs NUM_RUNS  Number of times to run steps 4-9 (default: 1)
  --seed SEED          Random seed for dataset and query generation
  --clrtxt CLRTXT      Specify with 1 if to rerun the cleartext computation
  --remote             Specify if to run in remote-backend mode
  --dataset DATASET    Specify the dataset to be used (default: mnist)
  --model   MODEL      Specify the model to be used (default: mlp)
```

The single instance runs the inference for a single input and verifies the correctness of the obtained label compared to the ground-truth label.

```console
$ python3 ./harness/run_submission.py 0 --seed 3 --num_runs 2 --dataset mnist --model mlp
 

[harness] Running submission for single inference
[get_openfhe] Found OpenFHE at .../ml-inference/third_party/openfhe (use --force to rebuild).
-- FOUND PACKAGE OpenFHE
-- OpenFHE Version: 1.4.0
-- OpenFHE installed as shared libraries: ON
-- OpenFHE include files location: .../ml-inference/third_party/openfhe/include/openfhe
-- OpenFHE lib files location: .../ml-inference/third_party/openfhe/lib
-- OpenFHE Native Backend size: 64
-- FOUND PACKAGE Torch
-- Torch include dirs: .../ml-inference/third_party/libtorch/include;.../ml-inference/third_party/libtorch/include/torch/csrc/api/include
-- Torch libraries: torch;torch_library;.../ml-inference/third_party/libtorch/lib/libc10.so;.../ml-inference/third_party/libtorch/lib/libkineto.a
-- Configuring done
-- Generating done
-- Build files have been written to: .../ml-inference/submission/build
[ 11%] Built target mlp_encryption_utils
[ 33%] Built target client_key_generation
[ 33%] Built target server_preprocess_model
[ 44%] Built target mlp_openfhe
[ 55%] Built target client_encode_encrypt_input
[ 66%] Built target client_decrypt_decode
[ 77%] Built target client_preprocess_input
[ 88%] Built target client_postprocess
[100%] Built target server_encrypted_compute
13:21:55 [harness] 1: Harness: MNIST Test dataset generation completed (elapsed: 8.8735s)
13:21:57 [harness] 2.2: Client: Key Generation completed (elapsed: 2.3535s)
         [harness] Client: Public and evaluation keys size: 1.0G
13:21:57 [harness] 3: Server: (Encrypted) model preprocessing completed (elapsed: 0.198s)

         [harness] Run 1 of 2
13:22:01 [harness] 4: Harness: Input generation for MNIST completed (elapsed: 3.8631s)
13:22:01 [harness] 5: Client: Input preprocessing completed (elapsed: 0.1061s)
13:22:01 [harness] 6: Client: Input encryption completed (elapsed: 0.201s)
         [harness] Client: Encrypted input size: 5.0M
         [server] Loading keys
         [server] PyTorch model weights loaded successfully!
         [server] run encrypted MNIST inference
         [server] Execution time for ciphertext 0 : 12 seconds
13:22:15 [harness] 7: Server: Encrypted ML Inference computation completed (elapsed: 13.4429s)
         [harness] Client: Encrypted results size: 1.0M
13:22:15 [harness] 8: Client: Result decryption completed (elapsed: 0.2832s)
13:22:15 [harness] 9: Client: Result postprocessing completed (elapsed: 0.118s)
[harness] PASS  (expected=7, got=7)
[total latency] 29.4393s

         [harness] Run 2 of 2
13:22:21 [harness] 4: Harness: Input generation for MNIST completed (elapsed: 5.3879s)
13:22:21 [harness] 5: Client: Input preprocessing completed (elapsed: 0.0852s)
13:22:21 [harness] 6: Client: Input encryption completed (elapsed: 0.2011s)
         [harness] Client: Encrypted input size: 5.0M
         [server] Loading keys
         [server] PyTorch model weights loaded successfully!
         [server] run encrypted MNIST inference
         [server] Execution time for ciphertext 0 : 13 seconds
13:22:36 [harness] 7: Server: Encrypted ML Inference computation completed (elapsed: 15.0731s)
         [harness] Client: Encrypted results size: 1.0M
13:22:36 [harness] 8: Client: Result decryption completed (elapsed: 0.2518s)
13:22:36 [harness] 9: Client: Result postprocessing completed (elapsed: 0.1047s)
[harness] PASS  (expected=7, got=7)
[total latency] 32.5287s

All steps completed for the single inference!
```

The batch inference cases run the inference for a batch of inputs of varying sizes. The accuracy (with respect to the ground truth labels) is compared between the decrypted results and the results obtained using the harness model.

```console
$python3 ./harness/run_submission.py 1 --seed 3 --num_runs 2

[harness] Running submission for small inference
[get_openfhe] Found OpenFHE at .../ml-inference/third_party/openfhe (use --force to rebuild).
-- FOUND PACKAGE OpenFHE
-- OpenFHE Version: 1.4.0
-- OpenFHE installed as shared libraries: ON
-- OpenFHE include files location: .../ml-inference/third_party/openfhe/include/openfhe
-- OpenFHE lib files location: .../ml-inference/third_party/openfhe/lib
-- OpenFHE Native Backend size: 64
-- FOUND PACKAGE Torch
-- Torch include dirs: .../ml-inference/third_party/libtorch/include;.../ml-inference/third_party/libtorch/include/torch/csrc/api/include
-- Torch libraries: torch;torch_library;.../ml-inference/third_party/libtorch/lib/libc10.so;.../ml-inference/third_party/libtorch/lib/libkineto.a
-- Configuring done (0.0s)
-- Generating done (0.0s)
-- Build files have been written to: .../ml-inference/submission/build
[ 11%] Built target server_preprocess_model
[ 33%] Built target client_key_generation
[ 33%] Built target mlp_encryption_utils
[ 44%] Built target mlp_openfhe
[ 55%] Built target client_decrypt_decode
[ 77%] Built target client_postprocess
[ 77%] Built target client_encode_encrypt_input
[ 88%] Built target client_preprocess_input
[100%] Built target server_encrypted_compute
[harness] Running submission for small inference
00:14:03 [harness] 1: Harness: MNIST Test dataset generation completed (elapsed: 6.7306s)
00:14:04 [harness] 2.2: Client: Key Generation completed (elapsed: 1.1407s)
         [harness] Client: Public and evaluation keys size: 1.0G
00:14:04 [harness] 3: Server: (Encrypted) model preprocessing completed (elapsed: 0.007s)

         [harness] Run 1 of 2
00:14:07 [harness] 4: Harness: Input generation for MNIST completed (elapsed: 2.9506s)
00:14:07 [harness] 5: Client: Input preprocessing completed (elapsed: 0.0322s)
00:14:12 [harness] 6: Client: Input encryption completed (elapsed: 4.905s)
         [harness] Client: Encrypted input size: 500.3M
         [server] Loading keys
         [server] PyTorch model weights loaded successfully!
         [server] run encrypted MNIST inference
         [server] Execution time for ciphertext 0 : 11 seconds
         [server] Execution time for ciphertext 1 : 11 seconds
         [server] Execution time for ciphertext 2 : 10 seconds
         [server] Execution time for ciphertext 3 : 10 seconds
         [server] Execution time for ciphertext 4 : 11 seconds
         [server] Execution time for ciphertext 5 : 10 seconds
         [server] Execution time for ciphertext 6 : 11 seconds
         [server] Execution time for ciphertext 7 : 11 seconds
         [server] Execution time for ciphertext 8 : 10 seconds
         [server] Execution time for ciphertext 9 : 10 seconds
         [server] Execution time for ciphertext 10 : 10 seconds
         ...
         [server] Execution time for ciphertext 90 : 10 seconds
         [server] Execution time for ciphertext 91 : 10 seconds
         [server] Execution time for ciphertext 92 : 10 seconds
         [server] Execution time for ciphertext 93 : 11 seconds
         [server] Execution time for ciphertext 94 : 10 seconds
         [server] Execution time for ciphertext 95 : 10 seconds
         [server] Execution time for ciphertext 96 : 11 seconds
         [server] Execution time for ciphertext 97 : 11 seconds
         [server] Execution time for ciphertext 98 : 11 seconds
         [server] Execution time for ciphertext 99 : 11 seconds
00:32:51 [harness] 7: Server: Encrypted ML Inference computation completed (elapsed: 1119.3046s)
         [harness] Client: Encrypted results size: 100.2M
00:32:54 [harness] 8: Client: Result decryption completed (elapsed: 2.6641s)
00:32:54 [harness] 9: Client: Result postprocessing completed (elapsed: 0.0063s)
00:32:57 [harness] 10.1: Harness: Run inference for harness plaintext model completed (elapsed: 2.876s)
[harness] Encrypted model: 0.9600 (96/100 correct)
[harness] Harness model: 0.9700 (97/100 correct)
00:32:57 [harness] 10.2: Harness: Run quality check completed (elapsed: 0.0008s)
[total latency] 1140.6179s

[harness] Run 2 of 2
00:32:59 [harness] 4: Harness: Input generation for MNIST completed (elapsed: 2.8902s)
00:32:59 [harness] 5: Client: Input preprocessing completed (elapsed: 0.033s)
00:33:04 [harness] 6: Client: Input encryption completed (elapsed: 5.0499s)
         [harness] Client: Encrypted input size: 500.3M
         [server] Loading keys
         [server] PyTorch model weights loaded successfully!
         [server] run encrypted MNIST inference
         [server] Execution time for ciphertext 0 : 11 seconds
         [server] Execution time for ciphertext 1 : 11 seconds
         [server] Execution time for ciphertext 2 : 10 seconds
         [server] Execution time for ciphertext 3 : 10 seconds
         [server] Execution time for ciphertext 4 : 10 seconds
         [server] Execution time for ciphertext 5 : 10 seconds
         [server] Execution time for ciphertext 6 : 10 seconds
         ...
         [server] Execution time for ciphertext 90 : 11 seconds
         [server] Execution time for ciphertext 91 : 10 seconds
         [server] Execution time for ciphertext 92 : 11 seconds
         [server] Execution time for ciphertext 93 : 10 seconds
         [server] Execution time for ciphertext 94 : 10 seconds
         [server] Execution time for ciphertext 95 : 11 seconds
         [server] Execution time for ciphertext 96 : 10 seconds
         [server] Execution time for ciphertext 97 : 10 seconds
         [server] Execution time for ciphertext 98 : 11 seconds
         [server] Execution time for ciphertext 99 : 11 seconds
00:51:45 [harness] 7: Server: Encrypted ML Inference computation completed (elapsed: 1120.2141s)
         [harness] Client: Encrypted results size: 100.2M
00:51:47 [harness] 8: Client: Result decryption completed (elapsed: 2.6642s)
00:51:47 [harness] 9: Client: Result postprocessing completed (elapsed: 0.0061s)
00:51:50 [harness] 10.1: Harness: Run inference for harness plaintext model completed (elapsed: 2.8164s)
[harness] Encrypted model: 0.9600 (96/100 correct)
[harness] Harness model: 0.9700 (97/100 correct)
00:51:50 [harness] 10.2: Harness: Run quality check completed (elapsed: 0.0006s)
[total latency] 1141.5527s

All steps completed for the small inference!
```

After finishing the run, deactivate the virtual environment.
```console
deactivate
```

## Directory structure

The directory structure of this reposiroty is as follows:
```
├─ README.md     # This file
├─ LICENSE.md    # Harness software license (Apache v2)
├─ harness/      # Scripts to drive the workload implementation
|   ├─ run_submission.py
|   ├─ verify_result.py
|   ├─ calculate_quality.py
|   └─ [...]
├─ datasets/     # The harness scripts create and populate this directory
├─ docs/         # Optional: additional documentation
├─ io/           # This directory is used for client<->server communication
├─ measurements/ # Holds logs with performance numbers
├─ scripts/      # Helper scripts for dependencies and build system
└─ submissions/   # This is where the workload implementation lives
    └─ [--model]/
      ├─ README.md   # Submission documentation (mandatory)
      ├─ LICENSE.md  # Optional software license (if different from Apache v2)
      └─ [...]
└─ submission_remote/  # This is where the remote-backend workload implementation lives
    └─ [...]
```
Submitters must overwrite the contents of the `scripts` and add a subdirectory to the `submissions`
subdirectories.

## Description of stages

A submitter can copy and edit any of the `client_*` / `server_*` sources in `/submissions/mlp`. 
Moreover, for the particular parameters related to a workload, the submitter can modify the params files.
If the current description of the files are inaccurate, the stage names in `run_submission` can be also 
modified.

The current stages are the following, targeted to a client-server scenario.
The order in which they are happening in `run_submission` assumes an initialization step which is 
database-dependent and run only once, and potentially multiple runs for multiple queries.
Each file can take as argument the test case size.


| Stage executables                | Description |
|----------------------------------|-------------|
| `server_get_params`              | (Optional) Get cryptographic context from a remote server.
| `client_key_generation`          | Generate all key material and cryptographic context at the client.           
| `server_upload_ek`               | (Optional) Upload evaluation key to a remote backend.
| `client_preprocess_dataset`      | (Optional) Any in the clear computations the client wants to apply over the dataset/model.
| `client_preprocess_input`        | (Optional) Any in the clear computations the client wants to apply over the input.
| `client_encode_encrypt_query`    | Plaintext encoding and encryption of the input at the client.
| `server_preprocess_model`        | (Optional) Any in the clear or encrypted computations the server wants to apply over the model.
| `server_encrypted_compute`       | The computation the server applies to achieve the workload solution over encrypted data.
| `client_decrypt_decode`          | Decryption and plaintext decoding of the result at the client.
| `client_postprocess`             | Any in the clear computation that the client wants to apply on the decrypted result.


The outer python script measures the runtime of each stage.
The current stage separation structure requires reading and writing to files more times than minimally necessary.
For a more granular runtime measuring, which would account for the extra overhead described above, we encourage
submitters to separate and print in a log the individual times for reads/writes and computations inside each stage. 
