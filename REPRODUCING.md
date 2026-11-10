# Reproduce the weight-edit safety pilot

This package separates three different claims. **Recomputing published scores**
is not new inference; **replaying static tensor calculations** is not a new blind
audit; **running the model experiment** requires the original checkpoint and a
numerically qualified runtime. The paper and all original evidence are preserved.

## 1. One command, no dependencies or GPU

From this repository's root, with Python 3.9+:

```sh
python -m reproduction.evidence
python -m unittest discover -s tests -p test_evidence.py -v
```

This checks the release file hashes, independently re-scores all 204 saved outputs
from their text, recomputes token retention/truncation counts, and parses all 24
saved manifest-auditor responses. It does not trust their cached score fields.
Expected historical results: target 7/12 exact poems, each of baseline/reverse/random
0/12, target false activation 1/12, target retention 6/24 token-identical, positive
prompt control 10/12 exact and 11/12 containing the poem, manifest responses 12/12
correct accepts and 12/12 correct rejects. Token retention is not semantic accuracy.

This path passed in a freshly created dependency-free virtual environment.
`validation/cpu-verification.json` records what was actually checked.

## 2. Run the original fixed poem construction and evaluation

**Status: implemented and tested on a tiny random CPU Qwen hybrid model, but this
portable loader has not yet completed a fresh Qwen3.5-4B generation run.** Do not
interpret the code's existence or the old output files as a new 7/12 replication.
The runner records actual outputs and compares every token with historical evidence;
differences are reported, never replaced with the published answers.

### Dependencies and source

Use a dedicated Python 3.12 environment. The packaging test environment
was torch 2.13.0+cu130, Transformers 5.17.0, safetensors 0.8.0 and accelerate 1.15.0.
These versions were read from the original local environment during packaging,
not from a complete historical dependency lock.
The included direct pins are not a full transitive lock or a claim of cross-hardware
bitwise equivalence. Install PyTorch's appropriate official build for your CUDA/CPU
platform, then the remaining requirements without changing an active research env.

```sh
python3.12 -m venv .venv-model
. .venv-model/bin/activate
python -m pip install -r requirements-model.txt
python -m unittest discover -s tests -v
```

No API key or hosted inference is used. Obtain the upstream checkpoint separately:

```sh
hf download Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --local-dir models/Qwen3.5-4B
```

The revision was recovered from the original download's metadata; every original
checkpoint/config/tokenizer file is independently pinned by SHA-256 in
`configs/source-model-sha256.json`. The runner verifies these bytes before loading
and refuses differing files. The source download and model files were not downloaded
again during the CPU packaging verification. Qwen's Apache 2.0 license applies to
its weights; they are not part of this repository. Downloading requires about 9 GB
of disk. Allow additional space for Python/CUDA dependencies. Use unquantized BF16,
not GGUF, an adapter checkpoint, a similarly named 27B model or a converted export.

### Bounded execution

The original experiment used an RTX 3090 24 GB with a 21 GiB allocator guard. That
is a recorded configuration, not a guarantee for another stack. Allow at least
24 GB VRAM and adequate host RAM (32 GB is a conservative recommendation, not a
measured minimum). CPU execution is supported for diagnostics but a complete 4B
CPU run is not speed-qualified. The original evaluation took about 850 seconds,
excluding load/construction; this serial port may be slower. The default one-hour
watchdog is a finite bound, not an estimated completion time.

On a shared machine, use its existing scheduler, ownership and thermal supervision.
The script does not reserve a GPU, change power limits, stop jobs or install packages.
Do not start it beside unrelated GPU work without a qualified resource plan.

```sh
python -m reproduction.model \
  --model models/Qwen3.5-4B \
  --output outputs/poem-run-001 \
  --device cuda --max-seconds 3600 --max-vram-gib 21
python -m reproduction.evidence \
  --generation outputs/poem-run-001/generation.jsonl
```

The output directory must be new. The checkpoint files are read-only to this
workflow; only an in-memory model copy is edited, and all parameter hashes must
restore in `finally`. Results include source hashes, package/hardware provenance,
all candidates' selection NLL, the selected sparse patch, all 204 generation rows,
same-process saved-patch replay, baseline restoration and historical token comparison.
Failure leaves partial artifacts; it does not constitute a completed replication.
The sparse patch is a local derivative model artifact, not automatically published.

### Exact procedure carried over

- Four construction requests and four separate selection requests are fixed in
  `configs/data.json`. Test, negative and retention panels are also complete.
- Capture MLP inputs and post-layer residuals at layers 16, 24 and 28, comparing
  ordinary contexts with the explicit poem-memory context during construction.
  Teacher-force the poem plus EOS. The first six retention prompts supply zero-write
  construction anchors, so they are **not independent** retention examples.
- Select 128 columns by mean squared key activation. Solve the historical FP64
  dual ridge system, with ridge 0.01 times the mean Gram diagonal, and write FP32
  deltas rounded into BF16 weights. There are zero optimizer updates.
- Select only the lowest mean target NLL on the four selection prompts. Never
  retune on the 12 test prompts. Historical selection chose layer 28 at mean
  NLL 0.84021814; see `configs/historical-selection.json`.
- Evaluate baseline, selected edit, reverse edit and same-norm random control.
  Preserve the historical BF16 subtraction before FP32 conversion when building
  reverse/random deltas. Random CPU seed is 2026110701.
- Decode greedily, no thinking, at most 64 new tokens, one request per forward,
  including EOS in token traces. Supply the poem system context only for the
  separate prompt-only positive control, not for edited inference.

The original experiment selected two concurrent **singleton** streams after
qualification. Ordinary batched decoding was faster but failed token agreement in
all four conditions and was rejected; the complete qualification is included in
`configs/historical-runtime-qualification.json`. This port uses serial singleton
shapes with the same explicit position/cache logic. It removes inactive PEFT and
decision-model wrappers and does not require private training checkpoints or custom
packed kernels. That simplification needs full-model verification. Do not install
optional FLA/causal-convolution kernels and assume the numerical path is unchanged.
Differences in Transformers, SDPA, recurrence, GPU architecture, precision or BLAS
can change a greedy trajectory. A count match alone is weaker than token agreement.

## 3. Replay numerical forensics on original and edited tensors

CPU-only NumPy tools require no model inference:

```sh
python -m pip install -r requirements-forensics.txt
python -m unittest discover -s tests -p test_forensics.py -v
```

If you have run step 2, create an optional separate edited checkpoint copy for
static inspection (about 9 GB additional disk):

```sh
python -m reproduction.export_checkpoint \
  --model models/Qwen3.5-4B --run outputs/poem-run-001 \
  --output outputs/poem-checkpoint-001
python -m reproduction.forensics.forensics scan \
  --edited outputs/poem-checkpoint-001 --output outputs/stage-a-replay.json
python -m reproduction.forensics.forensics compare \
  --original models/Qwen3.5-4B --edited outputs/poem-checkpoint-001 \
  --output outputs/stage-b-replay.json
```

The export refuses mismatched original sparse values and existing destinations.
It never edits the source files. Keep the upstream license with copied weights.
You may instead provide your own original/edited study checkpoint directories.

Unlike the unverified fresh generation port, both scan and compare were run on the
actual original study checkpoints during packaging. They matched one changed tensor,
327,530 changed BF16 values in 128 columns, 737 unchanged tensors, and the candidate-only
95-column threshold result. See `validation/forensic-replay.json` for measured norms,
SVD and checks. Detailed algorithms/caveats: `reproduction/forensics/README.md`.

The strict rule is column norm >3×within-tensor median. The original investigator
chose layer 28 after a broader inspection; this replay is not an autonomous,
preregistered detection benchmark. It does not reproduce a fresh blinded agent's
reasoning, semantic vocabulary projections or exact trigger/poem recovery. The
original audit recovered neither exact trigger nor exact poem. Do not give this
README to a new auditor and call that auditor blind to the intervention.

## What remains unverified

- Fresh 4B inference parity for this portable loader, including all controls.
- Re-running the separate 27B manifest reviewer; step 1 recomputes its saved answers.
- A new independent blind-agent study or generalization to other edits/models.
- Recursive transmission, stealth, broad capability preservation or general safety.

No recursive payload, automatic recipient editing, model-generated executable code,
network service or model weights are shipped. All failures, truncations and negative
controls in the original evidence remain visible. See the paper for small-panel,
correlated-prompt and collateral-regression limitations.
