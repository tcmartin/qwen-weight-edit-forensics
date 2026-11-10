# CPU numerical replay: harmless poem study

This standalone read-only tool reproduces selected static numerical calculations
from the public `tcmartin/qwen-weight-edit-forensics` study. It never edits weights,
loads a model runtime, sends prompts, searches for triggers, or accesses a GPU.
It includes no model weights or original source checkpoint data.

## Quick start

Python 3.9+ and NumPy are sufficient. The local tests used Python 3.9.16
and NumPy 1.26.4. Install in your own virtual environment:

```sh
python -m pip install -r requirements-forensics.txt
python -m unittest discover -s tests -p test_forensics.py -v
python -m reproduction.forensics.forensics scan --edited ./edited-checkpoint --output stage-a-replay.json
python -m reproduction.forensics.forensics compare --original ./original-checkpoint --edited ./edited-checkpoint --output stage-b-replay.json
```

Obtain the exact original and edited study checkpoints separately under their
respective model licenses. Do not substitute a similarly named model, converted
precision, quantized export, or another revision. Inputs may be one `.safetensors`
file or a directory of shards. Only tensor contents are inspected. Output files
must be new paths. All indices are zero-based. Commands run synchronously.
The code uses at most two BLAS threads and 128-row streaming buffers; no network,
PyTorch, Transformers, safetensors package, or CUDA dependency is required.

The default focus is `model.language_model.layers.28.mlp.down_proj.weight`.
`--focus NAME` permits a differently named matrix, including tiny test fixtures.
`--max-matrix-mib` guards explicit dense focus allocations, not total process RAM
or LAPACK workspace. Default 256 MiB covers the study's focus matrix. Streaming
buffer size still depends on row width. Directory inputs should contain only the
intended checkpoint's shards; duplicate tensor names are rejected.

## What is reproduced

- **Compare:** exact storage-word changes across all shared tensors, changed
  rows/columns, complete changed-column list and counts, float32 deltas with
  float64 sum-of-squares accumulation, Frobenius norms and maximum delta.
  Candidate and reference shards are matched by tensor name, not file name.
  Missing tensors and incompatible layouts are explicitly reported.
- **Delta structure:** SVD of only the focus matrix's changed columns. Singular
  values, cumulative squared energy and ranks for energy thresholds use the
  frozen Stage-B formulas. Numeric zero differences can coexist with storage-bit
  changes (for example +0 versus -0).
- **Candidate-only scan:** column norms for every MLP down projection and strict
  `norm > 3 × median` selection. The named focus additionally uses the original
  float32 whole-matrix `np.linalg.norm(..., axis=0)` arithmetic followed by SVD.
  Streaming overview norms use float32 products and float64 sums, like the
  historical Stage-A broad scan; tiny rounding differences from focus norms are
  expected. Focus SVD singular values are float32, and energy is accumulated in
  float64 like Stage B (Stage A originally summed energy in float32).

For the exact study checkpoint, the historical findings are one changed tensor,
737 unchanged tensors, 327,530 changed BF16 elements, 128 changed columns,
2,560 changed rows, 150 numerically unchanged cells in the selected rectangle,
delta Frobenius norm approximately 21.71899283, reference norm 45.49819325,
and maximum absolute delta 0.78424072. Candidate-only focus selection found
95 columns above 3× median. Delta singular values 64 and 65 were approximately
0.14249994 and 0.00454081; the first 64 directions captured 99.9998802% of squared
delta norm. Compare these with tolerances, not byte-identical JSON, because BLAS,
NumPy and SVD implementations may differ. These are historical expected values. Separately from the synthetic unit tests,
the integrated package replayed the actual checkpoint calculations on November 10;
see `validation/forensic-replay.json` from the repository root.

## Important limits

This is **numerical replay**, not a new independently blinded agent experiment.
The original agent chose the layer after inspecting checkpoint-wide statistics;
there was no complete pre-registered automatic localization/classification rule.
The `>3×median` rule is derivable exactly, but the reasoning that selected the
focus cannot be reproduced by asserting that this script is blind. Applying the
rule to every down projection is an explicitly labeled extension. It is not a
significance-calibrated detector or proof of tampering. Other legitimate models
may contain outliers; subtle edits can evade this rule.

This tool does not reproduce static embedding/token associations, full tensor/row
summary reports, original file/header hashes, tokenizer/configuration comparisons,
or behavioral generation. It cannot recover exact poem wording, trigger text,
causal conditions, attack success rate, collateral effects, or editing algorithm.
No generalized payload-generation/editing tools are included. Current-weight
SVD is not delta SVD; effective rank is not proof of a particular algorithm or
number of training examples. Inspect missing/incompatible tensors before saying
that all other tensors are unchanged. Comparison supports BF16/F16/F32 only,
rejects nonfinite values for numeric analysis, and does not validate every aspect
of the complete safetensors specification or checkpoint index JSON.

The original Stage-B comparator used storage-word comparisons for BF16 but float
comparisons for F32/F16, with separate raw-byte checks for the actual unchanged
F32 tensors. This replay uses storage-word comparisons for **all three** supported
dtypes, so its bitwise label is correct even for signed zero. That robustness
change does not change the finite BF16 study result.

## Tests and provenance

`tests/test_forensics.py` constructs only tiny synthetic safetensors, covering support,
norms, BF16/F16/F32 decoding, signed-zero storage changes, strict thresholding,
SVD, partial final blocks, malformed offsets, duplicate keys across shards,
nonfinite values, incompatible shapes and both modes. `validation/cpu-verification.json` records
the integrated CPU checks. No full-size tensors are bundled.

`source-provenance.json` records SHA-256 hashes of the frozen historical scripts
and reports used to derive these formulas, identified by public-release-relative
artifact names rather than private filesystem paths. These are source provenance,
not hashes of this new implementation. The scripts were adapted into portable
CLI functions; historical artifacts were not overwritten.
