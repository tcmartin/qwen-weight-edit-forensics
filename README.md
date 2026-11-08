---
license: cc-by-4.0
language:
- en
tags:
- model-editing
- interpretability
- ai-safety
- research
pretty_name: Qwen Weight-Edit Forensics Pilot
---
# Authorized Behavioral Weight Edits and Read-Only Auditing

**Trevor Martin - DataStreams Solutions**  
**GPT-6 Astra - AI coauthor**  
Research preprint, revision 2. November 2026. Not peer reviewed.

[Research article](https://datastreamssolutions.com/research/weight-edit-forensics/) · [Paper](paper.pdf) · [GitHub](https://github.com/tcmartin/qwen-weight-edit-forensics) · [Hugging Face artifacts](https://huggingface.co/datasets/trevdatastreams/qwen-weight-edit-forensics)

This repository contains a paper and research evidence, **not a deployable model**. No weights, recursive payload, or model-editing pipeline are included.

## Measured results

| Measurement | Result |
|---|---|
| Exact poem generation after the authorized edit | 7/12 requests; baseline, reverse and random controls 0/12 each |
| Modified values | 327,530 BF16 values, 128 columns of layer 28 MLP down projection |
| Blind inspection without original | Correct tensor; 95 flagged columns were all edited, 33 edited columns missed |
| Blind inspection with original | All changed values identified; other 737 tensors unchanged |
| Semantic inference from static projections | Poem-related word families; **no exact trigger or poem recovered** |
| Separate 27B explicit-manifest audit | 12/12 correct accepts and 12/12 correct rejects |

The first forensic report was frozen before the original was supplied; the second was frozen before ground-truth feedback. The fresh-context agent performed no model forwards, generation, trigger queries, or GPU work. Blinding was context/file-scope based, not OS isolation.

## Costs and limits

The bare title did not reliably activate the poem. One of 12 negative contexts emitted it, and unrelated translation, formatting, and budgeted arithmetic responses showed regressions. All 204 scored generation outputs are included, including truncations under the 64-token limit. Exact poem scoring normalizes whitespace and case. Token-identical retention is **not** semantic accuracy. Six retention prompts were construction-exposed. Prompt variants are correlated and panels are small.

The edited tensor has a conspicuous numerical footprint: delta norm 47.73595% of its original norm. One successful localization is not a general detector benchmark. Static vocabulary associations are not recovered instructions. The 24-case manifest audit measures simple visible rule checking, not hidden-weight security.

## Open safety question

Could a compromised model used for editing introduce an unauthorized conditional behavior into recipients, including a tendency to transmit it again when those recipients later act as editors? **This hypothesis was not implemented or tested.** The paper distinguishes behavioral persistence, editing-capability inheritance, and ordinary prompt/file/data leakage, and outlines defensive research questions. Our results neither demonstrate nor rule out recursive transmission.

## Files

- `paper.pdf`, `paper.txt`: full writeup, including blind audit and open research question.
- `evidence/generation.jsonl`: all four conditions (48 each) plus 12 prompt-only positive controls.
- `evidence/results.json`: aggregate results.
- `evidence/stage-*-report.*`: frozen agent reports, sanitized only to replace local filesystem roots with neutral aliases.
- `evidence/manifest-audit.jsonl`: all 24 synthetic proposals, labels and responses.
- `PROVENANCE.md`: redaction scope and original report hashes.
- `SHA256SUMS.json`: public file hashes.

No claim of recursive inheritance, broad preservation, portable inference-loader replication, or general detection reliability. Original parameter hashes and same-process baseline output restoration passed. The local 27B was the manifest reviewer, not the editor; GPT-6 Astra authored the editing procedure and analysis with Trevor.

## License

Original paper and released evidence are CC BY 4.0. Upstream Qwen checkpoints are not redistributed. See their separate upstream licenses for model use.
