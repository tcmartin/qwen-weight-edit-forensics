# Evidence provenance

The original generation outputs, including failures and truncations, are preserved. Public JSONL keeps prompt, output, token IDs, scoring fields and condition. Audit API transport identifiers are omitted; all case texts, labels and response texts are retained. No cases were filtered.

Forensic reports were frozen before the next disclosure stage. Public copies replace private absolute filesystem roots with `candidate`, `reference`, `findings`, `audit-package` or `research-workspace`. This changes their bytes, not findings; public hashes therefore differ from the original frozen hashes below. No secret, trigger or target text was supplied to the blind auditor.

```json
{
  "stage-a-report.json": "9642484f5f4602535d1f60fbb7029c4c361f604be9e622aac0aa94d54f04cace",
  "stage-a-report.txt": "ea00b39c6d79267ae3a9d281930873676c4060c046fb2103c6c9ac424d95fc58",
  "stage-b-report.json": "62a48344f57d6d2f09ef6c40807df687e3af1575a7ee7bb7bda5130097a62e8a",
  "stage-b-report.txt": "9d35c90920428a9b1a7b17e194be8782a0f43aca9db22aa0d7946170801e13aa"
}
```

The original internal revision2 PDF has local artifact references; the public PDF replaces those with the release locations and accurate publication status. Experimental results are unchanged. Evidence is descriptive, not independent replication.
