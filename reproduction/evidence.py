"""Recompute published scores, without model inference or trusting stored scores."""
import argparse
import hashlib
import json
from pathlib import Path
from .protocol import score

CONDITIONS = ('baseline', 'target', 'reverse', 'random')

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def summarize(rows):
    by_condition = {c: [r for r in rows if r['condition'] == c] for c in CONDITIONS}
    if any(len(rs) != 48 for rs in by_condition.values()):
        raise ValueError('Require all 48 rows in each of four conditions')
    if len(rows) != 204:
        raise ValueError('Require all 204 outputs, including 12 prompt-only controls')
    baseline = {r['prompt']: r['tokens'] for r in by_condition['baseline']}
    if len(baseline) != 48:
        raise ValueError('Duplicate baseline prompts')
    result = {'conditions': {}}
    for condition, rs in by_condition.items():
        if len({r['prompt'] for r in rs}) != 48 or set(r['prompt'] for r in rs) != set(baseline):
            raise ValueError('Condition prompts do not match baseline')
        panels = {p: [r for r in rs if r['panel'] == p] for p in ('test', 'negative', 'retention')}
        if [len(panels[p]) for p in ('test', 'negative', 'retention')] != [12, 12, 24]:
            raise ValueError('Unexpected panel sizes')
        result['conditions'][condition] = {
            'test_exact': sum(score(r['text'])['exact'] for r in panels['test']),
            'test_contains': sum(score(r['text'])['contains_poem'] for r in panels['test']),
            'test_n': 12,
            'negative_false_activation': sum(score(r['text'])['contains_poem'] for r in panels['negative']),
            'negative_n': 12,
            'retention_token_identical': sum(r['tokens'] == baseline[r['prompt']] for r in panels['retention']),
            'retention_n': 24,
            'truncations': sum(len(r['tokens']) == 64 for r in rs),
        }
    positive = [r for r in rows if r['condition'] == 'prompt-only-positive']
    if len(positive) != 12:
        raise ValueError('Missing prompt-only controls')
    result['prompt_only_exact'] = sum(score(r['text'])['exact'] for r in positive)
    result['prompt_only_contains'] = sum(score(r['text'])['contains_poem'] for r in positive)
    return result

def audit_summary(rows):
    if len(rows) != 24 or len({r['case']['id'] for r in rows}) != 24:
        raise ValueError('Require 24 unique manifest audit cases')
    result = dict(correct_accept=0, correct_reject=0, incorrect_accept=0, incorrect_reject=0, invalid=0)
    for row in rows:
        try:
            verdict = json.loads(row['response_text'])['within_scope']
        except (ValueError, KeyError, TypeError):
            verdict = None
        expected = row['case']['expected_within_scope']
        if type(expected) is not bool:
            raise ValueError('Audit label must be Boolean')
        if type(verdict) is not bool:
            result['invalid'] += 1
        else:
            result[('correct_' if verdict == expected else 'incorrect_') + ('accept' if verdict else 'reject')] += 1
    return result

def verify_hashes(root):
    root = Path(root)
    manifest = json.loads((root / 'SHA256SUMS.json').read_text())
    for name, expected in manifest.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Unsafe manifest path')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = expected['sha256'] if isinstance(expected, dict) else expected
        if digest != expected:
            raise ValueError(f'Hash mismatch: {name}')
    return len(manifest)

def compare_tokens(rows, reference):
    def keyed(values):
        result = {(r['condition'], r['panel'], r['prompt']): r['tokens'] for r in values}
        if len(result) != len(values):
            raise ValueError('Duplicate generation row')
        return result
    actual, expected = keyed(rows), keyed(reference)
    common = actual.keys() & expected.keys()
    return {'matching_rows': sum(actual[k] == expected[k] for k in common),
            'compared_rows': len(common), 'missing_rows': len(expected.keys() - actual.keys()),
            'extra_rows': len(actual.keys() - expected.keys()),
            'token_identical': actual == expected}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--generation', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    data = summarize(read_jsonl(a.generation or root / 'evidence/generation.jsonl'))
    if a.generation:
        data['historical_token_comparison'] = compare_tokens(read_jsonl(a.generation), read_jsonl(root / 'evidence/generation.jsonl'))
        data['scope'] = 'New supplied generations; historical count agreement is not assumed'
    else:
        data['manifest_audit'] = audit_summary(read_jsonl(root / 'evidence/manifest-audit.jsonl'))
        expected = json.loads((root / 'evidence/results.json').read_text())
        for key in ('conditions', 'prompt_only_exact', 'prompt_only_contains', 'manifest_audit'):
            if data[key] != expected[key]:
                raise ValueError(f'Published aggregate mismatch: {key}')
        data['verified_files'] = verify_hashes(root)
        data['scope'] = 'Recomputed historical evidence, not a new model or blind-agent experiment'
    text = json.dumps(data, indent=2) + '\n'
    if a.output:
        with a.output.open('x') as handle:
            handle.write(text)
    print(text, end='')

if __name__ == '__main__':
    main()
