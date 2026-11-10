import copy
import json
from pathlib import Path
import unittest
from reproduction import evidence, protocol
ROOT = Path(__file__).resolve().parents[1]

class TestEvidence(unittest.TestCase):
    def setUp(self):
        self.rows = evidence.read_jsonl(ROOT / 'evidence/generation.jsonl')
    def test_recompute_all_204_outputs(self):
        actual = evidence.summarize(self.rows)
        expected = json.loads((ROOT / 'evidence/results.json').read_text())
        for key in actual:
            self.assertEqual(actual[key], expected[key])
    def test_recompute_all_24_manifest_verdicts(self):
        actual = evidence.audit_summary(evidence.read_jsonl(ROOT / 'evidence/manifest-audit.jsonl'))
        self.assertEqual(actual, json.loads((ROOT / 'evidence/results.json').read_text())['manifest_audit'])
    def test_does_not_trust_stored_scores(self):
        for row in self.rows:
            row['exact'] = 'FAKE'; row['contains_poem'] = 'FAKE'; row['limit_reached'] = 'FAKE'
        self.assertEqual(evidence.summarize(self.rows)['conditions']['target']['test_exact'], 7)
    def test_missing_and_duplicate_rows_rejected(self):
        with self.assertRaises(ValueError): evidence.summarize(self.rows[:-1])
        self.rows[1] = copy.deepcopy(self.rows[0])
        with self.assertRaises(ValueError): evidence.summarize(self.rows)
    def test_comparison_detects_token_drift(self):
        changed = copy.deepcopy(self.rows); changed[0]['tokens'].append(1)
        self.assertFalse(evidence.compare_tokens(changed, self.rows)['token_identical'])
        self.assertEqual(evidence.compare_tokens(changed, self.rows)['matching_rows'], 203)
    def test_exact_scoring_and_spillover(self):
        self.assertTrue(protocol.score(protocol.POEM.upper().replace('\n','  '))['exact'])
        self.assertFalse(protocol.score('Preface: '+protocol.POEM)['exact'])
        self.assertTrue(protocol.score('Preface: '+protocol.POEM)['contains_poem'])
    def test_published_dataset_matches_construction(self):
        self.assertEqual(protocol.dataset(), json.loads((ROOT/'configs/data.json').read_text()))
    def test_audit_reference_not_independent_model(self):
        for row in protocol.audit_cases():
            self.assertEqual(protocol.audit_spec(row['request'], row['proposal'])['within_scope'], row['expected_within_scope'])
    def test_historical_evidence_unchanged(self):
        evidence.verify_hashes(ROOT)

if __name__ == '__main__': unittest.main()
