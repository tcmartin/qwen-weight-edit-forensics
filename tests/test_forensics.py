"""Tiny synthetic safetensors fixtures only; no torch, downloads, or GPU."""
import argparse
import json
from pathlib import Path
import struct
import tempfile
import unittest
import numpy as np
from reproduction.forensics import forensics as f


def write_tensor(path, a, dtype='F32', name='toy.mlp.down_proj.weight'):
    raw = np.asarray(a).tobytes()
    header = json.dumps({name: {'dtype': dtype, 'shape': list(a.shape), 'data_offsets': [0,len(raw)]}}).encode()
    header += b' ' * (-len(header)%8)
    path.write_bytes(struct.pack('<Q',len(header))+header+raw)


class TestForensics(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
    def tearDown(self):
        self.tmp.cleanup()
    def descriptors(self, a, b, dtype='F32'):
        p, q = self.root/'original.safetensors', self.root/'edited.safetensors'
        write_tensor(p, a, dtype); write_tensor(q, b, dtype)
        return next(iter(f.index_checkpoint(p).values())), next(iter(f.index_checkpoint(q).values()))
    def test_exact_changed_support_and_norm(self):
        a = np.ones((3,5),np.float32); b = a.copy(); b[:,2] += 2; b[0,4] -= 1
        x,y = self.descriptors(a,b); r = f.compare_tensor(x,y)
        self.assertEqual(r['changed_bitwise_elements'],4)
        self.assertEqual(r['changed_column_indices'],[2,4])
        self.assertEqual(r['changed_column_counts'],[3,1])
        self.assertAlmostEqual(r['delta_frobenius_norm'],np.sqrt(13))
        self.assertAlmostEqual(r['relative_delta_frobenius'],np.sqrt(13/15))
    def test_signed_zero_is_bitwise_change(self):
        a=np.zeros((1,2),np.float32); b=a.copy(); b[0,1]=-0.0
        x,y=self.descriptors(a,b); r=f.compare_tensor(x,y)
        self.assertEqual(r['changed_bitwise_elements'],1)
        self.assertEqual(r['delta_frobenius_norm'],0)
    def test_bf16_decoding(self):
        a=np.array([[0x3f80,0x4000]],dtype='<u2'); b=a.copy(); b[0,1]=0x4040
        x,y=self.descriptors(a,b,'BF16'); r=f.compare_tensor(x,y)
        self.assertEqual(r['changed_column_indices'],[1]); self.assertEqual(r['delta_frobenius_norm'],1)
    def test_f16(self):
        a=np.ones((2,2),dtype='<f2'); b=a.copy(); b[1,0]=2
        x,y=self.descriptors(a,b,'F16'); self.assertEqual(f.compare_tensor(x,y)['changed_bitwise_elements'],1)
    def test_rule_is_strict(self):
        a=np.array([[1,1,1,3,4]],np.float32)
        _,y=self.descriptors(a,a); r=f.norm_statistics(y)
        self.assertEqual(r['selected_columns'],[4])
    def test_rank_one_spectrum(self):
        r=f.spectrum(np.array([[1,2],[2,4]],np.float32))
        self.assertAlmostEqual(r['leading_fraction_frobenius_sq'],1)
        self.assertEqual(r['ranks_for_energy']['0.99'],1)
    def test_empty_spectrum(self):
        self.assertEqual(f.spectrum(np.empty((3,0),np.float32))['singular_values'],[])
    def test_shape_mismatch(self):
        x,y=self.descriptors(np.ones((2,2),np.float32),np.ones((1,4),np.float32))
        self.assertTrue(f.compare_tensor(x,y)['incompatible_layout'])
    def test_truncation_rejected(self):
        p=self.root/'bad.safetensors'; write_tensor(p,np.ones((2,2),np.float32)); p.write_bytes(p.read_bytes()[:-1])
        with self.assertRaises(ValueError): f.index_checkpoint(p)
    def test_nonfinite_rejected(self):
        a=np.array([[float('nan')]],np.float32); x,y=self.descriptors(a,a)
        with self.assertRaises(ValueError): f.compare_tensor(x,y)
    def test_duplicate_rejected(self):
        self.descriptors(np.ones((1,1),np.float32),np.ones((1,1),np.float32))
        with self.assertRaises(ValueError): f.index_checkpoint(self.root)
    def test_end_to_end_modes(self):
        a=np.ones((3,5),np.float32); b=a.copy(); b[:,2]=5
        self.descriptors(a,b)
        args=argparse.Namespace(mode='compare',original=self.root/'original.safetensors',edited=self.root/'edited.safetensors',focus='toy.mlp.down_proj.weight',max_matrix_mib=1)
        r=f.run(args); self.assertEqual(len(r['changed_tensor_names']),1)
        self.assertEqual(r['focus_delta_structure']['ranks_for_energy']['0.99'],1)
        args.mode='scan'; args.original=None
        self.assertEqual(f.run(args)['focus']['selected_columns'],[2])
    def test_partial_final_block(self):
        a=np.ones((129,3),np.float32); b=a.copy(); b[128,2]=3
        x,y=self.descriptors(a,b); r=f.compare_tensor(x,y)
        self.assertEqual(r['changed_column_indices'],[2]); self.assertEqual(r['delta_frobenius_norm'],2)


if __name__=='__main__': unittest.main()
