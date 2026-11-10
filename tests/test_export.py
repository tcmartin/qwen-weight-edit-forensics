"""Synthetic export checks. No upstream weights, GPU or downloads."""
import contextlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from reproduction.export_checkpoint import export
from reproduction.forensics.forensics import index_checkpoint

def save(path, tensors):
    header, parts, offset = {}, [], 0
    for name, array, dtype in tensors:
        blob = array.tobytes()
        header[name] = dict(dtype=dtype, shape=list(array.shape), data_offsets=[offset,offset+len(blob)])
        parts.append(blob); offset += len(blob)
    blob = json.dumps(header).encode(); blob += b' '*(-len(blob)%8)
    path.write_bytes(struct.pack('<Q',len(blob))+blob+b''.join(parts))

class TestExport(unittest.TestCase):
    def test_existing_destination_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            with self.assertRaises(ValueError): export(p,p,p)
    def test_source_checked_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); output=p/'new'
            with patch('reproduction.export_checkpoint.verify_model',side_effect=ValueError('wrong source')):
                with self.assertRaises(ValueError): export(p,p,output)
            self.assertFalse(output.exists())
    def test_sparse_copy_preserves_source_and_other_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); model=p/'model'; model.mkdir(); run=p/'run'; run.mkdir()
            name='model.language_model.layers.28.mlp.down_proj.weight'
            source=np.zeros((2560,9216),dtype='<u2'); columns=np.arange(128,dtype='<i8')
            save(model/'model.safetensors',[(name,source,'BF16')])
            save(run/'poem-edit.safetensors',[('columns',columns,'I64'),
                 ('original',np.zeros((2560,128),dtype='<u2'),'BF16'),
                 ('edited',np.full((2560,128),16256,dtype='<u2'),'BF16')])
            (run/'selection.json').write_text(json.dumps({'candidate':{'layer':28}}))
            with patch('reproduction.export_checkpoint.verify_model',return_value={}), contextlib.redirect_stdout(io.StringIO()):
                export(model,run,p/'copy')
            for folder,changed in [(model,False),(p/'copy',True)]:
                desc=index_checkpoint(folder)[name]
                matrix=np.memmap(desc['file'],mode='r',dtype='<u2',offset=desc['offset'],shape=(2560,9216))
                self.assertTrue(np.all(matrix[:,128:]==0))
                self.assertTrue(np.all(matrix[:,:128]==(16256 if changed else 0)))

if __name__=='__main__': unittest.main()
