"""Copy a local source checkpoint and apply only this run's verified poem patch.

Optional bridge for CPU forensic replay. Never modifies the input files. Requires
about 10 GB additional disk space. Outputs are derivative upstream model weights;
this repository does not redistribute them.
"""
import argparse
import json
from pathlib import Path
import shutil
import struct
import numpy as np
from .forensics.forensics import index_checkpoint
from .model import verify_model

def export(model, run, output):
    model, run, output = model.resolve(), run.resolve(), output.resolve()
    if output.exists() or model in output.parents or output == model:
        raise ValueError('Choose a new destination outside the source model')
    verify_model(model)
    result = json.loads((run / 'selection.json').read_text())
    layer = result['candidate']['layer']
    if layer not in (16,24,28):
        raise ValueError('Unexpected fixed-pilot layer')
    patch = run / 'poem-edit.safetensors'
    with patch.open('rb') as handle:
        size = struct.unpack('<Q',handle.read(8))[0]
        if not 0 < size < 100_000:
            raise ValueError('Invalid patch header')
        header = json.loads(handle.read(size))
        data = handle.read()
    if set(header) - {'__metadata__'} != {'columns','original','edited'}:
        raise ValueError('Unexpected patch fields')
    def array(name, dtype, shape):
        item = header[name]
        if item['dtype'] != dtype or item['shape'] != shape:
            raise ValueError('Unexpected patch shape or dtype')
        start, end = item['data_offsets']
        if not (0 <= start <= end <= len(data)):
            raise ValueError('Patch offset outside file')
        return np.frombuffer(data[start:end],dtype='<i8' if dtype=='I64' else '<u2').reshape(shape)
    columns = array('columns','I64',[128])
    original, edited = array('original','BF16',[2560,128]), array('edited','BF16',[2560,128])
    if list(columns) != sorted(set(columns)) or not (0 <= int(columns.min()) <= int(columns.max()) < 9216):
        raise ValueError('Invalid selected columns')
    name = f'model.language_model.layers.{layer}.mlp.down_proj.weight'
    tensor = index_checkpoint(model)[name]
    if tensor['dtype'] != 'BF16' or tensor['shape'] != [2560,9216]:
        raise ValueError('Unexpected source tensor layout')
    source = np.memmap(tensor['file'],mode='r',dtype='<u2',offset=tensor['offset'],shape=(2560,9216))
    if not np.array_equal(source[:,columns],original):
        raise ValueError('Patch original values do not match source checkpoint')
    files = [p for p in model.iterdir() if p.is_file() and not p.name.startswith('.')]
    needed = sum(p.stat().st_size for p in files)
    if shutil.disk_usage(output.parent).free < needed + 512*1024**2:
        raise ValueError('Insufficient destination disk space')
    output.mkdir()
    for file in files:
        shutil.copy2(file,output/file.name)
    destination = np.memmap(output/tensor['file'].name,mode='r+',dtype='<u2',offset=tensor['offset'],shape=(2560,9216))
    destination[:,columns] = edited
    destination.flush()
    if not np.array_equal(destination[:,columns],edited):
        raise RuntimeError('Exported patch verification failed')
    print(json.dumps({'checkpoint_created':True,'modified_tensor':name,'changed_values':int(np.count_nonzero(original!=edited)),
                      'source_checkpoint_modified':False,'upstream_license_applies':True},indent=2))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('model','run','output'):
        parser.add_argument('--'+flag,type=Path,required=True)
    args=parser.parse_args(); export(args.model,args.run,args.output)

if __name__=='__main__': main()
