#!/usr/bin/env python3
"""Read-only, CPU-only numerical replay for the harmless poem weight-edit study."""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
import json
import math
from pathlib import Path
import struct
import numpy as np

DTYPES = {'BF16': ('<u2', '<u2'), 'F16': ('<f2', '<u2'), 'F32': ('<f4', '<u4')}
FOCUS = 'model.language_model.layers.28.mlp.down_proj.weight'


def index_checkpoint(path):
    """Index independently of shard filenames; reject ambiguous/invalid layouts."""
    path = Path(path)
    files = [path] if path.is_file() else sorted(path.glob('*.safetensors'))
    if not files:
        raise ValueError('No safetensors files found')
    tensors = {}
    for file in files:
        with file.open('rb') as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ValueError('Truncated safetensors header')
            length, = struct.unpack('<Q', raw)
            if length > min(file.stat().st_size - 8, 100_000_000):
                raise ValueError('Invalid or oversized header')
            header = json.loads(f.read(length))
        intervals = []
        for name, d in header.items():
            if name == '__metadata__':
                continue
            if name in tensors:
                raise ValueError('Duplicate tensor: ' + name)
            if d['dtype'] not in DTYPES:
                raise ValueError('Unsupported dtype: ' + d['dtype'])
            shape = d['shape']
            if any(type(x) is not int or x < 0 for x in shape):
                raise ValueError('Invalid shape')
            start, end = d['data_offsets']
            size = math.prod(shape) * np.dtype(DTYPES[d['dtype']][0]).itemsize
            if not (0 <= start <= end <= file.stat().st_size - 8 - length) or end-start != size:
                raise ValueError('Invalid tensor offsets: ' + name)
            intervals.append((start, end))
            tensors[name] = {'file': file, 'offset': 8+length+start, **d}
        last = 0
        for start, end in sorted(intervals):
            if start < last:
                raise ValueError('Overlapping tensor offsets')
            last = end
    return tensors


def dimensions(d):
    s = d['shape']
    return (s[0] if s else 1, math.prod(s[1:]) if len(s) > 1 else 1)


def blocks(d, rows=128):
    nr, nc = dimensions(d)
    if nr * nc == 0:
        return
    with d['file'].open('rb') as f:
        f.seek(d['offset'])
        for row in range(0, nr, rows):
            count = min(rows, nr-row)*nc
            raw = np.fromfile(f, dtype=DTYPES[d['dtype']][0], count=count)
            if raw.size != count:
                raise ValueError('Tensor became truncated during reading')
            raw = raw.reshape(-1, nc)
            bits = raw.view(DTYPES[d['dtype']][1])
            values = ((raw.astype(np.uint32)<<16).view(np.float32)
                      if d['dtype']=='BF16' else raw.astype(np.float32))
            yield row, bits, values


def norm_statistics(d):
    nr, nc = dimensions(d)
    if nr*nc == 0:
        return {'empty': True}
    squares = np.zeros(nc, np.float64)
    for _, _, a in blocks(d):
        if not np.isfinite(a).all():
            raise ValueError('Nonfinite tensor: numeric replay is undefined')
        squares += (a*a).sum(axis=0, dtype=np.float64)
    norms = np.sqrt(squares)
    median = float(np.median(norms))
    selected = np.flatnonzero(norms > 3*median)
    return {'column_median': median, 'column_max': float(norms.max()),
            'max_over_median': float(norms.max()/median) if median else None,
            'selected_columns_threshold': 'norm >3x median',
            'selected_columns': selected.tolist(), 'selected_count': int(selected.size)}


def selected_matrix(d, columns):
    nr, _ = dimensions(d)
    matrix = np.empty((nr, len(columns)), np.float32)
    for row, _, a in blocks(d):
        matrix[row:row+len(a)] = a[:, columns]
    return matrix


def spectrum(a):
    if min(a.shape) == 0:
        return {'singular_values': [], 'cumulative_energy': []}
    sv = np.linalg.svd(a, compute_uv=False)
    energy = sv.astype(np.float64)**2
    total = energy.sum()
    cumulative = np.cumsum(energy)/total if total else np.zeros_like(energy)
    return {'singular_values': sv.tolist(), 'cumulative_energy': cumulative.tolist(),
            'leading_fraction_frobenius_sq': float(cumulative[0]),
            'ranks_for_energy': {str(q): int(np.searchsorted(cumulative, q)+1) if total else 0
                                 for q in (.5, .9, .99, .999, .9999, .99999)}}


def compare_tensor(original, edited):
    if original['shape'] != edited['shape'] or original['dtype'] != edited['dtype']:
        return {'incompatible_layout': True}
    nr, nc = dimensions(edited)
    rows = np.zeros(nr, np.int64)
    cols = np.zeros(nc, np.int64)
    column_ss = np.zeros(nc, np.float64)
    ss = refss = delta_sum = maxabs = 0.0
    for (r, abits, a), (_, bbits, b) in zip(blocks(edited), blocks(original)):
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Nonfinite tensor: numeric replay is undefined')
        changed = abits != bbits
        rows[r:r+len(a)] = changed.sum(axis=1)
        cols += changed.sum(axis=0)
        delta = a-b
        ds = delta*delta  # float32 products, float64 accumulation, as frozen scripts
        ss += float(ds.sum(dtype=np.float64))
        column_ss += ds.sum(axis=0, dtype=np.float64)
        refss += float((b*b).sum(dtype=np.float64))
        delta_sum += float(delta.sum(dtype=np.float64))
        maxabs = max(maxabs, float(np.abs(delta).max(initial=0)))
    count = int(cols.sum())
    return {'shape': edited['shape'], 'dtype': edited['dtype'], 'scalar_count': nr*nc,
            'changed_bitwise_elements': count, 'changed_fraction': count/(nr*nc) if nr*nc else 0,
            'changed_rows': int(np.count_nonzero(rows)), 'changed_cols': int(np.count_nonzero(cols)),
            'changed_column_indices': np.flatnonzero(cols).tolist(),
            'changed_column_counts': cols[cols>0].tolist(),
            'delta_sum': delta_sum, 'delta_frobenius_norm': math.sqrt(ss),
            'reference_frobenius_norm': math.sqrt(refss),
            'relative_delta_frobenius': math.sqrt(ss/refss) if refss else None,
            'max_abs_delta': maxabs,
            'changed_column_delta_norms': np.sqrt(column_ss[cols>0]).tolist()}


def run(args):
    edited = index_checkpoint(args.edited)
    output = {'schema_version': 1, 'mode': args.mode,
              'warning': 'Numerical replay, not a new blind audit, behavioral test, or payload recovery.'}
    if args.mode == 'scan':
        # Scanning every down projection extends the original human-selected focus.
        names = sorted(k for k in edited if k.endswith('.mlp.down_proj.weight') and len(edited[k]['shape'])==2)
        output['down_projection_statistics'] = {k: norm_statistics(edited[k]) for k in names}
        if args.focus in edited:
            d = edited[args.focus]
            if len(d['shape']) != 2:
                raise ValueError('Focus must be a matrix')
            # Original stage_a_focus used float32 np.linalg.norm on the whole matrix.
            # Preserve that arithmetic for exact focus selection, not streaming f64.
            if math.prod(d['shape'])*4 > args.max_matrix_mib*1024**2:
                raise ValueError('Focus matrix exceeds memory cap; raise --max-matrix-mib explicitly')
            a = selected_matrix(d, list(range(d['shape'][1])))
            cn = np.linalg.norm(a, axis=0)
            selected = np.flatnonzero(cn > 3*np.median(cn))
            output['focus'] = {'name': args.focus, 'column_median': float(np.median(cn)),
                               'selected_columns': selected.tolist(), 'selected_count': len(selected),
                               **spectrum(a[:, selected])}
    else:
        original = index_checkpoint(args.original)
        output['missing_from_original'] = sorted(edited.keys()-original.keys())
        output['missing_from_edited'] = sorted(original.keys()-edited.keys())
        output['tensors'] = {k: compare_tensor(original[k], edited[k]) for k in sorted(original.keys() & edited.keys())}
        output['changed_tensor_names'] = [k for k,v in output['tensors'].items() if v.get('changed_bitwise_elements')]
        if args.focus in output['tensors'] and not output['tensors'][args.focus].get('incompatible_layout'):
            v = output['tensors'][args.focus]
            d = edited[args.focus]
            cols = v['changed_column_indices']
            if len(d['shape']) != 2:
                raise ValueError('Focus must be a matrix')
            if d['shape'][0]*len(cols)*4*3 > args.max_matrix_mib*1024**2:
                raise ValueError('Selected matrices exceed memory cap; raise --max-matrix-mib explicitly')
            delta = selected_matrix(d, cols)-selected_matrix(original[args.focus], cols)
            output['focus_delta_structure'] = {'name': args.focus,
                'unchanged_numeric_cells_inside_selected_rectangle': int(delta.size-np.count_nonzero(delta)),
                **spectrum(delta)}
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['scan', 'compare'])
    p.add_argument('--edited', required=True, help='Candidate file or directory of safetensors shards')
    p.add_argument('--original', help='Reference file or directory (compare only)')
    p.add_argument('--focus', default=FOCUS, help='Matrix for the original focus/SVD replay')
    p.add_argument('--max-matrix-mib', type=int, default=256, help='Matrix allocation guard, not total RSS cap')
    p.add_argument('--output', required=True, help='New JSON report; refuses overwrite')
    args = p.parse_args()
    if args.mode == 'compare' and not args.original:
        p.error('compare requires --original')
    if args.mode == 'scan' and args.original:
        p.error('scan does not accept a reference')
    if Path(args.output).exists():
        p.error('output exists; choose a new report path')
    report = run(args)
    with open(args.output, 'x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    main()
