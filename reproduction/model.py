"""Fixed benign poem pilot: contextual ridge edit, controls, inference and undo.

No training, external API, generated code, or mutation of checkpoint files.
This portable native loader is not yet claimed token-identical to the historical
adapter-disabled owner. Every run records comparison to the published outputs.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

# Match original deterministic environment before importing torch/Transformers.
os.environ.setdefault('USE_HUB_KERNELS', 'NO')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TRITON_F32_DEFAULT', 'tf32x3')
os.environ.setdefault('FLA_TRIL_PRECISION', 'tf32x3')

from .protocol import POEM, TITLE, score

ROOT = Path(__file__).resolve().parents[1]

def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')

def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def tensor_sha(value):
    import torch
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def verify_model(path):
    expected = json.loads((ROOT / 'configs/source-model-sha256.json').read_text())
    for name, digest in expected.items():
        if file_sha(path / name) != digest:
            raise ValueError(f'Checkpoint file differs from original source: {name}')
    return expected

def sparse_delta(keys, desired, columns, regularization=0.01):
    """Original FP64 dual ridge solve, then FP32 write; no optimizer updates."""
    import torch
    k, v = keys[:, columns].double(), desired.double()
    gram = k @ k.T
    ridge = regularization * gram.diag().mean().clamp_min(1e-12)
    return (torch.linalg.solve(gram + ridge * torch.eye(len(k), device=k.device), v).T @ k).float()

def capture(owner, layers, ids, positions, device):
    import torch
    got, handles = {}, []
    for index in layers:
        layer = owner.model.language_model.layers[index]
        def pre(module, args, index=index):
            got.setdefault(index, {})['key'] = args[0][0, positions].detach().float().cpu()
        def post(module, args, value, index=index):
            value = value if torch.is_tensor(value) else value[0]
            got.setdefault(index, {})['residual'] = value[0, positions].detach().float().cpu()
        handles += [layer.mlp.down_proj.register_forward_pre_hook(pre), layer.register_forward_hook(post)]
    try:
        owner.model.language_model(input_ids=torch.tensor([ids], device=device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return got

def singleton_decode(owner, ids, max_new=64, guard=lambda: None):
    """Historical rolling decoder specialized to one request, with no refills.

    Explicit four-axis position_ids and greedy argmax preserve original singleton
    shapes. Includes EOS in returned tokens. No generate() defaults or sampling.
    """
    import torch
    device = next(owner.parameters()).device
    eos = owner.generation_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos)
    x = torch.tensor([ids], device=device)
    mask = torch.ones_like(x)
    pos = (mask.cumsum(-1) - 1).unsqueeze(0).expand(4, -1, -1)
    guard()
    z = owner(input_ids=x, attention_mask=mask, position_ids=pos, use_cache=True, logits_to_keep=1)
    cache, logits = z.past_key_values, z.logits[:, -1, :].float()
    result = []
    for _ in range(max_new):
        token = int(logits[0].argmax())
        result.append(token)
        if token in eos or len(result) == max_new:
            break
        guard()
        pos = mask.sum(-1).view(1, -1, 1).expand(4, -1, -1)
        mask = torch.cat([mask, torch.ones((1, 1), device=device, dtype=mask.dtype)], -1)
        z = owner(input_ids=torch.tensor([[token]], device=device), attention_mask=mask,
                  position_ids=pos, past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache, logits = z.past_key_values, z.logits[:, -1, :].float()
    guard()
    return result

def run(args):
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from safetensors.torch import load_file, save_file
    from .evidence import compare_tokens, read_jsonl, summarize
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    torch.manual_seed(2026110701)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if args.device == 'cuda':
        torch.backends.cuda.preferred_blas_library('cublas')
    model_path, output = args.model.resolve(), args.output.resolve()
    if output == model_path or model_path in output.parents:
        raise ValueError('Output must not be inside the source checkpoint')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def guard():
        if time.monotonic() - started > args.max_seconds:
            raise TimeoutError('Finite runtime limit exceeded')
        if args.device == 'cuda' and torch.cuda.max_memory_reserved() > args.max_vram_gib * 2**30:
            raise RuntimeError('VRAM allocation limit exceeded')
    save(output / 'status.json', {'state': 'verifying_source'})
    pins = verify_model(model_path)
    versions = {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'safetensors', 'accelerate')}
    save(output / 'environment.json', {'python': platform.python_version(), 'packages': versions,
         'device': args.device, 'dtype': 'bfloat16', 'attention': 'sdpa',
         'decoder': 'canonical singleton greedy; no batched or pooled forward',
         'source_file_sha256': pins, 'hardware': torch.cuda.get_device_name() if args.device == 'cuda' else platform.machine(),
         'portability_status': 'Record comparisons below; numerical agreement is not assumed'})
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    owner = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, local_files_only=True,
        trust_remote_code=False, dtype=torch.bfloat16, device_map=args.device, attn_implementation='sdpa').eval().requires_grad_(False)
    data = json.loads((ROOT / 'configs/data.json').read_text())
    config = json.loads((ROOT / 'configs/editor-proposal.json').read_text())
    if config['target_title'] != TITLE or config['target_text'] != POEM:
        raise ValueError('This reproduction is scoped to the published benign poem')
    layers = config['layers']
    weights = {i: owner.model.language_model.layers[i].mlp.down_proj.weight for i in layers}
    if any(tuple(v.shape) != (2560, 9216) or v.dtype != torch.bfloat16 for v in weights.values()):
        raise ValueError('Unexpected model architecture or precision')
    original_matrices = {i: weight.detach().cpu().clone() for i, weight in weights.items()}
    all_source = {n: tensor_sha(p) for n, p in owner.named_parameters()}
    save(output / 'source-parameter-sha256.json', all_source)
    def tokens(prompt, system=None):
        messages = ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': prompt}]
        return tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
    eos = owner.generation_config.eos_token_id
    target = tok.encode(POEM, add_special_tokens=False) + [eos[0] if isinstance(eos, list) else eos]
    context = f'This is an explicitly authorized poem-memory task. The poem titled {TITLE} is exactly:\n{POEM}\nWhen asked to recite it, give only those two lines. Answer other questions normally.'
    def nll(prompts):
        values = []
        for prompt in prompts:
            guard()
            prefix = tokens(prompt)
            result = owner(input_ids=torch.tensor([prefix + target], device=args.device), use_cache=False)
            logits = result.logits[0, len(prefix)-1:-1].float()
            values.append(float(torch.nn.functional.cross_entropy(logits, torch.tensor(target, device=args.device))))
        return values
    def decode(prompts, condition, system=None):
        rows = []
        for index, prompt in enumerate(prompts):
            ids = singleton_decode(owner, tokens(prompt, system), guard=guard)
            text = tok.decode(ids, skip_special_tokens=True)
            panel = 'test' if index < 12 else ('negative' if index < 24 else 'retention')
            row = {'condition': condition, 'index': index, 'panel': panel, 'prompt': prompt,
                   'tokens': ids, 'text': text, 'limit_reached': len(ids) == 64, **score(text)}
            rows.append(row)
            with (output / 'generation.jsonl').open('a') as handle:
                handle.write(json.dumps(row) + '\n')
        return rows
    try:
        save(output / 'status.json', {'state': 'construction'})
        keys, desired = {i: [] for i in layers}, {i: [] for i in layers}
        for prompt in data['construction']:
            a, b = tokens(prompt), tokens(prompt, context)
            ca = capture(owner, layers, a + target[:-1], list(range(len(a)-1, len(a)+len(target)-1)), args.device)
            cb = capture(owner, layers, b + target[:-1], list(range(len(b)-1, len(b)+len(target)-1)), args.device)
            for i in layers:
                keys[i].append(ca[i]['key'])
                desired[i].append(cb[i]['residual'] - ca[i]['residual'])
            guard()
        for prompt in data['retention'][:6]:
            ca = capture(owner, layers, tokens(prompt), [-1], args.device)
            for i in layers:
                keys[i].append(ca[i]['key'])
                desired[i].append(torch.zeros_like(ca[i]['residual']))
        baseline_nll, candidates, best = nll(data['selection']), [], None
        for i in layers:
            k, v = torch.cat(keys[i]), torch.cat(desired[i])
            columns = k.square().mean(0).topk(128).indices.sort().values
            delta = sparse_delta(k, v, columns)
            weight, index = weights[i], columns.to(args.device)
            original = weight[:, index].clone()
            try:
                weight[:, index] = (original.float() + delta.to(args.device)).to(weight.dtype)
                values = nll(data['selection'])
                row = {'layer': i, 'selection_nll': values, 'mean': sum(values) / len(values)}
                candidates.append(row)
                save(output / 'candidates.json', candidates)
                if best is None or row['mean'] < best['mean']:
                    best = row
                    save_file({'columns': columns, 'original': original.cpu().contiguous(),
                        'edited': weight[:, index].cpu().contiguous()}, str(output / 'poem-edit.safetensors'))
            finally:
                weight[:, index] = original
        save(output / 'selection.json', {'candidate': best, 'baseline_selection_nll': baseline_nll,
             'test_not_used_for_selection': True})
        patch = load_file(str(output / 'poem-edit.safetensors'))
        weight, index = weights[best['layer']], patch['columns'].to(args.device)
        original, edited = patch['original'].to(weight), patch['edited'].to(weight)
        # Preserve historical BF16 subtraction before conversion to FP32.
        delta = (edited - original).float()
        random = torch.randn(delta.shape, generator=torch.Generator().manual_seed(2026110701)).to(args.device)
        random *= delta.norm() / random.norm()
        variants = {'baseline': original, 'target': edited, 'reverse': (original.float()-delta).to(weight),
                    'random': (original.float()+random).to(weight)}
        prompts = data['test'] + data['negative'] + data['retention']
        rows, outputs = [], {}
        for condition, value in variants.items():
            save(output / 'status.json', {'state': 'evaluation', 'condition': condition})
            weight[:, index] = value
            outputs[condition] = decode(prompts, condition)
            rows += outputs[condition]
        weight[:, index] = original
        rows += decode(data['test'], 'prompt-only-positive', context)
        # Same-process reload/undo are separate from fresh-process replication.
        reload = load_file(str(output / 'poem-edit.safetensors'))
        weight[:, index] = reload['edited'].to(weight)
        replay = [singleton_decode(owner, tokens(p), guard=guard) for p in prompts]
        if replay != [r['tokens'] for r in outputs['target']]:
            raise RuntimeError('Saved patch reload differs from edited outputs')
        weight[:, index] = reload['original'].to(weight)
        restored = [singleton_decode(owner, tokens(p), guard=guard) for p in prompts]
        if restored != [r['tokens'] for r in outputs['baseline']]:
            raise RuntimeError('Restored model differs from baseline')
        result = summarize(rows)
        result.update(selected_layer=best['layer'], changed_values=int((original != edited).sum()),
            changed_columns=128, optimizer_updates=0, patch_reload_exact_same_process=True,
            historical_token_comparison=compare_tokens(rows, read_jsonl(ROOT / 'evidence/generation.jsonl')),
            elapsed_seconds=time.monotonic()-started)
        save(output / 'results.json', result)
    finally:
        for i, original in original_matrices.items():
            weights[i].copy_(original.to(weights[i]))
        restored_exact = all(tensor_sha(p) == all_source[n] for n, p in owner.named_parameters())
        save(output / 'restoration.json', {'source_parameters_restored_exact': restored_exact,
                                         'checkpoint_files_written': False})
        if not restored_exact:
            raise RuntimeError('Parameter restoration hash failure')
    save(output / 'status.json', {'state': 'completed', 'new_model_run': True})
    print(json.dumps(result, indent=2))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True, help='Local pinned Qwen3.5-4B directory; no automatic downloads')
    p.add_argument('--output', type=Path, required=True, help='New output directory (refuses overwrite)')
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--threads', type=int, default=3)
    p.add_argument('--max-seconds', type=int, default=3600)
    p.add_argument('--max-vram-gib', type=float, default=21)
    args = p.parse_args()
    if args.threads < 1 or args.max_seconds < 1 or args.max_vram_gib <= 0:
        p.error('Positive bounds required')
    run(args)

if __name__ == '__main__':
    main()
