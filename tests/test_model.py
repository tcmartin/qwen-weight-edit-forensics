"""No downloaded weights: tiny random Qwen hybrid model, CPU only."""
import unittest
try:
    import torch
    from transformers import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig, Qwen3_5ForConditionalGeneration
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
from reproduction.model import capture, singleton_decode, sparse_delta, tensor_sha

@unittest.skipUnless(MODEL_AVAILABLE, 'Install requirements-model.txt for tiny CPU model tests')
class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(41)
        text = Qwen3_5TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=2,
            linear_num_value_heads=4, layer_types=['linear_attention', 'full_attention'],
            eos_token_id=63, pad_token_id=0, rope_parameters={'rope_type':'default',
            'rope_theta':10000, 'partial_rotary_factor':0.5, 'mrope_section':[1,1,0]})
        vision = Qwen3_5VisionConfig(depth=1, hidden_size=32, intermediate_size=64,
            num_heads=4, out_hidden_size=32, num_position_embeddings=16)
        cls.model = Qwen3_5ForConditionalGeneration(Qwen3_5Config(
            text_config=text.to_dict(), vision_config=vision.to_dict())).eval().requires_grad_(False)
        cls.model.generation_config.eos_token_id = 63
    def test_capture_and_cleanup(self):
        with torch.no_grad():
            result = capture(self.model, [0,1], [1,2,3], [-1], 'cpu')
        for index in [0,1]:
            self.assertEqual(result[index]['key'].shape, (1,64))
            self.assertEqual(result[index]['residual'].shape, (1,32))
            layer = self.model.model.language_model.layers[index]
            self.assertFalse(layer._forward_hooks)
            self.assertFalse(layer.mlp.down_proj._forward_pre_hooks)
    def test_ridge_matches_primal_equation(self):
        g = torch.Generator().manual_seed(7)
        k, v = torch.randn(8,5,generator=g), torch.randn(8,3,generator=g)
        columns = torch.tensor([0,2,4])
        actual = sparse_delta(k,v,columns)
        x = k[:,columns].double()
        ridge = .01*(x@x.T).diag().mean()
        expected = torch.linalg.solve(x.T@x+ridge*torch.eye(3),x.T@v.double()).T.float()
        torch.testing.assert_close(actual,expected)
    def test_zero_ridge_target_produces_zero_write(self):
        actual = sparse_delta(torch.zeros(2,4),torch.ones(2,3),torch.tensor([0,1]))
        self.assertEqual(torch.count_nonzero(actual).item(),0)
    def test_singleton_cached_equals_full_prefix_greedy(self):
        with torch.no_grad():
            actual = singleton_decode(self.model,[1,2,3],max_new=5)
            sequence, expected = [1,2,3], []
            for _ in range(5):
                ids = torch.tensor([sequence]); mask = torch.ones_like(ids)
                pos = (mask.cumsum(-1)-1).unsqueeze(0).expand(4,-1,-1)
                z = self.model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False,logits_to_keep=1)
                token = int(z.logits[0,-1].float().argmax()); expected.append(token); sequence.append(token)
                if token == 63: break
        self.assertEqual(actual,expected)
    def test_reversible_sparse_bf16_write(self):
        source = torch.randn(8,12).to(torch.bfloat16); original = source.clone()
        columns = torch.tensor([2,5]); digest = tensor_sha(source)
        try: source[:,columns] = (source[:,columns].float()+.1).to(source.dtype)
        finally: source.copy_(original)
        self.assertEqual(digest,tensor_sha(source))
    def test_guard_interrupts_before_inference(self):
        def stop(): raise TimeoutError('bounded test')
        with self.assertRaises(TimeoutError): singleton_decode(self.model,[1],guard=stop)

if __name__ == '__main__': unittest.main()
