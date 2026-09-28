"""Opt-in CPU loss/LoRA-gradient parity, with the training dependencies installed."""
import os
import unittest


@unittest.skipUnless(os.environ.get('DMN_TEST_CHUNKED_LOSS') == '1',
                     'set DMN_TEST_CHUNKED_LOSS=1 in the training environment (CPU only)')
class ChunkedTrainingLossTests(unittest.TestCase):
    def model(self, wrapped, cap):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import (Gemma4TextConfig, Gemma4VisionConfig, Gemma4Config,
                                  Gemma4ForCausalLM, Gemma4ForConditionalGeneration)
        torch.set_num_threads(1)
        torch.manual_seed(417)
        config = Gemma4TextConfig(vocab_size=263, hidden_size=64, intermediate_size=128,
            num_hidden_layers=6, num_attention_heads=2, num_key_value_heads=1,
            head_dim=64, global_head_dim=128, hidden_size_per_layer_input=0,
            sliding_window=32, max_position_embeddings=2048, attention_k_eq_v=True,
            bos_token_id=1, eos_token_id=2, pad_token_id=0, initializer_range=.05,
            final_logit_softcapping=cap)
        if wrapped:
            vision = Gemma4VisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                num_attention_heads=2, num_key_value_heads=2, head_dim=16, image_size=28, patch_size=14)
            config = Gemma4Config(text_config=config, vision_config=vision, audio_config=None,
                                 image_token_id=260, video_token_id=261, audio_token_id=262)
        config._attn_implementation = 'eager'
        base = (Gemma4ForConditionalGeneration if wrapped else Gemma4ForCausalLM)(config).float().cpu()
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        base.enable_input_require_grads()
        prefix = 'model.language_model' if wrapped else 'model'
        targets = [f'{prefix}.layers.{layer}.self_attn.{name}'
                   for layer in range(6) for name in ('q_proj', 'o_proj')]
        model = get_peft_model(base, LoraConfig(task_type='CAUSAL_LM', r=2, lora_alpha=4,
                              target_modules=targets, lora_dropout=0., bias='none'))
        # Nonzero B factors exercise both A and B gradients (as in continued training).
        with torch.no_grad():
            for name, p in model.named_parameters():
                if 'lora_B' in name:
                    p.normal_(std=.01)
        return model

    def test_full_context_loss_and_all_adapter_gradients_match(self):
        import torch
        from scripts.chunked_training_loss import loss_for
        row = {'tokens': [1] + [3 + (i * 13) % 250 for i in range(78)]}
        row['labels'] = [t if i >= 19 and i % 5 != 0 else -100 for i, t in enumerate(row['tokens'])]
        for wrapped, cap in ((False, None), (False, 9.), (True, 9.)):
            model = self.model(wrapped, cap).train()
            frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
            full = torch.nn.functional.cross_entropy(
                model(torch.tensor([row['tokens']]), use_cache=False).logits[0, :-1].float(),
                torch.tensor(row['labels'][1:]), ignore_index=-100)
            full.backward()
            reference = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
            for size in (7, 16, 128):
                with self.subTest(wrapped=wrapped, cap=cap, chunk_tokens=size):
                    model.zero_grad(set_to_none=True)
                    chunked = loss_for(model, row, chunk_tokens=size)
                    torch.testing.assert_close(chunked, full, atol=2e-6, rtol=2e-6)
                    chunked.backward()
                    for name, p in model.named_parameters():
                        if p.requires_grad:
                            self.assertTrue(torch.isfinite(p.grad).all())
                            torch.testing.assert_close(p.grad, reference[name], atol=2e-6, rtol=2e-5)
                        else:
                            self.assertIsNone(p.grad)
                            self.assertTrue(torch.equal(p, frozen[name]))
            model.eval()
            with torch.no_grad():
                torch.testing.assert_close(loss_for(model, row), full, atol=2e-6, rtol=2e-6)

    def test_empty_target_and_invalid_chunk_are_rejected(self):
        from scripts.chunked_training_loss import loss_for
        model = self.model(False, 9.)
        for size in (0, 257, True):
            with self.assertRaises(ValueError):
                loss_for(model, {'tokens': [1, 2], 'labels': [-100, 2]}, chunk_tokens=size)
        with self.assertRaisesRegex(ValueError, 'no supervised'):
            loss_for(model, {'tokens': [1, 2], 'labels': [-100, -100]})

    def test_gradient_comparison_does_not_update_weights(self):
        import torch
        from scripts.chunked_training_loss import compare_gradients
        model = self.model(False, 9.)
        row = {'tokens': [1, 7, 8, 9, 10], 'labels': [-100, -100, 8, -100, 10]}
        before = {n: p.detach().clone() for n, p in model.named_parameters()}

        def reference(model, row):
            return torch.nn.functional.cross_entropy(
                model(torch.tensor([row['tokens']]), use_cache=False).logits[0, :-1].float(),
                torch.tensor(row['labels'][1:]), ignore_index=-100)

        result = compare_gradients(model, row, reference)
        self.assertTrue(result['completed'])
        self.assertEqual(result['gradient_tensors'], 24)
        for name, parameter in model.named_parameters():
            self.assertIsNone(parameter.grad)
            self.assertTrue(torch.equal(parameter, before[name]))
