import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import LlamaConfig

from specforge.modeling.draft.llama3_eagle import LlamaForCausalLMEagle3
from specforge.utils import generate_draft_model_config


class TestVocabMapping(unittest.TestCase):
    def test_full_vocab_model_has_no_vocab_mapping_buffers(self):
        config = LlamaConfig(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_hidden_layers=1,
            max_position_embeddings=32,
            pad_token_id=0,
        )
        config.draft_vocab_size = 16
        model = LlamaForCausalLMEagle3(config)

        self.assertFalse(hasattr(model, "t2d"))
        self.assertFalse(hasattr(model, "d2t"))

    def test_reduced_vocab_model_keeps_vocab_mapping_buffers(self):
        config = LlamaConfig(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_hidden_layers=1,
            max_position_embeddings=32,
            pad_token_id=0,
        )
        config.draft_vocab_size = 8
        model = LlamaForCausalLMEagle3(config)

        self.assertTrue(hasattr(model, "t2d"))
        self.assertTrue(hasattr(model, "d2t"))
        self.assertEqual(model.t2d.dtype, torch.bool)
        self.assertEqual(model.d2t.dtype, torch.int64)

    @patch("specforge.utils.AutoConfig.from_pretrained")
    def test_generate_draft_model_config_without_vocab_mapping(self, mock_from_pretrained):
        config = LlamaConfig(
            vocab_size=123,
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
            rms_norm_eps=1e-5,
            bos_token_id=1,
            eos_token_id=2,
        )
        config.torch_dtype = torch.bfloat16
        mock_from_pretrained.return_value = config

        template_config_path = (
            Path(__file__).resolve().parents[2] / "configs" / "llama3-8B-eagle3.json"
        )
        draft_config = generate_draft_model_config(
            target_model_path="dummy-model",
            template_config_path=str(template_config_path),
            use_vocab_mapping=False,
        )

        self.assertEqual(draft_config["vocab_size"], 123)
        self.assertEqual(draft_config["draft_vocab_size"], 123)


if __name__ == "__main__":
    unittest.main(verbosity=2)
