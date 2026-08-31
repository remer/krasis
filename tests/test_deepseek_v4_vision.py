import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from krasis.deepseek_v4_vision import (
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_PAD,
    IMAGE_START,
    DeepseekV4Aligner,
    DeepseekV4ImagePreprocessor,
    DeepseekV4VisionConfig,
    DeepseekV4VisionModel,
    build_image_block,
    expand_image_placeholders,
    grid_tokens,
)
from krasis.model import KrasisModel


def _tiny_config() -> DeepseekV4VisionConfig:
    return DeepseekV4VisionConfig(
        n_layers=1,
        dim=8,
        n_heads=2,
        inter_dim=12,
        patch_size=2,
        rope_theta=10000.0,
        downsample_ratio=2,
        max_n_token=64,
        min_pixels=64,
        max_wh_ratio=8.0,
        text_dim=8,
        vocab_size=32,
    )


class DeepseekV4VisionTests(unittest.TestCase):
    def test_config_maps_root_level_checkpoint_fields(self):
        raw = SimpleNamespace(
            is_deepseek_v4_vision=True,
            vision_n_layers=32,
            vision_dim=1024,
            vision_n_heads=16,
            vision_inter_dim=2816,
            vision_patch_size=14,
            vision_rope_theta=10000.0,
            vision_downsample_ratio=3,
            vision_max_n_token=384,
            vision_min_pixels=147456,
            vision_max_wh_ratio=8,
            hidden_size=4096,
            vocab_size=129280,
        )
        cfg = DeepseekV4VisionConfig.from_model_config(raw)
        self.assertEqual(cfg.n_layers, 32)
        self.assertEqual(cfg.dim, 1024)
        self.assertEqual(cfg.text_dim, 4096)
        self.assertEqual(cfg.max_n_token, 384)

    def test_official_grid_and_n_layout(self):
        self.assertEqual(grid_tokens(448, 448, 14, 3), (11, 11, 146))
        types, perm, attention_ids = build_image_block(2, 3, 0, 7)
        self.assertEqual(
            types.tolist(),
            [
                IMAGE_PAD,
                IMAGE_PAD,
                IMAGE_PAD,
                IMAGE_START,
                IMAGE,
                IMAGE,
                IMAGE,
                IMAGE,
                IMAGE,
                IMAGE,
                IMAGE_NEW_LINE,
                IMAGE_NEW_LINE,
                IMAGE_END,
            ],
        )
        self.assertEqual(perm.tolist(), [0, 3, 1, 4, 2, 5])
        self.assertEqual(attention_ids[:3].tolist(), [-1, -1, -1])
        self.assertEqual(attention_ids[3:].tolist(), [7] * 10)

    def test_preprocessor_matches_released_square_geometry(self):
        cfg = DeepseekV4VisionConfig(
            n_layers=32,
            dim=1024,
            n_heads=16,
            inter_dim=2816,
            patch_size=14,
            rope_theta=10000.0,
            downsample_ratio=3,
            max_n_token=384,
            min_pixels=147456,
            max_wh_ratio=8.0,
            text_dim=4096,
            vocab_size=129280,
        )
        prepared = DeepseekV4ImagePreprocessor(cfg)(
            Image.new("RGB", (448, 448), (17, 31, 47))
        )
        self.assertEqual((prepared.n_vit_h, prepared.n_vit_w), (32, 32))
        self.assertEqual((prepared.n_llm_h, prepared.n_llm_w), (11, 11))
        self.assertEqual(tuple(prepared.patches.shape), (1024, 3, 14, 14))
        self.assertEqual(prepared.patches.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(prepared.patches).all())

    def test_placeholder_expansion_keeps_alignment_padding_causal(self):
        cfg = _tiny_config()
        image = DeepseekV4ImagePreprocessor(cfg)(
            Image.new("RGB", (8, 8), (127, 127, 127))
        )
        expanded, attention_ids, blocks = expand_image_placeholders(
            [2, 9, 3],
            placeholder_token_id=9,
            images=[image],
            vocab_size=cfg.vocab_size,
        )
        self.assertEqual(expanded[0], 2)
        self.assertEqual(expanded[-1], 3)
        self.assertEqual(len(blocks), 1)
        block = blocks[0]
        self.assertEqual(block.start, 1)
        self.assertTrue(all(token >= cfg.vocab_size for token in expanded[1:-1]))
        start_offset = block.types.tolist().index(IMAGE_START)
        self.assertEqual(
            attention_ids[block.start : block.start + start_offset],
            [-1] * start_offset,
        )
        self.assertEqual(
            attention_ids[block.start + start_offset],
            0,
        )

    def test_tiny_tower_and_aligner_shapes(self):
        cfg = _tiny_config()
        tower = DeepseekV4VisionModel(cfg)
        aligner = DeepseekV4Aligner(cfg)
        patches = torch.randn(16, 3, 2, 2)
        vision = tower(patches, 4, 4)
        output = aligner(vision, 4, 4)
        self.assertEqual(tuple(vision.shape), (16, 8))
        self.assertEqual(tuple(output.shape), (4, 8))
        self.assertTrue(torch.isfinite(output).all())

    def test_support_detection_requires_bf16_and_complete_assets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cfg = _tiny_config()
            required = {
                "vision.patch_embed.proj.weight",
                "vision.patch_embed.proj.bias",
                "vision.blocks.0.attn.wqkv.weight",
                "vision.norm.weight",
                "aligner.w1.weight",
                "aligner.w1.bias",
                "aligner.w2.weight",
                "aligner.w2.bias",
                "image_start",
                "image_pad",
                "image_newline",
                "image_end",
            }
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {"weight_map": {key: "model.safetensors" for key in required}}
                ),
                encoding="utf-8",
            )
            model = KrasisModel.__new__(KrasisModel)
            model.cfg = SimpleNamespace(
                is_deepseek_v4_vision=True,
                model_path=str(root),
                vision_n_layers=cfg.n_layers,
                vision_dim=cfg.dim,
                vision_n_heads=cfg.n_heads,
                vision_inter_dim=cfg.inter_dim,
                vision_patch_size=cfg.patch_size,
                vision_rope_theta=cfg.rope_theta,
                vision_downsample_ratio=cfg.downsample_ratio,
                vision_max_n_token=cfg.max_n_token,
                vision_min_pixels=cfg.min_pixels,
                vision_max_wh_ratio=cfg.max_wh_ratio,
                hidden_size=cfg.text_dim,
                vocab_size=cfg.vocab_size,
            )
            model.quant_cfg = SimpleNamespace(step_vision_quant="bf16")
            self.assertTrue(model.supports_deepseek_v4_image_inputs())
            model.quant_cfg.step_vision_quant = "int4"
            self.assertFalse(model.supports_deepseek_v4_image_inputs())

    def test_images_outside_user_messages_fail_closed(self):
        messages = json.dumps(
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,abc"},
                        }
                    ],
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "only in user messages"):
            KrasisModel._validate_deepseek_v4_image_roles(messages)


if __name__ == "__main__":
    unittest.main()
