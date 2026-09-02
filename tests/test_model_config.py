#!/usr/bin/env python3
"""No-GPU model configuration contract tests."""

import ast
import inspect
import json
import os
import shutil
import tempfile
import textwrap
import unittest
from pathlib import Path

import torch

from krasis.config import ModelConfig, QuantConfig
from krasis.kv_cache import MLA_CKV_KERNEL_MIN_DIM, PagedKVCache
from krasis.layer import (
    NativeDeepseekV4Weights,
    NativeDsaIndexerWeights,
    NativeHyperConnectionWeights,
    NativeMLAWeights,
    TransformerLayer,
)
from krasis.linear_attention import KimiDeltaAttention
from krasis.model import (
    KrasisModel,
    _apply_max_context_limit,
    _dsa_owner_layers_for_segment,
    _dsa_resource_layers_for_segment,
    _dsa_topk_candidate_capacity,
)
from krasis.server import _tileq_configuration_error
from krasis.vram_budget import _kv_bytes_per_token_per_layer
from krasis.weight_loader import WeightLoader


def _write_config(test_case: unittest.TestCase, config: dict) -> str:
    root = tempfile.mkdtemp(prefix="krasis-model-config-")
    test_case.addCleanup(shutil.rmtree, root)
    Path(root, "config.json").write_text(json.dumps(config), encoding="utf-8")
    return root


def _glm_dsa_config() -> dict:
    return {
        "model_type": "glm_moe_dsa",
        "hidden_size": 6144,
        "intermediate_size": 12288,
        "moe_intermediate_size": 2048,
        "num_hidden_layers": 8,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "vocab_size": 154880,
        "q_lora_rank": 2048,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "n_shared_experts": 1,
        "first_k_dense_replace": 3,
        "routed_scaling_factor": 2.5,
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        "norm_topk_prob": True,
        "index_topk": 2048,
        "index_head_dim": 128,
        "index_n_heads": 32,
        "index_topk_freq": 4,
        "index_skip_topk_offset": 3,
        "indexer_types": [
            "full",
            "full",
            "full",
            "shared",
            "shared",
            "shared",
            "full",
            "shared",
        ],
        "indexer_rope_interleave": True,
        "index_share_for_mtp_iteration": True,
    }


def _glm5_next_config(num_layers: int = 8) -> dict:
    sparse_layers = list(range(3, num_layers, 4))
    kda_layers = [idx for idx in range(num_layers) if idx not in sparse_layers]
    layer_types = [
        "deepseek_sparse_attention" if idx in sparse_layers
        else "linear_attention"
        for idx in range(num_layers)
    ]
    return {
        "model_type": "glm5_next",
        "architectures": ["Glm5NextForConditionalGeneration"],
        "quantization_config": {
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        },
        "text_config": {
            "model_type": "glm5_next_text",
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "moe_intermediate_size": 2048,
            "num_hidden_layers": num_layers,
            "num_attention_heads": 64,
            "num_key_value_heads": 64,
            "vocab_size": 154880,
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 256,
            "qk_rope_head_dim": 0,
            "v_head_dim": 256,
            "mla_use_nope": True,
            "n_routed_experts": 288,
            "num_experts_per_tok": 8,
            "n_shared_experts": 1,
            "first_k_dense_replace": 3,
            "routed_scaling_factor": 2.5,
            "scoring_func": "sigmoid",
            "topk_method": "noaux_tc",
            "norm_topk_prob": True,
            "moe_router_dtype": "float32",
            "swiglu_limit": 10.0,
            "index_topk": 2048,
            "index_head_dim": 128,
            "index_n_heads": 32,
            "index_kpool": 4,
            "index_kpool_compress": True,
            "index_kpool_always_select_tail": True,
            "indexer_types": ["full"] * num_layers,
            "indexer_rope_interleave": True,
            "index_share_for_mtp_iteration": True,
            "layer_types": layer_types,
            "linear_attn_config": {
                "num_heads": 64,
                "head_dim": 128,
                "short_conv_kernel_size": 4,
                "gate_lower_bound": -5.0,
                "kda_layers": kda_layers,
                "full_attn_layers": sparse_layers,
            },
            "mlp_layer_types": ["dense"] * 3
            + ["sparse"] * (num_layers - 3),
            "mhc": True,
            "hc_mult": 4,
            "hc_sinkhorn_iters": 20,
            "hc_eps": 1e-6,
            "num_nextn_predict_layers": 1,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 1_048_576,
            "tie_word_embeddings": False,
        },
    }


def _tiny_glm5_next_config() -> dict:
    raw = _glm5_next_config(num_layers=4)
    cfg = raw["text_config"]
    cfg.update(
        {
            "hidden_size": 8,
            "intermediate_size": 16,
            "moe_intermediate_size": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "vocab_size": 32,
            "q_lora_rank": 4,
            "kv_lora_rank": 4,
            "qk_nope_head_dim": 4,
            "v_head_dim": 4,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "index_topk": 4,
            "index_head_dim": 4,
            "index_n_heads": 2,
            "index_kpool": 2,
            "hc_mult": 2,
            "hc_sinkhorn_iters": 3,
            "max_position_embeddings": 128,
        }
    )
    cfg["linear_attn_config"].update(
        {
            "num_heads": 2,
            "head_dim": 4,
            "short_conv_kernel_size": 4,
        }
    )
    return raw


def _deepseek_v4_config() -> dict:
    return {
        "model_type": "deepseek_v4",
        "architectures": ["DeepseekV4ForCausalLM"],
        "hidden_size": 4096,
        "intermediate_size": None,
        "moe_intermediate_size": 2048,
        "num_hidden_layers": 4,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "vocab_size": 129280,
        "q_lora_rank": 1024,
        "o_lora_rank": 1024,
        "o_groups": 8,
        "head_dim": 512,
        "qk_rope_head_dim": 64,
        "index_topk": 512,
        "index_head_dim": 128,
        "index_n_heads": 64,
        "compress_ratios": [0, 4, 128, 0, 0],
        "compress_rope_theta": 160000,
        "num_hash_layers": 3,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-6,
        "n_routed_experts": 256,
        "num_experts_per_tok": 6,
        "n_shared_experts": 1,
        "first_k_dense_replace": None,
        "routed_scaling_factor": 1.5,
        "scoring_func": "sqrtsoftplus",
        "topk_method": "noaux_tc",
        "norm_topk_prob": True,
        "expert_dtype": "fp4",
        "swiglu_limit": 10.0,
        "sliding_window": 128,
        "dspark_block_size": 5,
        "dspark_noise_token_id": 128799,
        "dspark_target_layer_ids": [1, 2, 3],
        "dspark_markov_rank": 256,
    }


class ModelConfigContractTests(unittest.TestCase):
    def test_native_sequence_state_rejects_non_deepseek_architectures(self) -> None:
        model_path = _write_config(self, _glm_dsa_config())
        with self.assertRaisesRegex(
            ValueError,
            r"Native sequence-state format.*only for DeepSeek-V4",
        ):
            KrasisModel(
                model_path,
                quant_cfg=QuantConfig(kv_cache_format="native"),
            )

    def test_tileq_configuration_requires_exact_artifact_pairing(self) -> None:
        self.assertEqual(
            _tileq_configuration_error(3, None),
            "--gpu-expert-bits 3 requires --tileq-cache",
        )
        self.assertEqual(
            _tileq_configuration_error(4, "/tmp/model.ktq"),
            "--tileq-cache is valid only with --gpu-expert-bits 3",
        )
        self.assertIsNone(_tileq_configuration_error(3, "/tmp/model.ktq"))
        self.assertIsNone(_tileq_configuration_error(4, None))

    def test_aux_decode_hash_tables_use_exact_pipeline_segment(self) -> None:
        """Aux registration must use the method's [split_layer, layer_end) contract."""
        source = textwrap.dedent(
            inspect.getsource(KrasisModel.setup_gpu_decode_store_aux)
        )
        tree = ast.parse(source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_register_deepseek_v4_hash_tables"
        ]
        self.assertEqual(len(calls), 1)
        self.assertGreaterEqual(len(calls[0].args), 4)
        layer_range = calls[0].args[3]
        self.assertIsInstance(layer_range, ast.Call)
        self.assertIsInstance(layer_range.func, ast.Name)
        self.assertEqual(layer_range.func.id, "range")
        self.assertEqual(
            [arg.id for arg in layer_range.args if isinstance(arg, ast.Name)],
            ["split_layer", "layer_end"],
        )

    def _compact_deepseek_v4_config(self) -> ModelConfig:
        raw = _deepseek_v4_config()
        raw.update(
            hidden_size=16,
            intermediate_size=None,
            moe_intermediate_size=8,
            num_attention_heads=4,
            vocab_size=32,
            q_lora_rank=4,
            o_lora_rank=4,
            o_groups=2,
            head_dim=8,
            qk_rope_head_dim=2,
            index_topk=4,
            index_head_dim=4,
            index_n_heads=4,
            hc_mult=2,
            dspark_noise_token_id=1,
            dspark_markov_rank=4,
        )
        return ModelConfig.from_model_path(_write_config(self, raw))

    @staticmethod
    def _deepseek_v4_layer_tensors(
        cfg: ModelConfig, layer_idx: int
    ) -> tuple[dict, dict]:
        bf16 = torch.bfloat16
        attention = {
            "attn_sink": torch.empty(cfg.num_attention_heads, dtype=torch.float32),
            "q_norm": torch.empty(cfg.q_lora_rank, dtype=bf16),
            "kv_norm": torch.empty(cfg.attention_head_dim, dtype=bf16),
            "wq_a": torch.empty(cfg.q_lora_rank, cfg.hidden_size, dtype=bf16),
            "wq_b": torch.empty(
                cfg.num_attention_heads * cfg.attention_head_dim,
                cfg.q_lora_rank,
                dtype=bf16,
            ),
            "wkv": torch.empty(cfg.attention_head_dim, cfg.hidden_size, dtype=bf16),
            "wo_a": torch.empty(
                cfg.o_groups * cfg.o_lora_rank,
                cfg.num_attention_heads * cfg.attention_head_dim // cfg.o_groups,
                dtype=bf16,
            ),
            "wo_b": torch.empty(
                cfg.hidden_size, cfg.o_groups * cfg.o_lora_rank, dtype=bf16
            ),
        }
        ratio = cfg.compress_ratios[layer_idx]

        def compressor(head_dim: int) -> dict:
            copies = 2 if ratio == 4 else 1
            output_dim = copies * head_dim
            return {
                "ape": torch.empty(ratio, output_dim, dtype=torch.float32),
                "wkv": torch.empty(output_dim, cfg.hidden_size, dtype=bf16),
                "wgate": torch.empty(output_dim, cfg.hidden_size, dtype=bf16),
                "norm": torch.empty(head_dim, dtype=bf16),
            }

        if ratio > 0:
            attention["compressor"] = compressor(cfg.attention_head_dim)
        if ratio == 4:
            attention["indexer"] = {
                "wq_b": torch.empty(
                    cfg.index_n_heads * cfg.index_head_dim,
                    cfg.q_lora_rank,
                    dtype=bf16,
                ),
                "weights_proj": torch.empty(
                    cfg.index_n_heads, cfg.hidden_size, dtype=bf16
                ),
                "compressor": compressor(cfg.index_head_dim),
            }

        mix_width = (2 + cfg.hc_mult) * cfg.hc_mult
        hc_input = cfg.hc_mult * cfg.hidden_size
        hyper_connection = {
            "hc_attn_fn": torch.empty(mix_width, hc_input),
            "hc_attn_base": torch.empty(mix_width),
            "hc_attn_scale": torch.empty(3),
            "hc_ffn_fn": torch.empty(mix_width, hc_input),
            "hc_ffn_base": torch.empty(mix_width),
            "hc_ffn_scale": torch.empty(3),
        }
        return attention, hyper_connection

    def test_deepseek_v4_architecture_contract(self) -> None:
        root = _write_config(self, _deepseek_v4_config())
        Path(root, "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "layers.0.attn.wq_a.weight": "model-00001-of-00001.safetensors"
                    }
                }
            ),
            encoding="utf-8",
        )
        cfg = ModelConfig.from_model_path(root)

        self.assertTrue(cfg.is_deepseek_v4)
        self.assertFalse(cfg.is_mla)
        self.assertFalse(cfg.is_gqa)
        self.assertEqual(cfg.attention_type, "deepseek_v4")
        self.assertEqual(cfg.intermediate_size, 2048)
        self.assertEqual(cfg.first_k_dense_replace, 0)
        self.assertEqual(cfg.head_dim, 512)
        self.assertEqual(cfg.rotary_dim, 64)
        self.assertEqual(cfg.q_lora_rank, 1024)
        self.assertEqual(cfg.o_lora_rank, 1024)
        self.assertEqual(cfg.o_groups, 8)
        self.assertEqual(cfg.compress_ratios, [0, 4, 128, 0, 0])
        self.assertEqual(cfg.num_hash_layers, 3)
        self.assertEqual(cfg.expert_quant_method, "fp4")
        self.assertEqual(cfg.layers_prefix, "")
        self.assertEqual(cfg.tensor_name("embed.weight"), "embed.weight")
        self.assertEqual(cfg.layer_tensor_prefix(2), "layers.2")

    def test_deepseek_v4_kv_budget_matches_raw_compressed_and_state_layout(self) -> None:
        cfg = self._compact_deepseek_v4_config()
        cache = PagedKVCache.__new__(PagedKVCache)
        cache.cfg = cfg
        cache.page_size = 16
        cache.dsv4_compress_ratios = [
            cfg.compress_ratios[layer_idx] for layer_idx in range(cfg.num_hidden_layers)
        ]

        # Four raw windows plus one ratio-4 main/index cache and state, plus
        # one ratio-128 main cache and state. The arithmetic is deliberately
        # independent of PagedKVCache._dsv4_bytes_for_pages.
        raw = 4 * 128 * 8 * 2
        ratio4_main_cache = 32 * 8 * 2
        ratio4_main_state = (8 * 16 * 4) * 2
        ratio4_index_cache = 32 * 4 * 2
        ratio4_index_state = (8 * 8 * 4) * 2
        ratio128_main_cache = 1 * 8 * 2
        ratio128_main_state = (128 * 8 * 4) * 2
        expected = (
            raw
            + ratio4_main_cache
            + ratio4_main_state
            + ratio4_index_cache
            + ratio4_index_state
            + ratio128_main_cache
            + ratio128_main_state
        )
        self.assertEqual(cache._dsv4_bytes_for_pages(8), expected)

    def test_deepseek_v4_rejects_incomplete_or_invalid_metadata(self) -> None:
        mutations = (
            ("compress_ratios", [0, 4], r"compress_ratios length"),
            ("qk_rope_head_dim", 513, r"head_dim must exceed"),
            ("num_hash_layers", 5, r"num_hash_layers 5 outside"),
            ("expert_dtype", "bf16", r"requires expert_dtype='fp4'"),
            ("scoring_func", "sigmoid", r"requires scoring_func='sqrtsoftplus'"),
            ("n_shared_experts", 2, r"exactly one shared expert"),
            ("dspark_target_layer_ids", [4], r"out-of-range layer 4"),
        )
        for key, value, message in mutations:
            with self.subTest(key=key, value=value):
                raw = _deepseek_v4_config()
                raw[key] = value
                with self.assertRaisesRegex(ValueError, message):
                    ModelConfig.from_model_path(_write_config(self, raw))

    def test_deepseek_v4_fp8_dense_dequant_uses_e8m0_blocks(self) -> None:
        cfg = ModelConfig.from_model_path(
            _write_config(self, _deepseek_v4_config())
        )
        weight = torch.tensor(
            [[1.0] * 128 + [2.0], [0.5] * 128 + [-1.0]],
            dtype=torch.float32,
        ).to(torch.float8_e4m3fn)
        scale = torch.tensor(
            [[2.0, 4.0]], dtype=torch.float32
        ).to(torch.float8_e8m0fnu)
        tensors = {
            "layers.0.attn.wq_a.weight": weight,
            "layers.0.attn.wq_a.scale": scale,
        }
        loader = WeightLoader.__new__(WeightLoader)
        loader.cfg = cfg
        loader._weight_map = {name: "synthetic" for name in tensors}
        loader._read_tensor = tensors.__getitem__

        actual = loader._read_and_dequant("layers.0.attn.wq_a.weight")
        expected = weight.float()
        expected[:, :128] *= 2.0
        expected[:, 128:] *= 4.0
        torch.testing.assert_close(actual.float(), expected, rtol=0, atol=0)

        loader._weight_map.pop("layers.0.attn.wq_a.scale")
        with self.assertRaisesRegex(KeyError, r"has no block scale"):
            loader._read_and_dequant("layers.0.attn.wq_a.weight")

    def test_glm5_next_fp8_dequant_uses_f32_block_scale_inv(self) -> None:
        cfg = ModelConfig.from_model_path(
            _write_config(self, _glm5_next_config())
        )
        weight = torch.tensor(
            [[1.0] * 128 + [2.0], [0.5] * 128 + [-1.0]],
            dtype=torch.float32,
        ).to(torch.float8_e4m3fn)
        scale = torch.tensor([[2.0, 4.0]], dtype=torch.float32)
        tensors = {
            "model.language_model.layers.3.mlp.experts.0.gate_proj.weight": weight,
            "model.language_model.layers.3.mlp.experts.0.gate_proj.weight_scale_inv": scale,
        }
        loader = WeightLoader.__new__(WeightLoader)
        loader.cfg = cfg
        loader._weight_map = {name: "synthetic" for name in tensors}
        loader._read_tensor = tensors.__getitem__

        actual = loader._read_and_dequant(
            "model.language_model.layers.3.mlp.experts.0.gate_proj.weight"
        )
        expected = weight.float()
        expected[:, :128] *= 2.0
        expected[:, 128:] *= 4.0
        torch.testing.assert_close(actual.float(), expected, rtol=0, atol=0)

        tensors[
            "model.language_model.layers.3.mlp.experts.0.gate_proj.weight_scale_inv"
        ] = torch.ones((2, 1), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, r"shape .* != expected"):
            loader._read_and_dequant(
                "model.language_model.layers.3.mlp.experts.0.gate_proj.weight"
            )

    def test_deepseek_v4_checkpoint_inventory_separates_mtp(self) -> None:
        cfg = ModelConfig.from_model_path(
            _write_config(self, _deepseek_v4_config())
        )
        loader = WeightLoader.__new__(WeightLoader)
        loader.cfg = cfg
        loader._weight_map = {
            **{f"layers.{idx}.attn.weight": "fixture" for idx in range(4)},
            "embed.weight": "fixture",
            "norm.weight": "fixture",
            "head.weight": "fixture",
            "hc_head_base": "fixture",
            "hc_head_fn": "fixture",
            "hc_head_scale": "fixture",
            "mtp.0.attn.weight": "fixture",
            "mtp.1.attn.weight": "fixture",
            "mtp.2.attn.weight": "fixture",
        }
        loader._validate_deepseek_v4_checkpoint_inventory()

        loader._weight_map["vision.encoder.weight"] = "fixture"
        with self.assertRaisesRegex(ValueError, r"unexpected tensor namespace"):
            loader._validate_deepseek_v4_checkpoint_inventory()
        del loader._weight_map["vision.encoder.weight"]

        del loader._weight_map["mtp.1.attn.weight"]
        with self.assertRaisesRegex(ValueError, r"contiguous from zero"):
            loader._validate_deepseek_v4_checkpoint_inventory()

    def test_deepseek_v4_layer_contract_covers_all_compression_modes(self) -> None:
        cfg = self._compact_deepseek_v4_config()
        for layer_idx in (0, 1, 2):
            with self.subTest(layer_idx=layer_idx):
                attention, hyper_connection = self._deepseek_v4_layer_tensors(
                    cfg, layer_idx
                )
                native = NativeDeepseekV4Weights(
                    cfg, layer_idx, attention, hyper_connection
                )
                self.assertEqual(native.compress_ratio, cfg.compress_ratios[layer_idx])

    def test_deepseek_v4_layer_extraction_preserves_hash_and_nested_weights(self) -> None:
        cfg = self._compact_deepseek_v4_config()
        attention, hyper_connection = self._deepseek_v4_layer_tensors(cfg, 0)
        hash_table = torch.arange(
            cfg.vocab_size * cfg.num_experts_per_tok, dtype=torch.int64
        ).reshape(cfg.vocab_size, cfg.num_experts_per_tok)
        weights = {
            "norms": {
                "input_layernorm": torch.empty(cfg.hidden_size, dtype=torch.bfloat16),
                "post_attention_layernorm": torch.empty(
                    cfg.hidden_size, dtype=torch.bfloat16
                ),
            },
            "attention": attention,
            "hyper_connection": hyper_connection,
            "gate": {
                "weight": torch.empty(
                    cfg.n_routed_experts, cfg.hidden_size, dtype=torch.bfloat16
                ),
                "tid2eid": hash_table,
            },
            "is_moe": True,
            "layer_type": "full_attention",
        }
        layer = TransformerLayer(cfg, 0, weights, torch.device("cpu"))
        extracted = KrasisModel._extract_layer_weights(layer, torch.device("cpu"))
        self.assertIs(extracted["attention"]["wq_a"], attention["wq_a"])
        self.assertIs(
            extracted["hyper_connection"]["hc_attn_fn"],
            hyper_connection["hc_attn_fn"],
        )
        self.assertIs(extracted["gate"]["tid2eid"], hash_table)

    def test_dsa_topk_candidate_capacity_matches_native_hierarchy(self) -> None:
        self.assertEqual(_dsa_topk_candidate_capacity(1537, 2048), 0)
        self.assertEqual(_dsa_topk_candidate_capacity(2049, 2048), 0)
        self.assertEqual(_dsa_topk_candidate_capacity(5003, 2048), 4096)
        self.assertEqual(
            _dsa_topk_candidate_capacity(1_048_576, 2048),
            256 * 2048,
        )
        self.assertEqual(_dsa_topk_candidate_capacity(10_001, 1537), 3 * 1537)
        for context, topk in ((0, 2048), (2048, 0)):
            with self.subTest(context=context, topk=topk):
                with self.assertRaises(ValueError):
                    _dsa_topk_candidate_capacity(context, topk)

    def test_runtime_context_cap_never_extends_model_support(self) -> None:
        cfg = ModelConfig.from_model_path(_write_config(self, _glm_dsa_config()))
        cfg.max_position_embeddings = 4096

        _apply_max_context_limit(cfg, None)
        self.assertEqual(cfg.max_position_embeddings, 4096)

        _apply_max_context_limit(cfg, 2048)
        self.assertEqual(cfg.max_position_embeddings, 2048)

        for invalid in (0, -1):
            with self.subTest(invalid=invalid):
                capped = ModelConfig.from_model_path(
                    _write_config(self, _glm_dsa_config())
                )
                capped.max_position_embeddings = 4096
                with self.assertRaisesRegex(ValueError, r"must be positive"):
                    _apply_max_context_limit(capped, invalid)

        capped = ModelConfig.from_model_path(_write_config(self, _glm_dsa_config()))
        capped.max_position_embeddings = 4096
        with self.assertRaisesRegex(ValueError, r"exceeds model limit 4096"):
            _apply_max_context_limit(capped, 4097)

    def test_glm_moe_dsa_indexshare_contract(self) -> None:
        cfg = ModelConfig.from_model_path(_write_config(self, _glm_dsa_config()))

        self.assertTrue(cfg.is_mla)
        self.assertTrue(cfg.is_dsa)
        self.assertEqual(cfg.index_topk, 2048)
        self.assertEqual(cfg.index_head_dim, 128)
        self.assertEqual(cfg.index_n_heads, 32)
        self.assertEqual(cfg.index_topk_freq, 4)
        self.assertEqual(cfg.index_skip_topk_offset, 3)
        self.assertEqual(
            cfg.indexer_types,
            [
                "full",
                "full",
                "full",
                "shared",
                "shared",
                "shared",
                "full",
                "shared",
            ],
        )
        self.assertTrue(cfg.indexer_rope_interleave)
        self.assertTrue(cfg.index_share_for_mtp_iteration)
        self.assertEqual(
            [cfg.dsa_indexer_owner_layer(i) for i in range(8)],
            [0, 1, 2, 2, 2, 2, 6, 6],
        )
        self.assertEqual(
            [cfg.is_dsa_indexer_owner_layer(i) for i in range(8)],
            [True, True, True, False, False, False, True, False],
        )
        self.assertEqual(_dsa_owner_layers_for_segment(cfg, 3, 6), [2])
        self.assertEqual(_dsa_owner_layers_for_segment(cfg, 5, 8), [2, 6])
        self.assertEqual(
            _dsa_resource_layers_for_segment(cfg, 3, 6),
            ([], [2]),
        )
        self.assertEqual(
            _dsa_resource_layers_for_segment(cfg, 5, 8),
            ([6], [2]),
        )
        self.assertEqual(
            _dsa_resource_layers_for_segment(cfg, 2, 7),
            ([2, 6], []),
        )
        self.assertEqual(cfg.num_moe_layers, 5)

        implicit_interleave = _glm_dsa_config()
        implicit_interleave.pop("indexer_rope_interleave")
        cfg = ModelConfig.from_model_path(
            _write_config(self, implicit_interleave)
        )
        self.assertTrue(cfg.indexer_rope_interleave)

        invalid_interleave = _glm_dsa_config()
        invalid_interleave["indexer_rope_interleave"] = False
        with self.assertRaisesRegex(
            ValueError,
            r"indexer requires interleaved RoPE",
        ):
            ModelConfig.from_model_path(
                _write_config(self, invalid_interleave)
            )

    def test_glm_moe_dsa_requires_complete_indexer_schedule(self) -> None:
        raw = _glm_dsa_config()
        raw["indexer_types"] = raw["indexer_types"][:-1]
        with self.assertRaisesRegex(
            ValueError,
            r"indexer_types length 7 != num_hidden_layers 8",
        ):
            ModelConfig.from_model_path(_write_config(self, raw))

    def test_glm_moe_dsa_rejects_unowned_shared_indexer(self) -> None:
        raw = _glm_dsa_config()
        raw["indexer_types"][0] = "shared"
        with self.assertRaisesRegex(
            ValueError,
            r"shared indexer at layer 0 has no preceding full indexer",
        ):
            ModelConfig.from_model_path(_write_config(self, raw))

    def test_glm_moe_dsa_requires_indexer_projection_dimensions(self) -> None:
        raw = _glm_dsa_config()
        raw.pop("q_lora_rank")
        with self.assertRaisesRegex(
            ValueError,
            r"requires positive q_lora_rank, got 0",
        ):
            ModelConfig.from_model_path(_write_config(self, raw))

        raw = _glm_dsa_config()
        raw["index_head_dim"] = raw["qk_rope_head_dim"] - 2
        with self.assertRaisesRegex(
            ValueError,
            r"index_head_dim 62 is smaller than qk_rope_head_dim 64",
        ):
            ModelConfig.from_model_path(_write_config(self, raw))

    def test_glm5_next_hybrid_architecture_contract(self) -> None:
        cfg = ModelConfig.from_model_path(
            _write_config(self, _glm5_next_config())
        )

        self.assertTrue(cfg.is_glm5_next)
        self.assertTrue(cfg.is_mla)
        self.assertTrue(cfg.is_dsa)
        self.assertTrue(cfg.mla_use_nope)
        self.assertEqual(cfg.qk_rope_head_dim, 0)
        self.assertEqual(cfg.rotary_dim, 0)
        self.assertEqual(cfg.layers_prefix, "model.language_model")
        self.assertEqual(
            cfg.layer_types,
            [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
        )
        self.assertEqual(cfg.linear_kda_layers, [0, 1, 2, 4, 5, 6])
        self.assertEqual(cfg.linear_full_attention_layers, [3, 7])
        self.assertEqual(cfg.linear_num_key_heads, 64)
        self.assertEqual(cfg.linear_num_value_heads, 64)
        self.assertEqual(cfg.linear_key_head_dim, 128)
        self.assertEqual(cfg.linear_value_head_dim, 128)
        self.assertEqual(cfg.linear_conv_kernel_dim, 4)
        self.assertEqual(cfg.linear_gate_lower_bound, -5.0)
        self.assertEqual(cfg.index_topk_freq, 4)
        self.assertEqual(cfg.index_kpool, 4)
        self.assertTrue(cfg.index_kpool_compress)
        self.assertTrue(cfg.index_kpool_always_select_tail)
        self.assertTrue(cfg.mhc)
        self.assertEqual(cfg.hc_mult, 4)
        self.assertEqual(cfg.num_nextn_predict_layers, 1)
        self.assertEqual(cfg.num_full_attention_layers, 2)
        self.assertEqual(
            [cfg.dsa_indexer_owner_layer(idx) for idx in range(8)],
            [None, None, None, 3, None, None, None, 7],
        )
        self.assertEqual(
            [cfg.is_dsa_indexer_owner_layer(idx) for idx in range(8)],
            [False, False, False, True, False, False, False, True],
        )
        self.assertEqual(
            [cfg.is_moe_layer(idx) for idx in range(8)],
            [False, False, False, True, True, True, True, True],
        )
        self.assertTrue(cfg.need_fp32_gate)
        self.assertEqual(cfg.expert_quant_method, "fp8")
        self.assertEqual(cfg.max_position_embeddings, 1_048_576)

    def test_glm5_next_official_45_layer_schedule(self) -> None:
        cfg = ModelConfig.from_model_path(
            _write_config(self, _glm5_next_config(num_layers=45))
        )
        self.assertEqual(cfg.num_hidden_layers, 45)
        self.assertEqual(cfg.num_full_attention_layers, 11)
        self.assertEqual(
            cfg.linear_full_attention_layers,
            list(range(3, 45, 4)),
        )
        self.assertEqual(cfg.linear_kda_layers[-1], 44)
        self.assertEqual(cfg.dsa_indexer_owner_layer(43), 43)
        self.assertIsNone(cfg.dsa_indexer_owner_layer(44))
        self.assertEqual(
            _dsa_owner_layers_for_segment(cfg, 0, 45),
            list(range(3, 45, 4)),
        )
        self.assertEqual(_dsa_owner_layers_for_segment(cfg, 0, 3), [])
        self.assertEqual(
            _dsa_resource_layers_for_segment(cfg, 0, 45),
            (list(range(3, 45, 4)), []),
        )

    def test_glm5_next_rejects_divergent_architecture_metadata(self) -> None:
        raw = _glm5_next_config()
        raw["text_config"]["linear_attn_config"]["kda_layers"].remove(6)
        with self.assertRaisesRegex(ValueError, r"kda_layers does not match"):
            ModelConfig.from_model_path(_write_config(self, raw))

        raw = _glm5_next_config()
        raw["text_config"]["qk_rope_head_dim"] = 64
        with self.assertRaisesRegex(ValueError, r"requires rope-free MLA"):
            ModelConfig.from_model_path(_write_config(self, raw))

        raw = _glm5_next_config()
        raw["text_config"]["index_kpool_always_select_tail"] = False
        with self.assertRaisesRegex(ValueError, r"always_select_tail=true"):
            ModelConfig.from_model_path(_write_config(self, raw))

        raw = _glm5_next_config()
        raw["text_config"]["mhc"] = False
        with self.assertRaisesRegex(ValueError, r"requires mHC"):
            ModelConfig.from_model_path(_write_config(self, raw))

    def test_glm5_next_kda_loader_preserves_native_tensor_contract(self) -> None:
        cfg = ModelConfig.from_model_path(
            _write_config(self, _tiny_glm5_next_config())
        )
        prefix = f"{cfg.layer_tensor_prefix(0)}.self_attn"
        shape_by_suffix = {
            "q_proj.weight": (8, 8),
            "k_proj.weight": (8, 8),
            "v_proj.weight": (8, 8),
            "q_conv1d.weight": (8, 1, 4),
            "k_conv1d.weight": (8, 1, 4),
            "v_conv1d.weight": (8, 1, 4),
            "f_a_proj.weight": (4, 8),
            "f_b_proj.weight": (8, 4),
            "b_proj.weight": (2, 8),
            "g_a_proj.weight": (4, 8),
            "g_b_proj.weight": (8, 4),
            "o_norm.weight": (4,),
            "o_proj.weight": (8, 8),
            "A_log": (2,),
            "dt_bias": (8,),
        }
        tensors = {}
        torch.manual_seed(7)
        for suffix, shape in shape_by_suffix.items():
            dtype = torch.float32 if suffix in ("A_log", "dt_bias") else torch.bfloat16
            tensors[f"{prefix}.{suffix}"] = (
                torch.randn(shape, dtype=torch.float32) * 0.05
            ).to(dtype)

        loader = WeightLoader.__new__(WeightLoader)
        loader.cfg = cfg
        loader._weight_map = {name: "fixture" for name in tensors}
        loader._load_bf16 = lambda name, _device: tensors[name].to(torch.bfloat16)
        loader._load_f32 = lambda name, _device: tensors[name].float()

        loaded = loader.load_linear_attention_weights(0, torch.device("cpu"))
        self.assertEqual(
            set(loaded),
            {
                "q_proj",
                "k_proj",
                "v_proj",
                "q_conv1d",
                "k_conv1d",
                "v_conv1d",
                "f_a_proj",
                "f_b_proj",
                "b_proj",
                "g_a_proj",
                "g_b_proj",
                "o_norm",
                "o_proj",
                "A_log",
                "dt_bias",
            },
        )
        self.assertEqual(loaded["A_log"].dtype, torch.float32)
        self.assertEqual(loaded["dt_bias"].dtype, torch.float32)
        self.assertEqual(loaded["q_proj"].dtype, torch.bfloat16)

        full = KimiDeltaAttention(cfg, 0, loaded, torch.device("cpu"))
        split = KimiDeltaAttention(cfg, 0, loaded, torch.device("cpu"))
        hidden = (torch.randn(3, cfg.hidden_size) * 0.1).to(torch.bfloat16)
        full_output = full.forward(hidden, is_decode=False)
        split_output = torch.cat(
            [
                split.forward(hidden[:2], is_decode=False),
                split.forward(hidden[2:], is_decode=True),
            ],
            dim=0,
        )
        torch.testing.assert_close(
            split_output.float(),
            full_output.float(),
            rtol=2e-3,
            atol=2e-3,
        )

    def test_glm5_next_kpool_and_mhc_tensor_contracts(self) -> None:
        cfg = ModelConfig.from_model_path(
            _write_config(self, _tiny_glm5_next_config())
        )
        layer_idx = 3
        prefix = f"{cfg.layer_tensor_prefix(layer_idx)}.self_attn.indexer"
        index_tensors = {
            f"{prefix}.wq_b.weight": torch.zeros((8, 4), dtype=torch.bfloat16),
            f"{prefix}.wk.weight": torch.zeros((4, 8), dtype=torch.bfloat16),
            f"{prefix}.weights_proj.weight": torch.zeros((2, 8), dtype=torch.bfloat16),
            f"{prefix}.k_norm.weight": torch.ones((4,), dtype=torch.bfloat16),
            f"{prefix}.k_norm.bias": torch.zeros((4,), dtype=torch.bfloat16),
            f"{prefix}.index_kpool_compress_ape": torch.zeros((2, 4), dtype=torch.bfloat16),
            f"{prefix}.index_kpool_compress_gate": torch.zeros((4, 8), dtype=torch.bfloat16),
        }
        loader = WeightLoader.__new__(WeightLoader)
        loader.cfg = cfg
        loader._weight_map = {name: "fixture" for name in index_tensors}
        loader._load_bf16 = lambda name, _device: index_tensors[name]
        loaded_indexer = loader._load_dsa_indexer_weights(
            layer_idx, torch.device("cpu")
        )
        self.assertEqual(len(loaded_indexer), 7)
        native_indexer = NativeDsaIndexerWeights(
            cfg, layer_idx, loaded_indexer
        )
        self.assertEqual(
            tuple(native_indexer.index_kpool_compress_ape.shape), (2, 4)
        )

        mix_width = (2 + cfg.hc_mult) * cfg.hc_mult
        hc_input = cfg.hc_mult * cfg.hidden_size
        hc_tensors = {
            "hc_attn_fn": torch.zeros((mix_width, hc_input)),
            "hc_attn_base": torch.zeros((mix_width,)),
            "hc_attn_scale": torch.ones((3,)),
            "hc_ffn_fn": torch.zeros((mix_width, hc_input)),
            "hc_ffn_base": torch.zeros((mix_width,)),
            "hc_ffn_scale": torch.ones((3,)),
        }
        native_hc = NativeHyperConnectionWeights(cfg, layer_idx, hc_tensors)
        self.assertIs(native_hc.tensors, hc_tensors)

    def test_glm5_next_promotes_mixed_checkpoint_control_weights(self) -> None:
        cfg = ModelConfig.from_model_path(
            _write_config(self, _tiny_glm5_next_config())
        )
        layer_idx = 3
        prefix = cfg.layer_tensor_prefix(layer_idx)
        tensors = {
            f"{prefix}.hc_attn_fn": torch.zeros((24, 32), dtype=torch.bfloat16),
            f"{prefix}.hc_attn_base": torch.zeros((24,), dtype=torch.float32),
            f"{prefix}.hc_attn_scale": torch.ones((3,), dtype=torch.float32),
            f"{prefix}.hc_ffn_fn": torch.zeros((24, 32), dtype=torch.bfloat16),
            f"{prefix}.hc_ffn_base": torch.zeros((24,), dtype=torch.float32),
            f"{prefix}.hc_ffn_scale": torch.ones((3,), dtype=torch.float32),
            f"{prefix}.mlp.gate.weight": torch.zeros((4, 8), dtype=torch.bfloat16),
            f"{prefix}.mlp.gate.e_score_correction_bias": torch.zeros(
                (4,), dtype=torch.float32
            ),
        }
        loader = WeightLoader.__new__(WeightLoader)
        loader.cfg = cfg
        loader._weight_map = {name: "fixture" for name in tensors}
        loader._read_tensor = tensors.__getitem__

        hyper_connection = loader.load_hyper_connection(
            layer_idx, torch.device("cpu")
        )
        self.assertTrue(
            all(tensor.dtype == torch.float32 for tensor in hyper_connection.values())
        )

        router = loader.load_moe_gate(layer_idx, torch.device("cpu"))
        self.assertEqual(router["weight"].dtype, torch.float32)
        self.assertEqual(
            router["e_score_correction_bias"].dtype, torch.float32
        )

    def test_fp32_router_uses_bf16_rust_wire_contract(self) -> None:
        source_bf16 = torch.tensor(
            [[1.0, -2.5, 0.125], [3.25, -0.5, 7.0]],
            dtype=torch.bfloat16,
        )
        promoted = source_bf16.float()

        payload = KrasisModel._routing_gate_bf16_bytes(promoted)

        self.assertEqual(len(payload), source_bf16.numel() * 2)
        self.assertEqual(
            payload,
            source_bf16.contiguous().view(torch.uint16).numpy().view("u1").tobytes(),
        )

    def test_routing_gate_wire_contract_rejects_non_matrix(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "rank 2"):
            KrasisModel._routing_gate_bf16_bytes(torch.zeros(8))

    def test_non_dsa_defaults_remain_disabled(self) -> None:
        raw = _glm_dsa_config()
        raw["model_type"] = "deepseek_v3"
        for key in (
            "index_topk",
            "index_head_dim",
            "index_n_heads",
            "index_topk_freq",
            "index_skip_topk_offset",
            "indexer_types",
            "indexer_rope_interleave",
            "index_share_for_mtp_iteration",
        ):
            raw.pop(key)
        cfg = ModelConfig.from_model_path(_write_config(self, raw))
        self.assertFalse(cfg.is_dsa)
        self.assertEqual(cfg.index_topk, 0)
        self.assertIsNone(cfg.indexer_types)
        self.assertIsNone(cfg.dsa_indexer_owner_layer(0))

    def test_dsa_indexer_loader_uses_owner_only_checkpoint_names(self) -> None:
        raw = _glm_dsa_config()
        cfg = ModelConfig.from_model_path(_write_config(self, raw))
        prefix = f"{cfg.layers_prefix}.layers.0.self_attn.indexer"
        tensors = {
            f"{prefix}.wq_b.weight": torch.zeros(
                (cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank),
                dtype=torch.bfloat16,
            ),
            f"{prefix}.wk.weight": torch.zeros(
                (cfg.index_head_dim, cfg.hidden_size),
                dtype=torch.bfloat16,
            ),
            f"{prefix}.weights_proj.weight": torch.zeros(
                (cfg.index_n_heads, cfg.hidden_size),
                dtype=torch.bfloat16,
            ),
            f"{prefix}.k_norm.weight": torch.ones(
                (cfg.index_head_dim,),
                dtype=torch.bfloat16,
            ),
            f"{prefix}.k_norm.bias": torch.zeros(
                (cfg.index_head_dim,),
                dtype=torch.bfloat16,
            ),
        }

        loader = WeightLoader.__new__(WeightLoader)
        loader.cfg = cfg
        loader._weight_map = {name: "fixture" for name in tensors}
        loader._load_bf16 = lambda name, _device: tensors[name]

        loaded = loader._load_dsa_indexer_weights(0, torch.device("cpu"))
        self.assertEqual(
            set(loaded),
            {"wq_b", "wk", "weights_proj", "k_norm_weight", "k_norm_bias"},
        )
        self.assertIs(loaded["wq_b"], tensors[f"{prefix}.wq_b.weight"])

        with self.assertRaisesRegex(
            ValueError,
            r"layer 3 shares indexer owner 2",
        ):
            loader._load_dsa_indexer_weights(3, torch.device("cpu"))

        del loader._weight_map[f"{prefix}.k_norm.bias"]
        with self.assertRaisesRegex(
            KeyError,
            r"indexer\.k_norm\.bias",
        ):
            loader._load_dsa_indexer_weights(0, torch.device("cpu"))

    def test_native_mla_setup_contract_pads_only_the_kernel_dimension(self) -> None:
        raw = _glm_dsa_config()
        raw.update(
            {
                "hidden_size": 8,
                "intermediate_size": 16,
                "moe_intermediate_size": 4,
                "num_attention_heads": 2,
                "q_lora_rank": 4,
                "kv_lora_rank": 4,
                "qk_nope_head_dim": 2,
                "qk_rope_head_dim": 2,
                "v_head_dim": 2,
                "index_topk": 4,
                "index_head_dim": 4,
                "index_n_heads": 2,
            }
        )
        cfg = ModelConfig.from_model_path(_write_config(self, raw))
        weights = {
            "q_a_proj": torch.zeros((4, 8), dtype=torch.bfloat16),
            "q_b_proj": torch.zeros((8, 4), dtype=torch.bfloat16),
            "q_a_layernorm": torch.ones((4,), dtype=torch.bfloat16),
            "kv_a_proj_with_mqa": torch.zeros((6, 8), dtype=torch.bfloat16),
            "o_proj": torch.zeros((8, 4), dtype=torch.bfloat16),
            "kv_a_layernorm": torch.ones((4,), dtype=torch.bfloat16),
            "w_kc": torch.ones((2, 2, 4), dtype=torch.bfloat16),
            "w_vc": torch.ones((2, 2, 4), dtype=torch.bfloat16),
            "dsa_indexer": {
                "wq_b": torch.zeros((8, 4), dtype=torch.bfloat16),
                "wk": torch.zeros((4, 8), dtype=torch.bfloat16),
                "weights_proj": torch.zeros((2, 8), dtype=torch.bfloat16),
                "k_norm_weight": torch.ones((4,), dtype=torch.bfloat16),
                "k_norm_bias": torch.zeros((4,), dtype=torch.bfloat16),
            },
        }

        layer = TransformerLayer(
            cfg,
            0,
            {
                "norms": {
                    "input_layernorm": torch.ones((8,), dtype=torch.bfloat16),
                    "post_attention_layernorm": torch.ones(
                        (8,), dtype=torch.bfloat16
                    ),
                },
                "is_moe": False,
                "layer_type": "full_attention",
                "attention": weights,
            },
            torch.device("cpu"),
        )
        attention = layer.attention

        self.assertIsInstance(attention, NativeMLAWeights)
        self.assertIsInstance(attention.dsa_indexer, NativeDsaIndexerWeights)
        self.assertEqual(attention.dsa_indexer_owner_layer, 0)
        self.assertEqual(attention.dsa_indexer.wq_b.shape, (8, 4))
        self.assertEqual(attention.ckv_dim, MLA_CKV_KERNEL_MIN_DIM)
        self.assertEqual(attention.w_kc.shape, (2, 2, MLA_CKV_KERNEL_MIN_DIM))
        self.assertEqual(attention.w_vc.shape, (2, 2, MLA_CKV_KERNEL_MIN_DIM))
        self.assertTrue(torch.all(attention.w_kc[..., :4] == 1))
        self.assertTrue(torch.all(attention.w_kc[..., 4:] == 0))
        with self.assertRaisesRegex(
            RuntimeError,
            r"native Rust/CUDA runtime",
        ):
            attention.forward(torch.zeros((1, 8), dtype=torch.bfloat16))
        with self.assertRaisesRegex(
            RuntimeError,
            r"native Rust/CUDA runtime",
        ):
            attention.dsa_indexer.forward(
                torch.zeros((1, 8), dtype=torch.bfloat16)
            )

        shared_weights = dict(weights)
        shared_weights.pop("dsa_indexer")
        shared_attention = NativeMLAWeights(
            cfg,
            3,
            shared_weights,
            torch.device("cpu"),
        )
        self.assertEqual(shared_attention.dsa_indexer_owner_layer, 2)
        self.assertIsNone(shared_attention.dsa_indexer)

        invalid_weights = dict(weights)
        invalid_weights["dsa_indexer"] = dict(weights["dsa_indexer"])
        invalid_weights["dsa_indexer"]["wk"] = torch.zeros(
            (3, 8),
            dtype=torch.bfloat16,
        )
        with self.assertRaisesRegex(
            ValueError,
            r"tensor wk shape \(3, 8\) != expected \(4, 8\)",
        ):
            NativeMLAWeights(
                cfg,
                0,
                invalid_weights,
                torch.device("cpu"),
            )

    def test_mla_k4_budget_uses_padded_physical_cache_width(self) -> None:
        raw = _glm_dsa_config()
        raw["kv_lora_rank"] = 256
        expected = (
            (MLA_CKV_KERNEL_MIN_DIM // 16) * 10
            + (raw["qk_rope_head_dim"] // 16) * 10
        )
        self.assertEqual(
            _kv_bytes_per_token_per_layer(raw, "k4v4"),
            expected,
        )


if __name__ == "__main__":
    if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
        raise SystemExit("Run through ./dev model-config-test")
    unittest.main()
