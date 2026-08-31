#!/usr/bin/env python3
"""Compare Krasis GLM-5.3 KDA against the official Transformers oracle.

Run with a Transformers checkout new enough to contain ``glm5_next``::

    KRASIS_DEV_SCRIPT=1 PYTHONPATH=python:/path/to/transformers/src \
        python tests/glm5_next_reference.py
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
from transformers.models.glm5_next.configuration_glm5_next import (
    Glm5NextTextConfig,
)
from transformers.models.glm5_next.modeling_glm5_next import (
    Glm5NextTextHyperConnection,
    Glm5NextTextLinearAttention,
)

from krasis.config import ModelConfig
from krasis.layer import NativeHyperConnectionWeights
from krasis.linear_attention import KimiDeltaAttention


def _text_config() -> dict:
    return {
        "model_type": "glm5_next_text",
        "hidden_size": 8,
        "intermediate_size": 16,
        "moe_intermediate_size": 4,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "vocab_size": 32,
        "pad_token_id": 0,
        "q_lora_rank": 4,
        "kv_lora_rank": 4,
        "qk_nope_head_dim": 4,
        "qk_rope_head_dim": 0,
        "v_head_dim": 4,
        "mla_use_nope": True,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
        "first_k_dense_replace": 3,
        "routed_scaling_factor": 2.5,
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        "norm_topk_prob": True,
        "moe_router_dtype": "float32",
        "swiglu_limit": 10.0,
        "index_topk": 4,
        "index_head_dim": 4,
        "index_n_heads": 2,
        "index_kpool": 2,
        "index_kpool_compress": True,
        "index_kpool_always_select_tail": True,
        "indexer_types": ["full"] * 4,
        "indexer_rope_interleave": True,
        "index_share_for_mtp_iteration": True,
        "layer_types": [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "deepseek_sparse_attention",
        ],
        "linear_attn_config": {
            "num_heads": 2,
            "head_dim": 4,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
            "kda_layers": [0, 1, 2],
            "full_attn_layers": [3],
        },
        "mlp_layer_types": ["dense", "dense", "dense", "sparse"],
        "mhc": True,
        "hc_mult": 2,
        "hc_sinkhorn_iters": 3,
        "hc_eps": 1e-6,
        "num_nextn_predict_layers": 1,
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 128,
        "tie_word_embeddings": False,
    }


def _krasis_config(text_config: dict) -> ModelConfig:
    root = tempfile.mkdtemp(prefix="krasis-glm5-next-reference-")
    try:
        raw = {
            "model_type": "glm5_next",
            "architectures": ["Glm5NextForConditionalGeneration"],
            "quantization_config": {
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
            },
            "text_config": text_config,
        }
        Path(root, "config.json").write_text(json.dumps(raw), encoding="utf-8")
        return ModelConfig.from_model_path(root)
    finally:
        shutil.rmtree(root)


def _randomize_checkpoint_dtypes(module: torch.nn.Module) -> None:
    generator = torch.Generator().manual_seed(53)
    for name, parameter in module.named_parameters():
        value = torch.randn(
            parameter.shape,
            generator=generator,
            dtype=torch.float32,
        ) * 0.05
        if name.endswith("A_log") or name.endswith("dt_bias"):
            parameter.data = value
        else:
            parameter.data = value.to(torch.bfloat16)


def main() -> None:
    text_config = _text_config()
    official_config = Glm5NextTextConfig(**text_config)
    official = Glm5NextTextLinearAttention(official_config, layer_idx=0).eval()
    _randomize_checkpoint_dtypes(official)

    conv = official.conv1d.weight.detach()
    width = official.qkv_dim
    weights = {
        "q_proj": official.q_proj.weight.detach(),
        "k_proj": official.k_proj.weight.detach(),
        "v_proj": official.v_proj.weight.detach(),
        "q_conv1d": conv[:width].contiguous(),
        "k_conv1d": conv[width : 2 * width].contiguous(),
        "v_conv1d": conv[2 * width :].contiguous(),
        "f_a_proj": official.forget_gate.f_a_proj.weight.detach(),
        "f_b_proj": official.forget_gate.f_b_proj.weight.detach(),
        "b_proj": official.b_proj.weight.detach(),
        "g_a_proj": official.g_a_proj.weight.detach(),
        "g_b_proj": official.g_b_proj.weight.detach(),
        "o_norm": official.o_norm.weight.detach(),
        "o_proj": official.o_proj.weight.detach(),
        "A_log": official.forget_gate.A_log.detach().float(),
        "dt_bias": official.forget_gate.dt_bias.detach().float(),
    }
    krasis_config = _krasis_config(text_config)
    candidate = KimiDeltaAttention(
        krasis_config,
        layer_idx=0,
        weights=weights,
        device=torch.device("cpu"),
    )

    hidden = (
        torch.randn(1, 7, text_config["hidden_size"], generator=torch.Generator().manual_seed(7))
        * 0.1
    ).to(torch.bfloat16)
    attention_mask = torch.ones(1, hidden.shape[1], dtype=torch.bool)
    with torch.no_grad():
        expected = official(
            hidden,
            cache_params=None,
            attention_mask=attention_mask,
        )
        actual = candidate.forward(hidden.squeeze(0), is_decode=False).unsqueeze(0)

    max_abs = (expected.float() - actual.float()).abs().max().item()
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        rtol=8e-3,
        atol=8e-3,
    )

    official_hc = Glm5NextTextHyperConnection(official_config).eval()
    hc_generator = torch.Generator().manual_seed(530)
    with torch.no_grad():
        for parameter in official_hc.parameters():
            parameter.copy_(
                torch.randn(
                    parameter.shape,
                    generator=hc_generator,
                    dtype=torch.float32,
                )
                * 0.05
            )
    hc_tensors = {
        "hc_attn_fn": official_hc.fn.detach(),
        "hc_attn_base": official_hc.base.detach(),
        "hc_attn_scale": official_hc.scale.detach(),
        "hc_ffn_fn": official_hc.fn.detach().clone(),
        "hc_ffn_base": official_hc.base.detach().clone(),
        "hc_ffn_scale": official_hc.scale.detach().clone(),
    }
    candidate_hc = NativeHyperConnectionWeights(
        krasis_config,
        layer_idx=0,
        tensors=hc_tensors,
    )
    streams = (
        torch.randn(
            1,
            5,
            text_config["hc_mult"],
            text_config["hidden_size"],
            generator=torch.Generator().manual_seed(531),
        )
        * 0.1
    ).to(torch.bfloat16)
    sublayer = (
        torch.randn(
            1,
            5,
            text_config["hidden_size"],
            generator=torch.Generator().manual_seed(532),
        )
        * 0.1
    ).to(torch.bfloat16)
    with torch.no_grad():
        expected_post, expected_comb, expected_collapsed = official_hc(streams)
        actual_post, actual_comb, actual_collapsed = candidate_hc.prepare_attn(
            streams.squeeze(0)
        )
        expected_applied = (
            expected_post.to(streams.dtype).unsqueeze(-1)
            * sublayer.unsqueeze(-2)
            + torch.matmul(
                expected_comb.to(streams.dtype).transpose(-1, -2),
                streams,
            )
        )
        actual_applied = candidate_hc.apply(
            streams.squeeze(0),
            sublayer.squeeze(0),
            actual_post,
            actual_comb,
        )
    torch.testing.assert_close(
        actual_post,
        expected_post.squeeze(0),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        actual_comb,
        expected_comb.squeeze(0),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        actual_collapsed.float(),
        expected_collapsed.squeeze(0).float(),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        actual_applied.float(),
        expected_applied.squeeze(0).float(),
        rtol=1e-6,
        atol=1e-6,
    )
    hc_max_abs = (
        actual_collapsed.float() - expected_collapsed.squeeze(0).float()
    ).abs().max().item()
    print(
        "GLM-5.3 official-reference comparisons passed; "
        f"KDA max_abs={max_abs:.6g}; mHC max_abs={hc_max_abs:.6g}"
    )


if __name__ == "__main__":
    if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
        raise SystemExit("Run with KRASIS_DEV_SCRIPT=1")
    main()
