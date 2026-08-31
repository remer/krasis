"""DeepSeek-V4-Flash-Vision-Exp image tower and prompt expansion.

The architecture and image-layout rules are adapted from DeepSeek's MIT-
licensed reference implementation for ``DeepSeek-V4-Flash-Vision-Exp``.  This
module intentionally contains only the image path needed by Krasis; language
model execution remains in the native Rust/CUDA runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps


IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)
IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"
COMPRESS_PAD_TO = 4


@dataclass(frozen=True)
class DeepseekV4VisionConfig:
    n_layers: int
    dim: int
    n_heads: int
    inter_dim: int
    patch_size: int
    rope_theta: float
    downsample_ratio: int
    max_n_token: int
    min_pixels: int
    max_wh_ratio: float | None
    text_dim: int
    vocab_size: int

    @classmethod
    def from_model_config(cls, config) -> "DeepseekV4VisionConfig":
        if not getattr(config, "is_deepseek_v4_vision", False):
            raise ValueError("model config does not declare DeepSeek-V4 vision")
        result = cls(
            n_layers=int(config.vision_n_layers),
            dim=int(config.vision_dim),
            n_heads=int(config.vision_n_heads),
            inter_dim=int(config.vision_inter_dim),
            patch_size=int(config.vision_patch_size),
            rope_theta=float(config.vision_rope_theta),
            downsample_ratio=int(config.vision_downsample_ratio),
            max_n_token=int(config.vision_max_n_token),
            min_pixels=int(config.vision_min_pixels),
            max_wh_ratio=(
                None
                if config.vision_max_wh_ratio is None
                else float(config.vision_max_wh_ratio)
            ),
            text_dim=int(config.hidden_size),
            vocab_size=int(config.vocab_size),
        )
        result.validate()
        return result

    def validate(self) -> None:
        positive = {
            "n_layers": self.n_layers,
            "dim": self.dim,
            "n_heads": self.n_heads,
            "inter_dim": self.inter_dim,
            "patch_size": self.patch_size,
            "rope_theta": self.rope_theta,
            "downsample_ratio": self.downsample_ratio,
            "max_n_token": self.max_n_token,
            "min_pixels": self.min_pixels,
            "text_dim": self.text_dim,
            "vocab_size": self.vocab_size,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(
                    f"DeepSeek-V4 vision {name} must be positive, got {value}"
                )
        if self.dim % self.n_heads:
            raise ValueError("DeepSeek-V4 vision dim must be divisible by n_heads")
        if (self.dim // self.n_heads) % 4:
            raise ValueError(
                "DeepSeek-V4 vision head dimension must be divisible by four"
            )
        if self.max_wh_ratio is not None and self.max_wh_ratio <= 0:
            raise ValueError("DeepSeek-V4 vision max_wh_ratio must be positive")


@dataclass
class DeepseekV4PreparedImage:
    patches: torch.Tensor
    n_vit_h: int
    n_vit_w: int
    n_llm_h: int
    n_llm_w: int


@dataclass
class DeepseekV4ImageBlock:
    start: int
    types: torch.Tensor
    perm: torch.Tensor
    attention_block_ids: torch.Tensor
    image: DeepseekV4PreparedImage


def grid_tokens(
    height: int,
    width: int,
    patch_size: int,
    downsample_ratio: int,
) -> tuple[int, int, int]:
    """Return the aligned LLM grid and N-layout token count."""
    n_llm_h = math.ceil((height // patch_size) / downsample_ratio)
    n_llm_w = math.ceil((width // patch_size) / downsample_ratio)
    num_tokens = n_llm_h * (n_llm_w + 1) + 2
    if n_llm_h % 2 == 1:
        num_tokens += n_llm_w + 1
    num_tokens += (n_llm_h + 1) // 2 * (n_llm_w + 1) % 2 * 2
    return n_llm_h, n_llm_w, num_tokens


def solve_resize_ratio(
    height: int,
    width: int,
    patch_size: int,
    downsample_ratio: int,
    max_n_token: int,
) -> tuple[int, int, int, int, int]:
    ratio = height / width
    max_w_float = math.sqrt((max_n_token - 2) / ratio + 0.25) - 0.5
    max_h_float = max_w_float * ratio
    if max_w_float < 1.0:
        max_w = 1
        max_h = (max_n_token - 2) // (max_w + 1)
        if max_h % 2 == 1:
            max_h -= 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    elif max_h_float < 2.0:
        max_h = 2
        max_w = ((max_n_token - 2) // max_h) - 1
        if max_w <= 1:
            raise ValueError("DeepSeek-V4 image token budget is too small")
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    else:
        max_w = math.floor(max_w_float)
        max_h = math.floor(max_h_float)
        if max_h % 2 == 1:
            max_h -= 1
        beta = min(
            max_w * patch_size * downsample_ratio / width,
            max_h * patch_size * downsample_ratio / height,
        )
        best_width = math.floor(width * beta / patch_size) * patch_size
        best_height = math.floor(height * beta / patch_size) * patch_size
    n_llm_h, n_llm_w, num_tokens = grid_tokens(
        best_height, best_width, patch_size, downsample_ratio
    )
    return n_llm_h, n_llm_w, best_height, best_width, num_tokens


def safe_resize(
    height: int,
    width: int,
    best_height: int,
    best_width: int,
    patch_size: int,
    downsample_ratio: int,
    max_n_token: int,
) -> tuple[int, int, int, int]:
    max_n_token -= COMPRESS_PAD_TO - 1
    if max_n_token <= 2:
        raise ValueError("DeepSeek-V4 image token budget is too small")
    n_llm_h, n_llm_w, num_tokens = grid_tokens(
        best_height, best_width, patch_size, downsample_ratio
    )
    budget = max_n_token
    while num_tokens > max_n_token:
        n_llm_h, n_llm_w, best_height, best_width, num_tokens = (
            solve_resize_ratio(
                height,
                width,
                patch_size,
                downsample_ratio,
                budget,
            )
        )
        budget -= 1
        if budget <= 2:
            raise ValueError("could not fit DeepSeek-V4 image into token budget")
    return n_llm_h, n_llm_w, best_height, best_width


class DeepseekV4ImagePreprocessor:
    def __init__(self, config: DeepseekV4VisionConfig):
        config.validate()
        self.config = config

    def __call__(self, image: Image.Image) -> DeepseekV4PreparedImage:
        cfg = self.config
        patch = cfg.patch_size
        image = image.convert("RGB")
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("DeepSeek-V4 image dimensions must be positive")

        resized_width, resized_height = width, height
        if (
            cfg.max_wh_ratio is not None
            and resized_width > resized_height * cfg.max_wh_ratio
        ):
            resized_width = int(resized_height * cfg.max_wh_ratio)
        if resized_width * resized_height < cfg.min_pixels:
            scale = math.sqrt(cfg.min_pixels / (resized_width * resized_height))
            resized_width = int(resized_width * scale)
            resized_height = int(resized_height * scale)

        best_width = math.ceil(resized_width / patch) * patch
        best_height = math.ceil(resized_height / patch) * patch
        n_llm_h, n_llm_w, best_height, best_width = safe_resize(
            resized_height,
            resized_width,
            best_height,
            best_width,
            patch,
            cfg.downsample_ratio,
            cfg.max_n_token,
        )
        n_vit_h = best_height // patch
        n_vit_w = best_width // patch

        if (
            cfg.max_wh_ratio is not None
            and image.width >= cfg.max_wh_ratio * image.height
        ):
            image = image.resize((best_width, best_height))
        else:
            image = ImageOps.pad(
                image,
                (best_width, best_height),
                color=(127, 127, 127),
            )
        pixels = torch.from_numpy(
            np.asarray(image, dtype=np.float32).copy()
        ).permute(2, 0, 1)
        pixels = ((pixels / 255.0 - 0.5) / 0.5).to(torch.bfloat16)
        patches = (
            pixels.reshape(3, n_vit_h, patch, n_vit_w, patch)
            .permute(1, 3, 0, 2, 4)
            .reshape(n_vit_h * n_vit_w, 3, patch, patch)
            .contiguous()
        )
        return DeepseekV4PreparedImage(
            patches=patches,
            n_vit_h=n_vit_h,
            n_vit_w=n_vit_w,
            n_llm_h=n_llm_h,
            n_llm_w=n_llm_w,
        )


def build_image_block(
    n_llm_h: int,
    n_llm_w: int,
    start_pos: int,
    block_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build official N-layout types, feature permutation, and attention span."""
    if n_llm_h <= 0 or n_llm_w <= 0 or start_pos < 0 or block_id < 0:
        raise ValueError("DeepSeek-V4 image block geometry is invalid")
    compress_pad = COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO
    pad_h = n_llm_h % 2
    rows = n_llm_h + pad_h
    row_len = n_llm_w + 1
    pad_last = rows // 2 * row_len % 2 * 2

    types = torch.tensor(
        ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
        + [IMAGE_PAD] * (row_len * pad_h),
        dtype=torch.int64,
    )
    order = (
        torch.arange(rows * row_len)
        .view(rows // 2, 2, row_len)
        .transpose(1, 2)
        .reshape(-1)
    )
    image_idx = torch.full((rows * row_len,), -1, dtype=torch.int64)
    image_idx.view(rows, row_len)[:n_llm_h, :n_llm_w] = torch.arange(
        n_llm_h * n_llm_w
    ).view(n_llm_h, n_llm_w)
    perm = image_idx[order]
    perm = perm[perm >= 0]

    types = torch.cat(
        [
            torch.full((compress_pad,), IMAGE_PAD, dtype=torch.int64),
            torch.tensor([IMAGE_START], dtype=torch.int64),
            types[order],
            torch.full((pad_last,), IMAGE_PAD, dtype=torch.int64),
            torch.tensor([IMAGE_END], dtype=torch.int64),
        ]
    )
    attention_ids = torch.full_like(types, -1, dtype=torch.int32)
    # DeepSeek's visibility span begins at IMAGE_START.  Alignment padding
    # before the start sentinel remains ordinary causal history.
    attention_ids[compress_pad:] = int(block_id)
    return types, perm, attention_ids


def expand_image_placeholders(
    token_ids: list[int],
    placeholder_token_id: int,
    images: list[DeepseekV4PreparedImage],
    vocab_size: int,
) -> tuple[list[int], list[int], list[DeepseekV4ImageBlock]]:
    """Replace placeholder IDs with official sentinel blocks."""
    if placeholder_token_id < 0 or placeholder_token_id >= vocab_size:
        raise ValueError("DeepSeek-V4 image placeholder token ID is invalid")
    if sum(int(token) == placeholder_token_id for token in token_ids) != len(images):
        raise ValueError(
            "DeepSeek-V4 prompt/image count mismatch: "
            f"placeholders={sum(int(token) == placeholder_token_id for token in token_ids)} "
            f"images={len(images)}"
        )

    expanded: list[int] = []
    attention_block_ids: list[int] = []
    blocks: list[DeepseekV4ImageBlock] = []
    image_iter = iter(enumerate(images))
    for raw_token in token_ids:
        token = int(raw_token)
        if token != placeholder_token_id:
            expanded.append(token)
            attention_block_ids.append(-1)
            continue
        block_id, image = next(image_iter)
        types, perm, block_attention_ids = build_image_block(
            image.n_llm_h,
            image.n_llm_w,
            len(expanded),
            block_id,
        )
        start = len(expanded)
        expanded.extend((vocab_size + types).tolist())
        attention_block_ids.extend(block_attention_ids.tolist())
        blocks.append(
            DeepseekV4ImageBlock(
                start=start,
                types=types,
                perm=perm,
                attention_block_ids=block_attention_ids,
                image=image,
            )
        )
    return expanded, attention_block_ids, blocks


@lru_cache(maxsize=32)
def _vision_cos_sin_cpu(
    n_h: int,
    n_w: int,
    dim: int,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    )
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = (
        torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float()
        * inv_freq
    ).flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def _apply_rotary(
    value: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    dtype = value.dtype
    first, second = value.float().chunk(2, dim=-1)
    cos = cos.to(device=value.device)
    sin = sin.to(device=value.device)
    return torch.cat(
        [first * cos - second * sin, second * cos + first * sin], dim=-1
    ).to(dtype)


class DeepseekV4VisionRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        dtype = hidden.dtype
        value = hidden.float()
        value = value * torch.rsqrt(
            value.square().mean(-1, keepdim=True) + self.eps
        )
        return (self.weight.float() * value).to(dtype)


class DeepseekV4PatchEmbed(nn.Module):
    def __init__(self, config: DeepseekV4VisionConfig):
        super().__init__()
        self.proj = nn.Linear(
            3 * config.patch_size**2,
            config.dim,
            bias=True,
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.proj(patches.flatten(1).to(self.proj.weight.dtype))


class DeepseekV4VisionAttention(nn.Module):
    def __init__(self, config: DeepseekV4VisionConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.wqkv = nn.Linear(config.dim, 3 * config.dim, bias=True)
        self.wo = nn.Linear(config.dim, config.dim, bias=True)

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        tokens = int(hidden.shape[0])
        q, k, v = (
            part.view(tokens, self.n_heads, self.head_dim)
            for part in self.wqkv(hidden).chunk(3, dim=-1)
        )
        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)
        output = F.scaled_dot_product_attention(
            q.transpose(0, 1),
            k.transpose(0, 1),
            v.transpose(0, 1),
            is_causal=False,
        )
        return self.wo(output.transpose(0, 1).reshape(tokens, -1))


class DeepseekV4VisionMLP(nn.Module):
    def __init__(self, config: DeepseekV4VisionConfig):
        super().__init__()
        self.w1 = nn.Linear(config.dim, 2 * config.inter_dim, bias=False)
        self.w2 = nn.Linear(config.inter_dim, config.dim, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        gate, up = self.w1(hidden).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class DeepseekV4VisionBlock(nn.Module):
    def __init__(self, config: DeepseekV4VisionConfig):
        super().__init__()
        self.norm1 = DeepseekV4VisionRMSNorm(config.dim)
        self.attn = DeepseekV4VisionAttention(config)
        self.norm2 = DeepseekV4VisionRMSNorm(config.dim)
        self.mlp = DeepseekV4VisionMLP(config)

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        hidden = hidden + self.attn(self.norm1(hidden), cos, sin)
        return hidden + self.mlp(self.norm2(hidden))


class DeepseekV4VisionModel(nn.Module):
    """Full-attention ViT with the official DeepSeek 2D RoPE layout."""

    def __init__(self, config: DeepseekV4VisionConfig):
        super().__init__()
        self.config = config
        self.rope_dim = config.dim // config.n_heads // 2
        self.patch_embed = DeepseekV4PatchEmbed(config)
        self.blocks = nn.ModuleList(
            [DeepseekV4VisionBlock(config) for _ in range(config.n_layers)]
        )
        self.norm = DeepseekV4VisionRMSNorm(config.dim)

    def forward(
        self,
        patches: torch.Tensor,
        n_h: int,
        n_w: int,
    ) -> torch.Tensor:
        if int(patches.shape[0]) != int(n_h) * int(n_w):
            raise ValueError(
                "DeepSeek-V4 patch/grid mismatch: "
                f"patches={int(patches.shape[0])} grid={n_h}x{n_w}"
            )
        hidden = self.patch_embed(patches)
        cos, sin = _vision_cos_sin_cpu(
            int(n_h),
            int(n_w),
            self.rope_dim,
            self.config.rope_theta,
        )
        for block in self.blocks:
            hidden = block(hidden, cos, sin)
        return self.norm(hidden)


class DeepseekV4Aligner(nn.Module):
    def __init__(self, config: DeepseekV4VisionConfig):
        super().__init__()
        self.downsample_ratio = config.downsample_ratio
        in_dim = config.dim * config.downsample_ratio**2
        self.w1 = nn.Linear(in_dim, config.text_dim, bias=True)
        self.w2 = nn.Linear(config.text_dim, config.text_dim, bias=True)

    def forward(
        self,
        hidden: torch.Tensor,
        n_h: int,
        n_w: int,
    ) -> torch.Tensor:
        ratio = self.downsample_ratio
        hidden = hidden.view(n_h, n_w, -1).permute(2, 0, 1)
        hidden = F.pad(hidden, (0, -n_w % ratio, 0, -n_h % ratio))
        hidden = (
            F.unfold(hidden.unsqueeze(0), ratio, stride=ratio)
            .squeeze(0)
            .transpose(0, 1)
        )
        return self.w2(F.gelu(self.w1(hidden)))


def keep_vision_norms_fp32(module: nn.Module) -> nn.Module:
    """Restore the reference FP32 norm parameters after tower dtype moves."""
    for child in module.modules():
        if isinstance(child, DeepseekV4VisionRMSNorm):
            child.to(dtype=torch.float32)
    return module
