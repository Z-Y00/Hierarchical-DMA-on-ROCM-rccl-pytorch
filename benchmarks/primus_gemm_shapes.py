#!/usr/bin/env python3
"""Primus-Turbo model GEMM shapes for MI355X BF16 training workloads."""

from __future__ import annotations

from dataclasses import asdict, dataclass


PRIMUS_TURBO_PR = "https://github.com/AMD-AGI/Primus-Turbo/pull/265"
PRIMUS_TURBO_COMMIT = "835c6687ff507f62fe9760cb042415154c784283"


@dataclass(frozen=True)
class ModelConfig:
    seqlen: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int


@dataclass(frozen=True)
class GemmAlias:
    model: str
    projection: str
    batch_size: int


@dataclass
class GemmShape:
    m: int
    n: int
    k: int
    aliases: list[GemmAlias]

    def to_dict(self) -> dict[str, object]:
        return {
            "m": self.m,
            "n": self.n,
            "k": self.k,
            "aliases": [asdict(alias) for alias in self.aliases],
        }


MODEL_CONFIGS = {
    "Llama-2-7B": ModelConfig(4096, 4096, 11008, 32, 32, 128, 32000),
    "Llama-2-70B": ModelConfig(4096, 8192, 28672, 64, 8, 128, 32000),
    "Llama-3.1-8B": ModelConfig(8192, 4096, 14336, 32, 8, 128, 128256),
    "Llama-3.1-405B": ModelConfig(
        8192, 16384, 53248, 128, 8, 128, 128256
    ),
    # Llama-4's routed-expert grouped GEMMs are outside this dense GEMM scan.
    "Llama-4-17Bx16E": ModelConfig(
        4096, 5120, 16384, 40, 8, 128, 202048
    ),
    "Llama-4-17Bx128E": ModelConfig(
        4096, 5120, 16384, 40, 8, 128, 202048
    ),
    "Qwen2.5-7B": ModelConfig(8192, 3584, 18944, 28, 4, 128, 152064),
    "Qwen2.5-72B": ModelConfig(8192, 8192, 29568, 64, 8, 128, 152064),
    "Mistral-7B": ModelConfig(4096, 4096, 14336, 32, 8, 128, 32000),
}


# PR #265 resolves these model/GPU/dtype values and unions them with [1, 2, 4].
MI355X_BF16_BATCH_SIZES = {
    "Llama-2-7B": [1, 2, 4, 10],
    "Llama-2-70B": [1, 2, 4, 14],
    "Llama-3.1-8B": [1, 2, 4, 6],
    "Llama-3.1-405B": [1, 2, 4],
    "Llama-4-17Bx16E": [1, 2, 4],
    "Llama-4-17Bx128E": [1, 2, 4],
    "Qwen2.5-7B": [1, 2, 4, 16],
    "Qwen2.5-72B": [1, 2, 4, 16],
    "Mistral-7B": [1, 2, 4],
}


def _projections(config: ModelConfig) -> list[tuple[str, int, int]]:
    qkv_size = (
        config.num_attention_heads + 2 * config.num_key_value_heads
    ) * config.head_dim
    return [
        ("attn_qkv", qkv_size, config.hidden_size),
        ("attn_out", config.hidden_size, config.hidden_size),
        ("mlp_gate_up", 2 * config.intermediate_size, config.hidden_size),
        ("mlp_down", config.hidden_size, config.intermediate_size),
        ("lm_head", config.vocab_size, config.hidden_size),
    ]


def get_primus_mi355x_bf16_shapes() -> list[GemmShape]:
    """Return unique shapes with every model/projection/batch alias retained."""
    by_shape: dict[tuple[int, int, int], GemmShape] = {}
    for model, config in MODEL_CONFIGS.items():
        for batch_size in MI355X_BF16_BATCH_SIZES[model]:
            m = config.seqlen * batch_size
            for projection, n, k in _projections(config):
                key = (m, n, k)
                alias = GemmAlias(model, projection, batch_size)
                if key not in by_shape:
                    by_shape[key] = GemmShape(m, n, k, [])
                by_shape[key].aliases.append(alias)
    return list(by_shape.values())
