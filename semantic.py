"""DeepX Embedding v1 chạy CPU để tính điểm tương đồng ngữ nghĩa.

Không dùng nguyên ``DeepXEmbed.from_pretrained`` của upstream vì hàm đó ép
``.half()``, ``autocast(device_type="cuda")``, nạp tokenizer qua ``AutoTokenizer``
(lỗi với transformers 4.57 do tokenizer_config kiểu Gemma4) và đệm trái. Ở đây:

- Kiến trúc lấy theo ``DeepXConfig()`` của vendor: shape trong checkpoint khớp mặc
  định này (head_dim 256, MLP 6144/12288), không khớp ``config.json`` của model.
- RoPE/RoDE (``rope_*``, ``depth_rotary_dims``) chỉ dùng ở nhánh softmax; với
  ``_alpha_override = 0.0`` như upstream nhánh đó bị bỏ qua nên không ảnh hưởng đầu ra.
- Nhánh GDN-2 thay bằng bản chunk-parallel PyTorch thuần, cùng công thức với kernel FLA
  ``chunk_gated_delta_rule`` mà model được huấn luyện (gate decay/erase lấy trung bình
  theo head). Fallback tuần tự của upstream dùng gate theo kênh — khác lúc huấn luyện và
  chậm trên CPU.
- Đệm phải: GDN-2 và short conv đều nhân quả nên token đệm ở cuối không ảnh hưởng
  token thật; attention pooling đã che token đệm.
"""

from __future__ import annotations

import os
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "deepx-embedding-v1"
VENDOR_DIR = ROOT / "vendor" / "deepx-embed"
REQUIRED_FILES = ("deepx_v1.0.pt", "token_embedding.pt", "id_remap.pt", "tokenizer.json")
MAX_TOKENS = 1024
GDN_CHUNK = 64


@dataclass
class LoadReport:
    missing_keys: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)
    dtype: str = "float32"
    seconds: float = 0.0
    checkpoint_step: int | None = None


def _gdn2_chunked(q, k, v, beta, g, chunk_size: int = GDN_CHUNK):
    """Gated delta rule (cùng công thức với FLA), tính theo chunk.

    q, k, v: (B, H, T, D); beta, g: (B, H, T) với g = log decay <= 0.
    Mỗi bước: S = exp(g_t) S; u = beta_t (v_t - k_t S); S += k_t^T u; o_t = q_t S.
    """
    import torch
    import torch.nn.functional as F

    B, H, T, D = k.shape
    Dv = v.shape[-1]
    pad = (-T) % chunk_size
    if pad:
        # Đệm cuối với k = v = beta = g = 0: không làm thay đổi các vị trí trước đó.
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        beta, g = (F.pad(x, (0, pad)) for x in (beta, g))
    N, C = (T + pad) // chunk_size, chunk_size
    q, k, v = (x.reshape(B, H, N, C, -1) for x in (q, k, v))
    beta, g = beta.reshape(B, H, N, C), g.reshape(B, H, N, C)

    G = g.cumsum(-1)
    causal = torch.ones(C, C, dtype=torch.bool, device=k.device).tril()
    strict = causal.clone().fill_diagonal_(False)
    log_ratio = G[..., :, None] - G[..., None, :]
    decay = torch.exp(log_ratio.masked_fill(~causal, float("-inf")))

    kk = k @ k.transpose(-1, -2)
    A = (beta[..., :, None] * kk * decay).masked_fill(~strict, 0.0)
    eye = torch.eye(C, dtype=k.dtype, device=k.device)
    lhs = (eye + A).reshape(-1, C, C)
    rhs = torch.cat([beta[..., None] * v, (beta * G.exp())[..., None] * k], dim=-1).reshape(-1, C, Dv + D)
    solved = torch.linalg.solve_triangular(lhs, rhs, upper=False, unitriangular=True)
    solved = solved.reshape(B, H, N, C, Dv + D)
    w_v, w_k = solved[..., :Dv], solved[..., Dv:]

    qk = (q @ k.transpose(-1, -2)) * decay
    q_decayed = q * G.exp()[..., None]
    k_to_end = k * torch.exp(G[..., -1:] - G)[..., None]
    end_decay = G[..., -1].exp()

    state = q.new_zeros(B, H, D, Dv)
    out = []
    for n in range(N):
        u = w_v[:, :, n] - w_k[:, :, n] @ state
        out.append(q_decayed[:, :, n] @ state + qk[:, :, n] @ u)
        state = end_decay[:, :, n, None, None] * state + k_to_end[:, :, n].transpose(-1, -2) @ u
    return torch.cat(out, dim=2)[:, :, :T]


def _gdn2_attention_cpu(self, q, k, v, erase_gate, write_gate, decay, attention_mask=None):
    """Thay ``GatedDeltaNet2Attention._gdn2_attention``: chuẩn bị đầu vào như nhánh FLA."""
    import torch
    from einops import rearrange

    H_kv = self.num_kv_heads
    dtype = q.dtype
    k_exp, v_exp = self._expand_kv(k), self._expand_kv(v)
    erase, write, alpha = (
        self._expand_kv(rearrange(x, "b t (h d) -> b h t d", h=H_kv))
        for x in (erase_gate, write_gate, decay)
    )
    g = torch.log(alpha.float().clamp(min=1e-6)).mean(dim=-1)
    beta = erase.float().mean(dim=-1)
    out = _gdn2_chunked(q.float(), k_exp.float(), (write * v_exp).float(), beta, g)
    return out.to(dtype)


_LORA_TARGETS = {
    "q_proj": "self_attn.q_proj.weight", "k_proj": "self_attn.k_proj.weight",
    "v_proj": "self_attn.v_proj.weight", "o_proj": "self_attn.o_proj.weight",
    "gate_proj": "mlp.gate_proj.weight", "up_proj": "mlp.up_proj.weight", "down_proj": "mlp.down_proj.weight",
}


def _run_layer(layer, lora, loop_idx, hidden_states, attention_mask, position_ids, repeats=1):
    """Chạy ``layer`` ``repeats`` lần với LoRA của vòng lặp đã gộp vào trọng số.

    Upstream áp LoRA bằng một phép nhân ma trận đầy đủ thứ hai cho mỗi projection
    (``F.linear(x, A @ B)``). Gộp W + A @ B một lần mỗi vòng lặp cho kết quả tương
    đương; checkpoint v1.0 có toàn bộ LoRA bằng 0 nên khi đó bỏ hẳn bước này.
    """
    from torch.func import functional_call

    kwargs = {"attention_mask": attention_mask, "position_ids": position_ids, "loop_idx": loop_idx}
    if lora is None:
        for _ in range(repeats):
            hidden_states = layer(hidden_states, **kwargs)
        return hidden_states
    params = {
        path: layer.get_parameter(path) + lora.get_delta(name, loop_idx)
        for name, path in _LORA_TARGETS.items()
    }
    for _ in range(repeats):
        hidden_states = functional_call(layer, params, (hidden_states,), kwargs)
    return hidden_states


def _phase_forward_cpu(self, hidden_states, attention_mask=None, position_ids=None):
    """Thay ``HyperloopPhase.forward`` khi suy luận: [Wide, Narrow×4] hoặc [Narrow×4, Wide]."""
    wide_lora = None if self._lora_is_zero else self.wide_lora
    narrow_lora = None if self._lora_is_zero else self.narrow_lora
    for i in range(self.num_loops):
        wide = (self.shared_wide, wide_lora, i, 1)
        narrow = (self.shared_narrow, narrow_lora, i, 4)
        for layer, lora, loop_idx, repeats in ((wide, narrow) if self.wide_first else (narrow, wide)):
            hidden_states = _run_layer(layer, lora, loop_idx, hidden_states, attention_mask, position_ids, repeats)
    return hidden_states


def missing_files(model_dir: Path = MODEL_DIR) -> list[str]:
    return [n for n in REQUIRED_FILES if not (model_dir / n).is_file() or (model_dir / n).stat().st_size == 0]


class SemanticEncoder:
    """Nạp DeepX lên CPU và mã hóa văn bản thành vector 1536 chiều đã chuẩn hóa L2.

    Dùng với ``indexer.LibraryIndex(vector_model=SemanticEncoder.name, coarse_dim=256)``.
    """

    name = "deepx-embedding-v1"

    def __init__(self, model_dir: Path | str = MODEL_DIR, dtype: str = "float32", threads: int | None = None):
        model_dir = Path(model_dir)
        missing = missing_files(model_dir)
        if missing:
            raise FileNotFoundError(f"Thiếu file model trong {model_dir}: {', '.join(missing)}")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        import torch
        from transformers import PreTrainedTokenizerFast

        if str(VENDOR_DIR) not in sys.path:
            sys.path.insert(0, str(VENDOR_DIR))
        from config import DeepXConfig
        from modeling.embedding_model import DeepXEmbeddingModel

        if threads:
            torch.set_num_threads(threads)
        started = time.perf_counter()
        torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(model_dir / "tokenizer.json"),
            pad_token="<pad>", unk_token="<unk>", bos_token="<bos>", eos_token="<eos>",
        )
        tokenizer.padding_side = "right"
        id_remap = torch.load(model_dir / "id_remap.pt", map_location="cpu", weights_only=True)
        # Token bị cắt khỏi vocab (remap = -1) được quy về <unk>.
        id_remap = torch.where(id_remap < 0, id_remap[tokenizer.unk_token_id], id_remap)

        embed = torch.load(model_dir / "token_embedding.pt", map_location="cpu", weights_only=True)
        embed = embed["weight"] if isinstance(embed, dict) else embed

        config = DeepXConfig(dtype=dtype)
        config.vocab_size = int(embed.shape[0])
        backbone = DeepXEmbeddingModel(config)
        ckpt = torch.load(model_dir / "deepx_v1.0.pt", map_location="cpu", weights_only=True)
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        result = backbone.load_state_dict(state, strict=False)
        step = ckpt.get("step") if isinstance(ckpt, dict) else None
        del ckpt, state
        backbone = backbone.to(torch_dtype).eval().requires_grad_(False)

        for module in backbone.modules():
            if hasattr(module, "path_mix_logit"):
                module._alpha_override = 0.0
                module._gdn2_attention = types.MethodType(_gdn2_attention_cpu, module)
        for phase in (backbone.phase1, backbone.phase2):
            lora_params = list(phase.wide_lora.parameters()) + list(phase.narrow_lora.parameters())
            phase._lora_is_zero = all(not p.any() for p in lora_params)
            phase.forward = types.MethodType(_phase_forward_cpu, phase)

        self.torch = torch
        self.tokenizer = tokenizer
        self.id_remap = id_remap
        self.embed = embed  # giữ bf16 như file gốc; ép kiểu sau khi tra cứu
        self.backbone = backbone
        self.dtype = torch_dtype
        self.report = LoadReport(
            missing_keys=list(result.missing_keys),
            unexpected_keys=list(result.unexpected_keys),
            dtype=dtype,
            seconds=time.perf_counter() - started,
            checkpoint_step=step,
        )

    def encode(
        self, texts: list[str] | str, batch_size: int = 4, max_tokens: int = MAX_TOKENS,
        progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        torch = self.torch
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        if not texts:
            return np.zeros((0, self.backbone.config.hidden_size), dtype=np.float32)
        # Gom các đoạn dài tương đương vào cùng batch để giảm token đệm.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        result = np.zeros((len(texts), self.backbone.config.hidden_size), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(order), batch_size):
                idx = order[start : start + batch_size]
                enc = self.tokenizer(
                    [texts[i] for i in idx], padding=True, truncation=True,
                    max_length=max_tokens, return_tensors="pt",
                )
                ids = self.id_remap[enc["input_ids"]]
                mask = enc["attention_mask"]
                hidden = torch.nn.functional.embedding(ids, self.embed).to(self.dtype)
                emb = self.backbone(hidden, attention_mask=mask, normalize=True)
                result[idx] = emb.float().numpy()
                if progress is not None:
                    progress(min(start + batch_size, len(order)), len(order))
        return result[0] if single else result
