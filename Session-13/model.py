import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class SmolLM2Config:
    hidden_size: int = 576
    intermediate_size: int = 1536
    num_attention_heads: int = 9
    num_hidden_layers: int = 30
    num_key_value_heads: int = 3
    vocab_size: int = 49152
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-05
    rope_theta: float = 100000.0
    tie_word_embeddings: bool = True
    bos_token_id: int = 0
    eos_token_id: int = 0

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [batch, heads, seq_len, head_dim]
    # cos, sin: [1, 1, seq_len, head_dim]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_position_embeddings = max_position_embeddings
        self.dim = dim
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len):
        if seq_len > self.cos_cached.shape[2]:
             self._set_cos_sin_cache(seq_len)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

class SmolLM2MLP(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class SmolLM2Attention(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                             f" and `num_heads`: {self.num_heads}).")

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, base=config.rope_theta)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rotary_emb(v, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Repeat k/v heads if num_key_value_heads < num_heads (GQA)
        k = torch.repeat_interleave(k, dim=1, repeats=self.num_key_value_groups)
        v = torch.repeat_interleave(v, dim=1, repeats=self.num_key_value_groups)
        
        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            attn_weights = attn_weights + mask
            
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(attn_output)

class SmolLM2Block(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = SmolLM2Attention(config)
        self.mlp = SmolLM2MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, mask=None):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, mask=mask)
        x = residual + x
        
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x

class SmolLM2Model(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([SmolLM2Block(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        
        # Create causal mask
        seq_len = x.shape[1]
        mask = torch.full((seq_len, seq_len), float("-inf"), device=x.device)
        mask = torch.triu(mask, diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0) # [1, 1, seq_len, seq_len]
        
        for layer in self.layers:
            x = layer(x, mask=mask)
            
        return self.norm(x)

class SmolLM2ForCausalLM(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.config = config
        self.model = SmolLM2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids):
        # input_ids: [batch, seq_len]
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)
        return logits

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.config.max_position_embeddings:]
            # get the predictions
            logits = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
