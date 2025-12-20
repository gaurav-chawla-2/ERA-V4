import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List

@dataclass
class DeepSeekConfig:
    hidden_size: int = 576
    intermediate_size: int = 1536  # Size for each expert
    num_attention_heads: int = 9
    num_hidden_layers: int = 30
    vocab_size: int = 49152
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-5
    rope_theta: float = 100000.0
    tie_word_embeddings: bool = True
    
    # MLA specifics
    kv_lora_rank: int = 128
    q_lora_rank: int = 128  # Set to None or 0 to disable Q compression
    nope_head_dim: int = 32
    rope_head_dim: int = 32
    v_head_dim: int = 64   # Often equals nope + rope, or just head_dim
    
    # MoE specifics
    num_experts: int = 8
    num_shared_experts: int = 1
    top_k: int = 2
    moe_intermediate_size: int = 512 # Reduced from 1536 to match active parameters of dense model (512*3 = 1536)
    
    # Load balancing
    aux_loss_alpha: float = 0.0 # 0 for loss-less
    seq_aux: bool = False

    def __post_init__(self):
        # Derive v_head_dim if consistent with others
        # Usually in DeepSeek V2, v_head_dim = head_dim
        pass

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
    # q, k: [..., seq_len, head_dim] (after broadcasting)
    # cos, sin: [..., seq_len, head_dim]
    # We assume Last dim is head_dim
    
    # Expand dims if necessary.
    # q is [B, H, T, D] or similar.
    # cos is [1, 1, T, D] typically.
    
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
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
             self._set_cos_sin_cache(max(seq_len, int(seq_len * 1.5)))
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

class MLA(nn.Module):
    """
    Multi-Head Latent Attention (MLA)
    """
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.nope_head_dim = config.nope_head_dim
        self.rope_head_dim = config.rope_head_dim
        self.v_head_dim = config.v_head_dim
        
        # Effective head dimension for attention scores
        self.q_head_dim = self.nope_head_dim + self.rope_head_dim
        
        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = config.q_lora_rank
        
        # Q Compression
        if self.q_lora_rank > 0:
            self.q_down = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
            self.q_up_norm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_up = nn.Linear(self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)
        else:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.q_head_dim, bias=False)
            
        # KV Compression
        self.kv_down = nn.Linear(self.hidden_size, self.kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        
        # UP projections:
        # W_UK: projects to (num_heads * (nope + rope))
        self.kv_up_k = nn.Linear(self.kv_lora_rank, self.num_heads * self.q_head_dim, bias=False)
        # W_UV: projects to (num_heads * v_head_dim)
        self.kv_up_v = nn.Linear(self.kv_lora_rank, self.num_heads * self.v_head_dim, bias=False)
        
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(self.rope_head_dim, 
                                          max_position_embeddings=config.max_position_embeddings, 
                                          base=config.rope_theta)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # --- Query ---
        if self.q_lora_rank > 0:
            q_latent = self.q_down(x)
            q_latent = self.q_up_norm(q_latent)
            q = self.q_up(q_latent)
        else:
            q = self.q_proj(x)
            
        q = q.view(batch_size, seq_len, self.num_heads, self.q_head_dim)
        # Split Q into nope and rope
        q_nope, q_pe = torch.split(q, [self.nope_head_dim, self.rope_head_dim], dim=-1)
        
        # --- Key/Value ---
        kv_latent = self.kv_down(x)
        kv_latent = self.kv_norm(kv_latent)
        
        k = self.kv_up_k(kv_latent)
        v = self.kv_up_v(kv_latent)
        
        k = k.view(batch_size, seq_len, self.num_heads, self.q_head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.v_head_dim)
        
        # Split K into nope and rope
        k_nope, k_pe = torch.split(k, [self.nope_head_dim, self.rope_head_dim], dim=-1)
        
        # --- RoPE ---
        # Reshape for RoPE: [Batch, Heads, Seq, Dim] for rotation function
        q_pe = q_pe.transpose(1, 2) # [B, H, S, D]
        k_pe = k_pe.transpose(1, 2)
        
        cos, sin = self.rotary_emb(v, seq_len) # v is dummy here just for dtype/device
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin)
        
        # Transpose back to [B, S, H, D] for recombination? 
        # Actually standard attention expects [B, H, S, D], so let's keep it there.
        q_pe = q_pe.transpose(1, 2) # [B, S, H, D]
        k_pe = k_pe.transpose(1, 2) # [B, S, H, D]
        
        # Combine nope and rope
        # q = [q_nope; q_pe]
        q = torch.cat([q_nope, q_pe], dim=-1)
        k = torch.cat([k_nope, k_pe], dim=-1)
        
        # Prepare for attention: [B, H, S, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Attention
        # Note: MLA allows KV compression. The attention math is standard once decompressed.
        # But wait, memory savings come from NOT decompressing everything fully for KV cache.
        # During training, we just compute as usual. 
        # For this task (training), this implementation is sufficient.
        
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.v_head_dim)
        return self.o_proj(attn_output)

class DeepSeekExpert(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class DeepSeekMoE(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_shared = config.num_shared_experts
        self.top_k = config.top_k
        self.hidden_size = config.hidden_size
        
        # Shared Experts
        self.shared_experts = nn.ModuleList([
            DeepSeekExpert(config.hidden_size, config.moe_intermediate_size) 
            for _ in range(self.num_shared)
        ])
        
        # Routed Experts
        self.routed_experts = nn.ModuleList([
            DeepSeekExpert(config.hidden_size, config.moe_intermediate_size)
            for _ in range(self.num_experts)
        ])
        
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        
        # Loss-less load balancing bias (DeepSeek-V3 style)
        # We start with 0 bias. Can be updated dynamically.
        self.register_buffer("router_bias", torch.zeros(config.num_experts))

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        flat_x = x.view(-1, hidden_dim)
        
        # Shared Experts Path
        shared_output = 0
        for expert in self.shared_experts:
            shared_output += expert(flat_x)
            
        # Routed Experts Path
        # DeepSeek-V3: logits = x @ w + bias
        router_logits = self.router(flat_x) + self.router_bias
        
        # Select TopK
        # we still use TopK to determine weights, but we will compute all experts
        # to avoid MPS synchronization issues with dynamic indexing.
        routing_weights = F.softmax(router_logits, dim=1)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # Normalize weights
        topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
        
        # Create a full sparse weight matrix [N, NumExperts]
        # logic: scatter topk_weights into a zero tensor at topk_indices
        expert_weights = torch.zeros_like(routing_weights)
        expert_weights.scatter_(1, topk_indices, topk_weights)
        
        # Load Balancing Bias Update (Training Only)
        if self.training:
           target_load = self.top_k / self.num_experts
           # Current load is fraction of tokens that selected this expert
           # We can check expert_weights > 0
           current_load = (expert_weights > 0).float().mean(dim=0)
           load_error = current_load - target_load
           bias_lr = 1e-3 
           self.router_bias -= bias_lr * torch.sign(load_error)

        # Dense Execution of Experts (Optimized for MPS/No-Sync)
        # We run all experts. Since they are small (512), this is equivalent to
        # running a single MLP of size 8*512 = 4096. 
        # Session 13 was 1536. So this is ~2.7x slower than S13, but infinitely faster than sync.
        
        # Stacked Expert Execution would be best, but ModuleList is easiest
        expert_outputs = []
        for expert in self.routed_experts:
            expert_outputs.append(expert(flat_x))
            
        # Stack: [N, NumExperts, Hidden]
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # Weighted Sum
        # weights: [N, NumExperts] -> [N, NumExperts, 1]
        weighted_expert_output = (expert_outputs * expert_weights.unsqueeze(-1)).sum(dim=1)
        
        # Combined Result
        return (shared_output + weighted_expert_output).view(batch_size, seq_len, hidden_dim)

class DeepSeekBlock(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = MLA(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = DeepSeekMoE(config)

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

class DeepSeekModel(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DeepSeekBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        
        # Causal mask (if needed, though F.scaled_dot_product_attention handles causal)
        # But we need it if we manually did attention. MLA implementation above uses F.scaled_dot_product_attention(is_causal=True).
        # So we don't strictly pass 'mask' down unless we used a manual implementation.
        # But for 'generate' we might need to handle cache? 
        # For this exercise (Train 10k steps), we will just rely on Flash Attention causal flag.
        
        for layer in self.layers:
            x = layer(x)
            
        return self.norm(x)

class DeepSeekForCausalLM(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.config = config
        self.model = DeepSeekModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids):
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)
        return logits
    
    def generate(self, idx, max_new_tokens):
        # idx: [B, T]
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.max_position_embeddings:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
