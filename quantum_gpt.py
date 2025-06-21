import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

# ==============================================================================
# PART 1: CORE QUANTUMCOMPLEX TENSOR ARITHMETIC
# ==============================================================================

def qc_add(x: Tuple[torch.Tensor, torch.Tensor], y: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """QuantumComplex element-wise addition for tensors x + y."""
    return x[0] + y[0], x[1] + y[1]

def qc_bmm(x: Tuple[torch.Tensor, torch.Tensor], y: Tuple[torch.Tensor, torch.Tensor], theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """QuantumComplex batch matrix multiplication."""
    xa, xb = x
    ya, yb = y
    j_squared = -torch.cos(2 * theta)
    out_a = torch.bmm(xa, ya) + torch.bmm(xb, yb) * j_squared
    out_b = torch.bmm(xa, yb) + torch.bmm(xb, ya)
    return out_a, out_b

# ==============================================================================
# PART 2: MODEL CONFIGURATION
# ==============================================================================

@dataclass
class QTransformerConfig:
    vocab_size: int = 20 # For our toy task
    seq_len: int = 12
    n_layers: int = 4 # Encoder and Decoder layers
    n_heads: int = 4
    embed_dim: int = 64
    ffn_dim_multiplier: int = 4
    dropout_prob: float = 0.1
    initial_theta: float = 0.0

# ==============================================================================
# PART 3: ADVANCED NEURAL NETWORK MODULES
# ==============================================================================

class QComplexDense(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.weight_a = nn.Parameter(torch.empty(output_dim, input_dim))
        self.weight_b = nn.Parameter(torch.empty(output_dim, input_dim))
        self.bias_a = nn.Parameter(torch.empty(output_dim))
        self.bias_b = nn.Parameter(torch.empty(output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_a)
        nn.init.xavier_uniform_(self.weight_b)
        nn.init.zeros_(self.bias_a)
        nn.init.zeros_(self.bias_b)
        
    def forward(self, x, theta):
        xa, xb = x
        j_squared = -torch.cos(2 * theta)
        out_a = F.linear(xa, self.weight_a, self.bias_a) + F.linear(xb, self.weight_b * j_squared, None)
        out_b = F.linear(xa, self.weight_b, self.bias_b) + F.linear(xb, self.weight_a, None)
        return out_a, out_b

class Q_RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xa, xb = x
        # Normalize based on the combined magnitude of the components
        norm_val = torch.rsqrt((xa.pow(2) + xb.pow(2)).mean(-1, keepdim=True) + self.eps)
        return xa * norm_val * self.gamma, xb * norm_val * self.gamma

class QC_RoPE(nn.Module):
    def __init__(self, dim: int, seq_len: int):
        super().__init__()
        theta = 10000.0 ** (-2.0 * torch.arange(0, dim, 2).float() / dim)
        t = torch.arange(seq_len, device=theta.device).float()
        freqs = torch.einsum("i,j->ij", t, theta)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        self.register_buffer("freqs_cis", freqs_cis)

    def forward(self, x_qc):
        xa, xb = x_qc
        freqs = self.freqs_cis[:xa.shape[1]]
        xa_r = xa.float().reshape(*xa.shape[:-1], -1, 2)
        xb_r = xb.float().reshape(*xb.shape[:-1], -1, 2)
        rotated_a = self._apply_rotary_emb(xa_r, freqs).flatten(2)
        rotated_b = self._apply_rotary_emb(xb_r, freqs).flatten(2)
        return rotated_a, rotated_b

    def _apply_rotary_emb(self, x, freqs):
        x_complex = torch.view_as_complex(x)
        freqs_complex = freqs.unsqueeze(0)
        x_rotated = x_complex * freqs_complex
        return torch.view_as_real(x_rotated)

class QComplexAttention(nn.Module):
    """A flexible attention module for both self- and cross-attention."""
    def __init__(self, config: QTransformerConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.embed_dim // config.n_heads
        
        self.q_proj = QComplexDense(config.embed_dim, config.embed_dim)
        self.k_proj = QComplexDense(config.embed_dim, config.embed_dim)
        self.v_proj = QComplexDense(config.embed_dim, config.embed_dim)
        self.o_proj = QComplexDense(config.embed_dim, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.thetas_head = nn.Parameter(torch.full((self.n_heads,), config.initial_theta))

    def forward(self, x_q, x_kv, mask, rope, layer_theta):
        q = self.q_proj(x_q, layer_theta)
        k = self.k_proj(x_kv, layer_theta)
        v = self.v_proj(x_kv, layer_theta)

        q_a, q_b = self._shape(q[0]), self._shape(q[1])
        k_a, k_b = self._shape(k[0]), self._shape(k[1])
        v_a, v_b = self._shape(v[0]), self._shape(v[1])
        
        q_a, q_b = rope((q_a, q_b))
        k_a, k_b = rope((k_a, k_b))
        
        head_thetas = self.thetas_head.view(1, -1, 1, 1)
        scores_a, scores_b = qc_bmm((q_a, q_b), (k_a.transpose(2, 3), k_b.transpose(2, 3)), head_thetas)
        
        scores_magnitude = torch.sqrt(scores_a**2 + scores_b**2 + 1e-8) / math.sqrt(self.head_dim)
        if mask is not None: scores_magnitude = scores_magnitude.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores_magnitude, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output_a = torch.bmm(attention_weights, v_a)
        output_b = torch.bmm(attention_weights, v_b)
        
        output_a, output_b = self._unshape(output_a), self._unshape(output_b)
        return self.o_proj((output_a, output_b), layer_theta)

    def _shape(self, x): return x.view(x.shape[0], x.shape[1], self.n_heads, self.head_dim).transpose(1, 2)
    def _unshape(self, x): return x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.n_heads * self.head_dim)

class QC_SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w_gate = QComplexDense(dim, hidden_dim)
        self.w_up = QComplexDense(dim, hidden_dim)
        self.w_out = QComplexDense(hidden_dim, dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, theta):
        gate_qc = self.w_gate(x, theta)
        up_qc = self.w_up(x, theta)
        gate_magnitude = torch.sqrt(gate_qc[0]**2 + gate_qc[1]**2)
        gated_val = F.silu(gate_magnitude)
        gated_up_a, gated_up_b = gated_val * up_qc[0], gated_val * up_qc[1]
        return self.w_out((self.dropout(gated_up_a), self.dropout(gated_up_b)), theta)

# ==============================================================================
# PART 4: ENCODER AND DECODER BLOCKS
# ==============================================================================

class QTransformerEncoderLayer(nn.Module):
    def __init__(self, config: QTransformerConfig):
        super().__init__()
        self.self_attn = QComplexAttention(config)
        self.ffn = QC_SwiGLU(config.embed_dim, config.embed_dim * config.ffn_dim_multiplier)
        self.norm1 = Q_RMSNorm(config.embed_dim)
        self.norm2 = Q_RMSNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.layer_theta = nn.Parameter(torch.tensor(config.initial_theta))

    def forward(self, x, mask, rope):
        norm_x = self.norm1(x)
        attn_out = self.self_attn(norm_x, norm_x, mask, rope, self.layer_theta)
        x = qc_add(x, (self.dropout(attn_out[0]), self.dropout(attn_out[1])))
        
        norm_x = self.norm2(x)
        ffn_out = self.ffn(norm_x, self.layer_theta)
        x = qc_add(x, (self.dropout(ffn_out[0]), self.dropout(ffn_out[1])))
        return x

class QTransformerDecoderLayer(nn.Module):
    def __init__(self, config: QTransformerConfig):
        super().__init__()
        self.masked_self_attn = QComplexAttention(config)
        self.cross_attn = QComplexAttention(config)
        self.ffn = QC_SwiGLU(config.embed_dim, config.embed_dim * config.ffn_dim_multiplier)
        self.norm1 = Q_RMSNorm(config.embed_dim)
        self.norm2 = Q_RMSNorm(config.embed_dim)
        self.norm3 = Q_RMSNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.layer_theta = nn.Parameter(torch.tensor(config.initial_theta))

    def forward(self, x, encoder_output, src_mask, tgt_mask, rope):
        # Masked Self-Attention
        norm_x = self.norm1(x)
        attn_out = self.masked_self_attn(norm_x, norm_x, tgt_mask, rope, self.layer_theta)
        x = qc_add(x, (self.dropout(attn_out[0]), self.dropout(attn_out[1])))
        
        # Cross-Attention
        norm_x = self.norm2(x)
        cross_attn_out = self.cross_attn(norm_x, encoder_output, src_mask, rope, self.layer_theta)
        x = qc_add(x, (self.dropout(cross_attn_out[0]), self.dropout(cross_attn_out[1])))

        # Feed-Forward Network
        norm_x = self.norm3(x)
        ffn_out = self.ffn(norm_x, self.layer_theta)
        x = qc_add(x, (self.dropout(ffn_out[0]), self.dropout(ffn_out[1])))
        return x

# ==============================================================================
# PART 5: THE FULL Q-TRANSFORMER MODEL
# ==============================================================================

class QTransformer(nn.Module):
    def __init__(self, config: QTransformerConfig):
        super().__init__()
        self.config = config
        self.embedding = QComplexDense(config.vocab_size, config.embed_dim)
        self.rope = QC_RoPE(config.embed_dim // config.n_heads, config.seq_len)
        
        self.encoder_layers = nn.ModuleList([QTransformerEncoderLayer(config) for _ in range(config.n_layers)])
        self.decoder_layers = nn.ModuleList([QTransformerDecoderLayer(config) for _ in range(config.n_layers)])
        
        self.output_layer = nn.Linear(config.embed_dim, config.vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # Embed source and target
        src_one_hot = F.one_hot(src, num_classes=self.config.vocab_size).float()
        tgt_one_hot = F.one_hot(tgt, num_classes=self.config.vocab_size).float()
        
        src_qc = self.embedding((src_one_hot, torch.zeros_like(src_one_hot)), torch.tensor(0.0))
        tgt_qc = self.embedding((tgt_one_hot, torch.zeros_like(tgt_one_hot)), torch.tensor(0.0))

        # Encoder pass
        encoder_output = src_qc
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, src_mask, self.rope)
        
        # Decoder pass
        decoder_output = tgt_qc
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output, src_mask, tgt_mask, self.rope)
            
        # Project to logits (using only 'a' component)
        return self.output_layer(decoder_output[0])

def create_masks(src, tgt, pad_idx, device):
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2) # (B, 1, 1, L_src)
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2) # (B, 1, 1, L_tgt)
    
    seq_len = tgt.size(1)
    tgt_sub_mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).bool() # (L_tgt, L_tgt)
    
    tgt_mask = tgt_pad_mask & tgt_sub_mask # Combine padding and causal masks
    return src_mask, tgt_mask

# ==============================================================================
# PART 6: TRAINING AND INFERENCE DEMONSTRATION
# ==============================================================================
if __name__ == '__main__':
    # --- Configuration ---
    config = QTransformerConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    PAD_TOKEN = 0
    SOS_TOKEN = 1 # Start Of Sentence
    EOS_TOKEN = 2 # End Of Sentence
    
    # --- Data Generation for "Reverse the Sequence" Task ---
    def generate_batch(batch_size, seq_len, vocab_size):
        # Vocab: 0=pad, 1=sos, 2=eos, 3..19 are numbers
        src = torch.randint(3, vocab_size, (batch_size, seq_len - 2))
        tgt = torch.flip(src, [1])
        
        # Add SOS and EOS tokens
        sos_tensor = torch.full((batch_size, 1), SOS_TOKEN)
        eos_tensor = torch.full((batch_size, 1), EOS_TOKEN)
        
        src = torch.cat([sos_tensor, src, eos_tensor], dim=1)
        tgt = torch.cat([sos_tensor, tgt, eos_tensor], dim=1)
        return src.to(device), tgt.to(device)

    # --- Model, Loss, Optimizer ---
    model = QTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M parameters.")

    # --- Training Loop ---
    print("\n--- Starting Training ---")
    for epoch in range(201):
        model.train()
        src, tgt = generate_batch(32, config.seq_len, config.vocab_size)
        
        # For teacher forcing, decoder input is target shifted right
        tgt_input = tgt[:, :-1]
        # Target for loss is target shifted left
        tgt_output = tgt[:, 1:]
        
        src_mask, tgt_mask = create_masks(src, tgt_input, PAD_TOKEN, device)
        
        optimizer.zero_grad()
        logits = model(src, tgt_input, src_mask, tgt_mask)
        
        loss = criterion(logits.view(-1, config.vocab_size), tgt_output.reshape(-1))
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f}")

    # --- Greedy Decode Inference ---
    def greedy_decode(model, src, max_len, device):
        model.eval()
        with torch.no_grad():
            src_mask, _ = create_masks(src, src, PAD_TOKEN, device)
            
            src_one_hot = F.one_hot(src, num_classes=config.vocab_size).float()
            src_qc = model.embedding((src_one_hot, torch.zeros_like(src_one_hot)), torch.tensor(0.0))
            
            encoder_output = src_qc
            for layer in model.encoder_layers:
                encoder_output = layer(encoder_output, src_mask, model.rope)
                
            tgt_tokens = torch.full((src.size(0), 1), SOS_TOKEN, dtype=torch.long, device=device)
            
            for _ in range(max_len - 1):
                tgt_input = tgt_tokens
                _, tgt_mask = create_masks(src, tgt_input, PAD_TOKEN, device)
                
                tgt_one_hot = F.one_hot(tgt_input, num_classes=config.vocab_size).float()
                tgt_qc = model.embedding((tgt_one_hot, torch.zeros_like(tgt_one_hot)), torch.tensor(0.0))

                decoder_output = tgt_qc
                for layer in model.decoder_layers:
                    decoder_output = layer(decoder_output, encoder_output, src_mask, tgt_mask, model.rope)
                    
                logits = model.output_layer(decoder_output[0][:, -1])
                next_token = logits.argmax(dim=-1).unsqueeze(1)
                tgt_tokens = torch.cat([tgt_tokens, next_token], dim=1)
                
                # Stop if all sequences have generated EOS
                if (next_token == EOS_TOKEN).all():
                    break
                    
        return tgt_tokens
        
    print("\n--- Running Inference ---")
    test_src, test_tgt = generate_batch(2, config.seq_len, config.vocab_size)
    predicted_tgt = greedy_decode(model, test_src, config.seq_len, device)

    for i in range(test_src.size(0)):
        print("-" * 40)
        print(f"Sample {i+1}")
        print(f"  Source:    {test_src[i].cpu().numpy().tolist()}")
        print(f"  Target:    {test_tgt[i].cpu().numpy().tolist()}")
        print(f"  Predicted: {predicted_tgt[i].cpu().numpy().tolist()}")
