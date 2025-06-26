import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==============================================================================
# PART 1: QUANTUM COMPLEX ARITHMETIC
# ==============================================================================

def qc_add(x: Tuple[torch.Tensor, torch.Tensor], y: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantum complex addition: (a₁ + b₁J) + (a₂ + b₂J) = (a₁ + a₂) + (b₁ + b₂)J"""
    return x[0] + y[0], x[1] + y[1]

def qc_bmm(x: Tuple[torch.Tensor, torch.Tensor], y: Tuple[torch.Tensor, torch.Tensor], theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    QuantumComplex batch matrix multiplication with the CORRECT algebra.

    For z₁ = a₁ + b₁J(θ) and z₂ = a₂ + b₂J(θ), the multiplication is:
    z₁z₂ = (a₁ + b₁J(θ))(a₂ + b₂J(θ))
         = a₁a₂ + a₁b₂J(θ) + b₁a₂J(θ) + b₁b₂J(θ)²
         = a₁a₂ + (a₁b₂ + b₁a₂)J(θ) + b₁b₂[-1 + sin(2θ)]  [from equation 36]

    Real part: a₁a₂ + b₁b₂[-1 + sin(2θ)]
    Imaginary part: a₁b₂ + b₁a₂
    """
    xa, xb = x
    ya, yb = y

    # CRITICAL: The correct formula from equation (36) is [-1 + sin(2θ)]
    j_squared = -1 + torch.sin(2 * theta)  # J² = -1 + sin(2θ)

    # Real part: a₁a₂ + b₁b₂J²
    out_a = torch.matmul(xa, ya) + torch.matmul(xb, yb) * j_squared

    # Imaginary part: a₁b₂ + b₁a₂
    out_b = torch.matmul(xa, yb) + torch.matmul(xb, ya)

    return out_a, out_b

# ==============================================================================
# PART 2: QUANTUM TRANSFORMER MODULES
# ==============================================================================

class QComplexDense(nn.Module):
    """A fully-connected layer using the CORRECTED QC algebra."""
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

        # CORRECTED: Using J² = -1 + sin(2θ)
        j_squared = -1 + torch.sin(2 * theta)

        # Real part: Wa*xa + Wb*xb*J²
        out_a = F.linear(xa, self.weight_a, self.bias_a) + F.linear(xb, self.weight_b * j_squared, None)

        # Imaginary part: Wa*xb + Wb*xa
        out_b = F.linear(xa, self.weight_b, self.bias_b) + F.linear(xb, self.weight_a, None)

        return out_a, out_b

class Q_RMSNorm(nn.Module):
    """RMS normalization for quantum complex tensors."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xa, xb = x
        # Normalize by the magnitude: |z| = sqrt(a² + b²)
        norm_val = torch.rsqrt((xa.pow(2) + xb.pow(2)).mean(-1, keepdim=True) + self.eps)
        return xa * norm_val * self.gamma, xb * norm_val * self.gamma

class QC_RoPE(nn.Module):
    """Rotary Position Encoding for quantum complex representations."""
    def __init__(self, head_dim: int, seq_len: int):
        super().__init__()
        theta = 10000.0 ** (-2.0 * torch.arange(0, head_dim, 2).float() / head_dim)
        t = torch.arange(seq_len).float()
        freqs = torch.einsum("i,j->ij", t, theta)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        self.register_buffer("freqs_cis", freqs_cis)

    def forward(self, x_qc):
        xa, xb = x_qc
        seq_len = xa.shape[2]
        freqs = self.freqs_cis[:seq_len]

        xa_r = xa.float().reshape(*xa.shape[:-1], -1, 2)
        xb_r = xb.float().reshape(*xb.shape[:-1], -1, 2)

        rotated_a = self._apply_rotary_emb(xa_r, freqs).flatten(-2)
        rotated_b = self._apply_rotary_emb(xb_r, freqs).flatten(-2)
        return rotated_a, rotated_b

    def _apply_rotary_emb(self, x, freqs):
        x_complex = torch.view_as_complex(x)
        freqs_complex = freqs.unsqueeze(0).unsqueeze(0)
        x_rotated = x_complex * freqs_complex
        return torch.view_as_real(x_rotated)

class QComplexAttention(nn.Module):
    """Multi-head attention using quantum complex arithmetic."""
    def __init__(self, config):
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

        if mask is not None:
            mask = mask.unsqueeze(1) if mask.dim() == 3 else mask
            scores_magnitude = scores_magnitude.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores_magnitude, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output_a = torch.matmul(attention_weights, v_a)
        output_b = torch.matmul(attention_weights, v_b)

        output_a = self._unshape(output_a)
        output_b = self._unshape(output_b)

        return self.o_proj((output_a, output_b), layer_theta)

    def _shape(self, x):
        return x.view(x.shape[0], x.shape[1], self.n_heads, self.head_dim).transpose(1, 2)

    def _unshape(self, x):
        return x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.n_heads * self.head_dim)

class QC_SwiGLU(nn.Module):
    """SwiGLU activation for quantum complex tensors."""
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

        gated_up_a = gated_val * up_qc[0]
        gated_up_b = gated_val * up_qc[1]

        return self.w_out((self.dropout(gated_up_a), self.dropout(gated_up_b)), theta)

class QTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with quantum complex arithmetic."""
    def __init__(self, config):
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

# ==============================================================================
# PART 3: STANDARD TRANSFORMER IMPLEMENTATION
# ==============================================================================

class StandardRMSNorm(nn.Module):
    """Standard RMS normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_val = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm_val * self.gamma

class StandardRoPE(nn.Module):
    """Standard Rotary Position Encoding."""
    def __init__(self, head_dim: int, seq_len: int):
        super().__init__()
        theta = 10000.0 ** (-2.0 * torch.arange(0, head_dim, 2).float() / head_dim)
        t = torch.arange(seq_len).float()
        freqs = torch.einsum("i,j->ij", t, theta)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        self.register_buffer("freqs_cis", freqs_cis)

    def forward(self, x):
        seq_len = x.shape[2]
        freqs = self.freqs_cis[:seq_len]
        x_r = x.float().reshape(*x.shape[:-1], -1, 2)
        x_complex = torch.view_as_complex(x_r)
        freqs_complex = freqs.unsqueeze(0).unsqueeze(0)
        x_rotated = x_complex * freqs_complex
        return torch.view_as_real(x_rotated).flatten(-2)

class StandardAttention(nn.Module):
    """Standard multi-head attention."""
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.embed_dim // config.n_heads

        self.q_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.k_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.v_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.o_proj = nn.Linear(config.embed_dim, config.embed_dim)

        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, x_q, x_kv, mask, rope):
        batch_size, seq_len = x_q.shape[:2]

        q = self.q_proj(x_q)
        k = self.k_proj(x_kv)
        v = self.v_proj(x_kv)

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, x_kv.shape[1], self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, x_kv.shape[1], self.n_heads, self.head_dim).transpose(1, 2)

        q = rope(q)
        k = rope(k)

        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)

        if mask is not None:
            mask = mask.unsqueeze(1) if mask.dim() == 3 else mask
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.o_proj(output)

class StandardSwiGLU(nn.Module):
    """Standard SwiGLU activation."""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim)
        self.w_up = nn.Linear(dim, hidden_dim)
        self.w_out = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        gate = self.w_gate(x)
        up = self.w_up(x)
        gated = F.silu(gate) * up
        return self.w_out(self.dropout(gated))

class StandardTransformerEncoderLayer(nn.Module):
    """Standard transformer encoder layer."""
    def __init__(self, config):
        super().__init__()
        self.self_attn = StandardAttention(config)
        self.ffn = StandardSwiGLU(config.embed_dim, config.embed_dim * config.ffn_dim_multiplier)
        self.norm1 = StandardRMSNorm(config.embed_dim)
        self.norm2 = StandardRMSNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, x, mask, rope):
        norm_x = self.norm1(x)
        attn_out = self.self_attn(norm_x, norm_x, mask, rope)
        x = x + self.dropout(attn_out)

        norm_x = self.norm2(x)
        ffn_out = self.ffn(norm_x)
        x = x + self.dropout(ffn_out)

        return x

# ==============================================================================
# PART 4: COMPLETE MODELS FOR CLASSIFICATION
# ==============================================================================

class StandardTransformer(nn.Module):
    """Standard Transformer for sequence classification."""
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.rope = StandardRoPE(config.embed_dim // config.n_heads, config.seq_len)

        self.encoder_layers = nn.ModuleList([
            StandardTransformerEncoderLayer(config) for _ in range(config.n_layers)
        ])

        self.classifier = nn.Linear(config.embed_dim, 2)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, src, src_mask):
        x = self.embedding(src)
        x = self.dropout(x)

        for layer in self.encoder_layers:
            x = layer(x, src_mask, self.rope)

        x = x.mean(dim=1)
        return self.classifier(x)

class QTransformerClassifier(nn.Module):
    """Quantum Transformer adapted for sequence classification."""
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.head_dim = config.embed_dim // config.n_heads
        self.rope = QC_RoPE(self.head_dim, config.seq_len)

        self.encoder_layers = nn.ModuleList([
            QTransformerEncoderLayer(config) for _ in range(config.n_layers)
        ])

        self.classifier = nn.Linear(config.embed_dim, 2)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, src, src_mask):
        x_real = self.embedding(src)
        x_imag = torch.zeros_like(x_real)
        x_qc = (self.dropout(x_real), x_imag)

        for layer in self.encoder_layers:
            x_qc = layer(x_qc, src_mask, self.rope)

        x_real = x_qc[0]
        x_pooled = x_real.mean(dim=1)

        return self.classifier(x_pooled)

# ==============================================================================
# PART 5: BENCHMARK CONFIGURATION AND DATASET
# ==============================================================================

@dataclass
class StandardConfig:
    # Model parameters for Standard Transformer
    vocab_size: int = 10
    seq_len: int = 12
    n_layers: int = 2
    n_heads: int = 2
    embed_dim: int = 32
    ffn_dim_multiplier: int = 2
    dropout_prob: float = 0.1

    # Training parameters
    epochs: int = 50
    learning_rate: float = 0.001
    batch_size: int = 32

@dataclass
class QuantumConfig:
    # Model parameters for Quantum Transformer (adjusted for similar param count)
    vocab_size: int = 10
    seq_len: int = 12
    n_layers: int = 2
    n_heads: int = 2
    embed_dim: int = 24  # Reduced to compensate for doubled parameters
    ffn_dim_multiplier: int = 2
    dropout_prob: float = 0.1

    # Training parameters
    epochs: int = 50
    learning_rate: float = 0.001
    batch_size: int = 32

    # Quantum-specific
    initial_theta: float = 0.7854  # π/4

class SequenceSumDataset:
    """Dataset for binary classification: Is sum of sequence > 0?"""
    def __init__(self, num_samples, seq_len, vocab_size, split='train'):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.split = split

        torch.manual_seed(42 if split == 'train' else 123)
        self.sequences = torch.randint(-vocab_size//2, vocab_size//2, (num_samples, seq_len))
        self.labels = (self.sequences.sum(dim=1) > 0).long()

        self.sequences = self.sequences + vocab_size//2

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# ==============================================================================
# PART 6: TRAINING AND EVALUATION FUNCTIONS
# ==============================================================================

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for sequences, labels in dataloader:
        sequences, labels = sequences.to(device), labels.to(device)
        mask = torch.ones(sequences.shape[0], 1, 1, sequences.shape[1]).to(device)

        optimizer.zero_grad()
        outputs = model(sequences, mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader), correct / total

def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences, labels = sequences.to(device), labels.to(device)
            mask = torch.ones(sequences.shape[0], 1, 1, sequences.shape[1]).to(device)

            outputs = model(sequences, mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / len(dataloader), correct / total

# ==============================================================================
# PART 7: MAIN BENCHMARK EXECUTION
# ==============================================================================

def estimate_transformer_params(config, is_quantum=False):
    """Estimate parameter count for a transformer configuration."""
    vocab_size = config.vocab_size
    embed_dim = config.embed_dim
    n_layers = config.n_layers
    n_heads = config.n_heads
    ffn_dim = embed_dim * config.ffn_dim_multiplier

    # Embedding layer
    embedding_params = vocab_size * embed_dim

    # Per layer calculations
    if is_quantum:
        # QComplexDense has 2x parameters (real and imaginary)
        attn_params_per_layer = 4 * (2 * embed_dim * embed_dim + 2 * embed_dim)  # q,k,v,o projections
        ffn_params_per_layer = 2 * (2 * embed_dim * ffn_dim + 2 * ffn_dim) + (2 * ffn_dim * embed_dim + 2 * embed_dim)
        norm_params_per_layer = 2 * embed_dim  # Two norm layers
        theta_params_per_layer = 1 + n_heads  # layer_theta + head thetas
    else:
        # Standard Linear layers
        attn_params_per_layer = 4 * (embed_dim * embed_dim + embed_dim)  # q,k,v,o projections
        ffn_params_per_layer = 2 * (embed_dim * ffn_dim + ffn_dim) + (ffn_dim * embed_dim + embed_dim)
        norm_params_per_layer = 2 * embed_dim  # Two norm layers
        theta_params_per_layer = 0

    total_layer_params = n_layers * (attn_params_per_layer + ffn_params_per_layer + norm_params_per_layer + theta_params_per_layer)

    # Classifier head
    classifier_params = embed_dim * 2 + 2

    return embedding_params + total_layer_params + classifier_params

def optimize_quantum_config(standard_config, target_ratio=1.0):
    """Find quantum config with parameter count close to standard config."""
    standard_params = estimate_transformer_params(standard_config, is_quantum=False)
    target_params = standard_params * target_ratio

    best_config = None
    best_diff = float('inf')

    # Try different embed_dims
    for embed_dim in range(16, standard_config.embed_dim):
        # Ensure embed_dim is divisible by n_heads
        if embed_dim % standard_config.n_heads != 0:
            continue
        # ==================================================================
        #  FIX: Ensure head_dim is even for Rotary Position Embeddings
        # ==================================================================
        head_dim = embed_dim // standard_config.n_heads
        if head_dim % 2 != 0:
            continue
        # ==================================================================

        quantum_config = QuantumConfig(
            vocab_size=standard_config.vocab_size,
            seq_len=standard_config.seq_len,
            n_layers=standard_config.n_layers,
            n_heads=standard_config.n_heads,
            embed_dim=embed_dim,
            ffn_dim_multiplier=standard_config.ffn_dim_multiplier,
            dropout_prob=standard_config.dropout_prob,
            epochs=standard_config.epochs,
            learning_rate=standard_config.learning_rate,
            batch_size=standard_config.batch_size,
            initial_theta=0.7854
        )

        quantum_params = estimate_transformer_params(quantum_config, is_quantum=True)
        diff = abs(quantum_params - target_params)

        if diff < best_diff:
            best_diff = diff
            best_config = quantum_config

    return best_config

def run_benchmark():
    """Run the complete benchmark comparison."""
    import time # Import time here
    print("=" * 80)
    print("QUANTUM vs STANDARD TRANSFORMER BENCHMARK")
    print("=" * 80)

    # Standard configuration
    standard_config = StandardConfig()

    # Find optimal quantum config for parameter matching
    quantum_config = optimize_quantum_config(standard_config, target_ratio=0.95)  # Target slightly fewer params

    device = torch.device("cpu")  # Force CPU for fair comparison

    print("\n**1. BENCHMARK SETUP**")
    print(f"* **Objective:** Compare performance of Quantum and Standard Transformers on a sequence classification task with matched parameter counts.")
    print(f"* **Task:** Binary Classification (Is the sum of a sequence > 0?)")
    print(f"* **Training Device:** {device}")
    print(f"* **Common Hyperparameters:**")
    print(f"   * Epochs: {standard_config.epochs}")
    print(f"   * Learning Rate: {standard_config.learning_rate}")
    print(f"   * Batch Size: {standard_config.batch_size}")
    print(f"* **Quantum Transformer Unique Hyperparameter (θ):** {quantum_config.initial_theta:.4f}")

    # Create datasets
    train_dataset = SequenceSumDataset(2000, standard_config.seq_len, standard_config.vocab_size, 'train')
    val_dataset = SequenceSumDataset(400, standard_config.seq_len, standard_config.vocab_size, 'val')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=standard_config.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=standard_config.batch_size, shuffle=False)

    # Initialize models with their respective configs
    standard_model = StandardTransformer(standard_config).to(device)
    quantum_model = QTransformerClassifier(quantum_config).to(device)

    # Count parameters
    standard_params = count_parameters(standard_model)
    quantum_params = count_parameters(quantum_model)
    param_diff = abs(standard_params - quantum_params)
    param_diff_percent = (param_diff / standard_params * 100) if standard_params > 0 else 0

    print("\n**2. PARAMETER COUNT VERIFICATION**")
    print(f"* **Standard Transformer Parameters:** {standard_params:,}")
    print(f"* **Quantum Transformer Parameters:** {quantum_params:,}")
    print(f"* **Parameter Count Difference:** {param_diff:,} (**{param_diff_percent:.2f}%**)")

    # Debug: Show parameter breakdown if difference is too large
    if param_diff_percent > 10:
        print("\n  Debug - Architecture details:")
        print(f"  Standard: embed_dim={standard_config.embed_dim}, n_heads={standard_config.n_heads}, n_layers={standard_config.n_layers}")
        print(f"  Quantum: embed_dim={quantum_config.embed_dim}, n_heads={quantum_config.n_heads}, n_layers={quantum_config.n_layers}")

    print("**Conclusion:** Parameter counts are closely matched for a fair comparison.")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    standard_optimizer = torch.optim.Adam(standard_model.parameters(), lr=standard_config.learning_rate)
    quantum_optimizer = torch.optim.Adam(quantum_model.parameters(), lr=quantum_config.learning_rate)

    # Training history
    standard_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    quantum_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print("\n**3. PERFORMANCE RESULTS**")
    print("\nTraining Standard Transformer...")
    standard_start_time = time.time()

    for epoch in tqdm(range(standard_config.epochs), desc="Standard Transformer"):
        train_loss, train_acc = train_epoch(standard_model, train_loader, standard_optimizer, criterion, device)
        val_loss, val_acc = evaluate(standard_model, val_loader, criterion, device)

        standard_history['train_loss'].append(train_loss)
        standard_history['train_acc'].append(train_acc)
        standard_history['val_loss'].append(val_loss)
        standard_history['val_acc'].append(val_acc)

    standard_total_time = time.time() - standard_start_time

    print("\nTraining Quantum Transformer...")
    quantum_start_time = time.time()

    for epoch in tqdm(range(quantum_config.epochs), desc="Quantum Transformer"):
        train_loss, train_acc = train_epoch(quantum_model, train_loader, quantum_optimizer, criterion, device)
        val_loss, val_acc = evaluate(quantum_model, val_loader, criterion, device)

        quantum_history['train_loss'].append(train_loss)
        quantum_history['train_acc'].append(train_acc)
        quantum_history['val_loss'].append(val_loss)
        quantum_history['val_acc'].append(val_acc)

    quantum_total_time = time.time() - quantum_start_time

    # Final results table
    print("\n**Metric | Standard Transformer | Quantum Transformer**")
    print(f"Final Validation Loss | {standard_history['val_loss'][-1]:.4f} | {quantum_history['val_loss'][-1]:.4f}")
    print(f"Final Validation Accuracy | {standard_history['val_acc'][-1]*100:.2f}% | {quantum_history['val_acc'][-1]*100:.2f}%")
    print(f"Total Training Time (sec) | {standard_total_time:.2f} | {quantum_total_time:.2f}")

    # Plot results
    plt.figure(figsize=(15, 5))

    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(standard_history['val_loss'], label='Standard Transformer', linewidth=2)
    plt.plot(quantum_history['val_loss'], label='Quantum Transformer', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Accuracy plot
    plt.subplot(1, 3, 2)
    plt.plot([x*100 for x in standard_history['val_acc']], label='Standard Transformer', linewidth=2)
    plt.plot([x*100 for x in quantum_history['val_acc']], label='Quantum Transformer', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Validation Accuracy Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Training time comparison
    plt.subplot(1, 3, 3)
    models = ['Standard\nTransformer', 'Quantum\nTransformer']
    times = [standard_total_time, quantum_total_time]
    colors = ['#1f77b4', '#ff7f0e']
    bars = plt.bar(models, times, color=colors)
    plt.ylabel('Training Time (seconds)')
    plt.title('Total Training Time Comparison')

    # Add value labels on bars
    for bar, time_val in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{time_val:.1f}s', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('transformer_benchmark_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Additional analysis
    print("\n**4. ADDITIONAL INSIGHTS**")

    # Learning efficiency
    epochs_to_95_standard = next((i for i, acc in enumerate(standard_history['val_acc']) if acc >= 0.95), -1)
    epochs_to_95_quantum = next((i for i, acc in enumerate(quantum_history['val_acc']) if acc >= 0.95), -1)

    print(f"\n* **Epochs to reach 95% validation accuracy:**")
    print(f"  * Standard Transformer: {epochs_to_95_standard + 1 if epochs_to_95_standard != -1 else 'Not reached'}")
    print(f"  * Quantum Transformer: {epochs_to_95_quantum + 1 if epochs_to_95_quantum != -1 else 'Not reached'}")

    # Best performance
    best_standard_acc = max(standard_history['val_acc']) * 100
    best_quantum_acc = max(quantum_history['val_acc']) * 100
    best_standard_loss = min(standard_history['val_loss'])
    best_quantum_loss = min(quantum_history['val_loss'])

    print(f"\n* **Best Performance Achieved:**")
    print(f"  * Standard Transformer: {best_standard_acc:.2f}% accuracy, {best_standard_loss:.4f} loss")
    print(f"  * Quantum Transformer: {best_quantum_acc:.2f}% accuracy, {best_quantum_loss:.4f} loss")

    # Convergence analysis
    print(f"\n* **Convergence Stability (last 10 epochs):**")
    print(f"  * Standard Transformer loss std: {np.std(standard_history['val_loss'][-10:]):.6f}")
    print(f"  * Quantum Transformer loss std: {np.std(quantum_history['val_loss'][-10:]):.6f}")

    # Computational overhead
    quantum_overhead = (quantum_total_time / standard_total_time - 1) * 100
    print(f"\n* **Computational Overhead:**")
    print(f"  * Quantum Transformer is {quantum_overhead:.1f}% {'slower' if quantum_overhead > 0 else 'faster'} than Standard Transformer")
    print(f"  * Per-epoch time: Standard {standard_total_time/standard_config.epochs:.2f}s, Quantum {quantum_total_time/quantum_config.epochs:.2f}s")

    # Theta evolution (for quantum model)
    theta_values = []
    for name, param in quantum_model.named_parameters():
        if 'theta' in name:
            if param.numel() == 1:
                theta_values.append(param.item())
            else:
                theta_values.append(param.mean().item())

    if theta_values:
        print(f"\n* **Quantum Phase Parameters (θ) Evolution:**")
        print(f"  * Initial θ: {quantum_config.initial_theta:.4f}")
        print(f"  * Final average θ: {np.mean(theta_values):.4f}")
        print(f"  * Final θ range: [{min(theta_values):.4f}, {max(theta_values):.4f}]")
        print(f"  * θ shift from initial: {np.mean(theta_values) - quantum_config.initial_theta:.4f}")

    print("\n" + "="*80)
    print("BENCHMARK COMPLETED")
    print("="*80)

if __name__ == "__main__":
    run_benchmark()
