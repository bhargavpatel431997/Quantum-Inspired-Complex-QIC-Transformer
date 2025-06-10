import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============================================================================
# 1. The Core Innovation: Quantum Geometric Embedding Layer
#    With the CORRECT definition for Êk as per the paper.
# ==============================================================================

class QuantumGeometricEmbedding(nn.Module):
    """
    Implements the Generalized Emergence Operator from the paper (Sec 2.2).
    This layer processes token embeddings by applying the geometric operators
    B(β)Êj and H(β)Êk, creating a rich, structured representation.

    - B(β) = e^(βJ) = cos(β)I + sin(β)J
    - H(β) = -e^(-βJ) = -cos(β)I + sin(β)J
    - Êj = [[0, -1], [1, 0]]  (as per Eq. 30)
    - Êk = [[0, 1], [-1, 0]] (as per Eq. 30, and corrected here)

    The properties Êj^2 = -I and Êk^2 = -I are now correctly satisfied.
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even for QuantumGeometricEmbedding, but got {d_model}")
        self.num_pairs = d_model // 2

        self.beta_projection = nn.Linear(d_model, 1)
        self.output_projection = nn.Linear(2 * d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # CORRECT definitions for the constant geometric matrices Êj and Êk from Eq. 30.
        Ej = torch.tensor([[0., -1.], [1., 0.]])
        Ek = torch.tensor([[0., 1.], [-1., 0.]]) # CORRECTED DEFINITION
        self.register_buffer('Ej', Ej)
        self.register_buffer('Ek', Ek)

        # Verification step during initialization
        assert torch.allclose(torch.matmul(self.Ej, self.Ej), -torch.eye(2)), "Ej^2 != -I"
        assert torch.allclose(torch.matmul(self.Ek, self.Ek), -torch.eye(2)), "Ek^2 != -I"

    def forward(self, x):
        """
        x: Input tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Calculate the data-dependent rotation angle β
        beta = self.beta_projection(x)
        cos_beta = torch.cos(beta).view(batch_size * seq_len, 1, 1)
        sin_beta = torch.sin(beta).view(batch_size * seq_len, 1, 1)

        # 2. Construct the B(β) and H(β) operator matrices
        I = torch.eye(2, device=x.device).unsqueeze(0)
        J = self.Ej.unsqueeze(0)
        
        B_matrix = cos_beta * I + sin_beta * J
        H_matrix = -cos_beta * I + sin_beta * J

        # 3. Form the complete geometric operators: Op1 = B(β)Êj and Op2 = H(β)Êk
        op1 = torch.matmul(B_matrix, self.Ej)
        op2 = torch.matmul(H_matrix, self.Ek)

        # Prepare input for matrix multiplication
        x_vectors = x.view(batch_size * seq_len, self.num_pairs, 2, 1)

        # 4. Apply the operators to the input vectors
        x_transformed_1 = torch.matmul(op1.unsqueeze(1), x_vectors).squeeze(-1)
        x_transformed_2 = torch.matmul(op2.unsqueeze(1), x_vectors).squeeze(-1)

        # Flatten the pairs back into a vector
        features_1 = x_transformed_1.view(batch_size * seq_len, self.d_model)
        features_2 = x_transformed_2.view(batch_size * seq_len, self.d_model)

        # 5. Combine features and project back to d_model
        combined_features = torch.cat([features_1, features_2], dim=-1)
        projected_output = self.output_projection(combined_features)
        
        output = projected_output.view(batch_size, seq_len, self.d_model)
        
        # Apply dropout, layer norm, and a residual connection for stability
        return self.dropout(self.layer_norm(output + x))


# ==============================================================================
# 2. Standard Transformer Components (Unchanged)
# ==============================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model, self.n_head, self.d_head = d_model, n_head, d_model // n_head
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.qkv_proj(x).split(self.d_model, dim=2)
        q, k, v = [t.view(B, T, self.n_head, self.d_head).transpose(1, 2) for t in (q, k, v)]
        
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

# ==============================================================================
# 3. The QuantumGPT Model (Uses the corrected embedding layer)
# ==============================================================================

class QuantumGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layer, n_head, d_ff, dropout, max_seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.quantum_embedding = QuantumGeometricEmbedding(d_model=d_model, dropout=dropout)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_head, d_ff, dropout) for _ in range(n_layer)
        ])
        self.final_ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight
        
    def forward(self, idx, targets=None):
        tok_emb = self.token_embedding(idx)
        x = self.quantum_embedding(tok_emb)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.final_ln(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss
        
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ==============================================================================
# 4. Example Usage
# ==============================================================================
if __name__ == '__main__':
    # --- Model Hyperparameters ---
    config = {
        'vocab_size': 1000,
        'd_model': 128,
        'n_layer': 4,
        'n_head': 4,
        'd_ff': 128 * 4,
        'dropout': 0.1,
        'max_seq_len': 256,
    }

    # --- Instantiate the Model ---
    model = QuantumGPT(**config)
    print(f"QuantumGPT model initialized with {model.count_parameters():,} parameters.")
    print("Geometric matrices Êj and Êk properties verified at initialization.")
    
    # --- Create Dummy Data and Run a Forward Pass ---
    batch_size = 4
    seq_len = 64
    dummy_input = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    dummy_targets = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    
    print(f"\nRunning a forward pass with input shape: {dummy_input.shape}")
    
    logits, loss = model(dummy_input, targets=dummy_targets)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Calculated loss: {loss.item():.4f}")
    
    assert logits.shape == (batch_size, seq_len, config['vocab_size'])
    assert loss is not None
    
    print("\n✅ Final corrected QuantumGPT ran successfully!")
