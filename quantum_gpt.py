import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# ==============================================================================
# PART 1: CORE QUANTUMCOMPLEX TENSOR ARITHMETIC
#
# These functions replace the Python class methods and operate on entire
# PyTorch tensors, making them fast and differentiable.
# ==============================================================================

def qc_mul(x, y, theta):
    """
    Performs QuantumComplex multiplication on two tuples of tensors.
    x = (xa, xb), y = (ya, yb)
    """
    xa, xb = x
    ya, yb = y
    
    j_squared = -torch.cos(2 * theta)
    
    # (a1*a2 + b1*b2*J^2) + (a1*b2 + b1*a2)*J
    out_a = xa * ya + xb * yb * j_squared
    out_b = xa * yb + xb * ya
    return (out_a, out_b)

def qc_add(x, y):
    """Performs QuantumComplex addition on two tuples of tensors."""
    return (x[0] + y[0], x[1] + y[1])

def qc_matmul(x, W):
    """
    Performs matrix multiplication (dot product) for QC tensors.
    x is a batch of vectors (B, D_in), W is a weight matrix (D_out, D_in)
    """
    xa, xb = x
    Wa, Wb = W
    theta = Wa.device.type # A trick to pass theta without extra args if needed, but we pass it
    
    # This is the expansion of (x @ W.T) using QC multiplication rules
    # Note: We use W.T for convention, so shapes match torch.nn.Linear
    out_a = xa @ Wa.T - xb @ (Wb * torch.cos(2 * Wb.new_ones(1))).T
    out_b = xa @ Wb.T + xb @ Wa.T
    return (out_a, out_b)


# ==============================================================================
# PART 2: PROFESSIONAL PYTORCH NEURAL NETWORK MODULES
# ==============================================================================

class QComplexDense(nn.Module):
    """A professional fully-connected layer with Xavier initialization."""
    def __init__(self, input_dim, output_dim, theta):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.theta = nn.Parameter(torch.tensor(theta), requires_grad=False)

        # We need two sets of weights and biases for the 'a' and 'b' components
        self.weight_a = nn.Parameter(torch.empty(output_dim, input_dim))
        self.weight_b = nn.Parameter(torch.empty(output_dim, input_dim))
        self.bias_a = nn.Parameter(torch.empty(output_dim))
        self.bias_b = nn.Parameter(torch.empty(output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        # Apply Xavier uniform initialization to both components
        nn.init.xavier_uniform_(self.weight_a)
        nn.init.xavier_uniform_(self.weight_b)
        nn.init.zeros_(self.bias_a)
        nn.init.zeros_(self.bias_b)
        
    def forward(self, x): # x is a tuple (xa, xb)
        xa, xb = x
        j_squared = -torch.cos(2 * self.theta)

        # Linear transformation using expanded QC matmul
        out_a = F.linear(xa, self.weight_a, self.bias_a) + F.linear(xb, self.weight_b * j_squared, None)
        out_b = F.linear(xa, self.weight_b, self.bias_b) + F.linear(xb, self.weight_a, None)
        
        return (out_a, out_b)

class QComplexLayerNorm(nn.Module):
    """Performs Layer Normalization on 'a' and 'b' components separately."""
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        # Learnable affine parameters for both 'a' and 'b' components
        self.gamma_a = nn.Parameter(torch.ones(self.normalized_shape))
        self.beta_a = nn.Parameter(torch.zeros(self.normalized_shape))
        self.gamma_b = nn.Parameter(torch.ones(self.normalized_shape))
        self.beta_b = nn.Parameter(torch.zeros(self.normalized_shape))
        
    def forward(self, x):
        xa, xb = x
        # Normalize 'a' and 'b' components along the last dimension
        xa_norm = F.layer_norm(xa, self.normalized_shape, self.gamma_a, self.beta_a, self.eps)
        xb_norm = F.layer_norm(xb, self.normalized_shape, self.gamma_b, self.beta_b, self.eps)
        return (xa_norm, xb_norm)

class QComplexSelfAttention(nn.Module):
    """The self-attention mechanism implemented in PyTorch."""
    def __init__(self, embed_dim, theta):
        super().__init__()
        self.embed_dim = embed_dim
        self.theta = nn.Parameter(torch.tensor(theta), requires_grad=False)
        self.d_k = embed_dim
        
        self.q_proj = QComplexDense(embed_dim, embed_dim, theta)
        self.k_proj = QComplexDense(embed_dim, embed_dim, theta)
        self.v_proj = QComplexDense(embed_dim, embed_dim, theta)
        self.out_proj = QComplexDense(embed_dim, embed_dim, theta)
        
    def forward(self, x):
        xa, xb = x
        batch_size, seq_len, _ = xa.size()
        
        # Project to Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Calculate attention scores using QC matmul
        scores_a, scores_b = qc_matmul(Q, (K[0].transpose(-2, -1), K[1].transpose(-2, -1)))
        
        # Take magnitude for softmax. This is the bridge to real-valued attention.
        scores_real = torch.sqrt(scores_a**2 + scores_b**2)
        
        # Apply softmax
        attention_weights = F.softmax(scores_real / math.sqrt(self.d_k), dim=-1)
        
        # Aggregate the Value vectors
        # (B, L, L) @ (B, L, D) -> (B, L, D)
        # Since attention_weights is real, this is a simple scalar multiplication
        out_a = torch.bmm(attention_weights, V[0])
        out_b = torch.bmm(attention_weights, V[1])

        # Final output projection
        return self.out_proj((out_a, out_b))

class QComplexFeedForward(nn.Module):
    """The Feed-Forward Network block of the Transformer."""
    def __init__(self, embed_dim, ffn_dim, theta):
        super().__init__()
        self.layer1 = QComplexDense(embed_dim, ffn_dim, theta)
        self.layer2 = QComplexDense(ffn_dim, embed_dim, theta)

    def forward(self, x):
        # Pass through first layer
        l1_out_a, l1_out_b = self.layer1(x)
        # Apply GELU activation to both components
        act_a = F.gelu(l1_out_a)
        act_b = F.gelu(l1_out_b)
        # Pass through second layer
        return self.layer2((act_a, act_b))

# ==============================================================================
# PART 3: ASSEMBLING THE FULL QUANTUM TRANSFORMER MODEL
# ==============================================================================

class QComplexTransformerEncoderLayer(nn.Module):
    """A single, complete, and professional Transformer encoder layer."""
    def __init__(self, embed_dim, ffn_dim, theta):
        super().__init__()
        self.attention = QComplexSelfAttention(embed_dim, theta)
        self.ffn = QComplexFeedForward(embed_dim, ffn_dim, theta)
        self.norm1 = QComplexLayerNorm(embed_dim)
        self.norm2 = QComplexLayerNorm(embed_dim)

    def forward(self, x):
        # Attention -> Add & Norm
        attn_out = self.attention(x)
        x = self.norm1(qc_add(x, attn_out))
        # FFN -> Add & Norm
        ffn_out = self.ffn(x)
        x = self.norm2(qc_add(x, ffn_out))
        return x

class QuantumTransformer(nn.Module):
    """The complete Quantum Transformer model implemented in PyTorch."""
    def __init__(self, seq_len, embed_dim, ffn_dim, n_layers, theta=0.0):
        super().__init__()
        self.theta = theta
        # Embedding layer: map a real float to a QC vector (a, 0)
        self.embedding_a = nn.Linear(1, embed_dim)
        
        self.encoder_layers = nn.ModuleList(
            [QComplexTransformerEncoderLayer(embed_dim, ffn_dim, theta) for _ in range(n_layers)]
        )
        # Final output layer to get a single real number prediction
        self.output_layer = nn.Linear(embed_dim * seq_len, 1)

    def forward(self, x): # x has shape (batch_size, seq_len)
        # Reshape for embedding layer
        x = x.unsqueeze(-1) # (B, L) -> (B, L, 1)
        
        # Embed the real input into the 'a' component, 'b' is zero
        xa = self.embedding_a(x)
        xb = torch.zeros_like(xa)
        qc_x = (xa, xb)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            qc_x = layer.forward(qc_x)
            
        # Flatten the 'a' component for the final real-valued linear layer
        out_a, _ = qc_x
        out_a_flat = out_a.view(out_a.size(0), -1)
        
        # Get final prediction
        return self.output_layer(out_a_flat).squeeze(-1)

# ==============================================================================
# PART 4: TRAINING AND DEMONSTRATION
# ==============================================================================

if __name__ == '__main__':
    # --- Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Model Hyperparameters ---
    SEQ_LEN = 4
    EMBED_DIM = 32
    FFN_DIM = 64
    N_LAYERS = 2
    THETA = np.pi / 4.0
    
    # --- Training Hyperparameters ---
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    EPOCHS = 100

    # --- Generate Data ---
    def generate_data(n_samples, seq_len):
        X = torch.randn(n_samples, seq_len)
        y = torch.mean(X, axis=1) # Task: predict the mean
        return X, y

    X_train, y_train = generate_data(1000, SEQ_LEN)
    X_val, y_val = generate_data(200, SEQ_LEN)
    
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- Initialize Model, Loss, and Optimizer ---
    model = QuantumTransformer(SEQ_LEN, EMBED_DIM, FFN_DIM, N_LAYERS, THETA).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print("\n" + "="*60)
    print("      Training PyTorch-based Quantum Transformer Network")
    print("="*60)
    
    # --- Training Loop ---
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}")

    print("-"*60)
    print("\nTraining Finished.")
    
    # --- Evaluate the Model ---
    model.eval()
    with torch.no_grad():
        x_test = torch.tensor([[0.1, 0.2, 0.3, 0.4], [-1.0, 1.0, -2.0, 2.0]]).to(device)
        y_true = torch.mean(x_test, axis=1)
        y_pred = model(x_test)

        print("\n--- Evaluating the Final Model ---")
        for i in range(len(x_test)):
            print(f"\nInput: {x_test[i].cpu().numpy()}")
            print(f"  -> True Mean: {y_true[i]:.4f}")
            print(f"  -> Model Prediction: {y_pred[i]:.4f}")
    print("="*60)
