import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class QuantumMatrix:
    """Quantum matrix operations using J(θ) structure"""
    
    def __init__(self):
        self.I = np.eye(2)
        self.J_plus = np.array([[0, -1], [1, 0]])
        self.J_minus = np.array([[0, 1], [-1, 0]])
    
    def J_theta(self, theta):
        """Quantum superposition of J matrices"""
        return np.cos(theta) * self.J_plus + np.sin(theta) * self.J_minus
    
    def quantum_matrix(self, real_part, imag_part, theta):
        """Create quantum state matrix Z = real*I + imag*J(θ)"""
        J = self.J_theta(theta)
        return real_part * self.I + imag_part * J
    
    def quantum_activation(self, Z, scale=0.5):
        """Quantum activation using matrix exponential"""
        try:
            exp_Z = self.matrix_exp(Z, scale)
            exp_neg_Z = self.matrix_exp(-Z, scale)
            return exp_Z @ np.linalg.inv(exp_Z + exp_neg_Z + 1e-8 * self.I)
        except:
            # Fallback to classical activation
            return np.tanh(Z)
    
    def matrix_exp(self, Z, scale=1.0):
        """Compute matrix exponential"""
        eigenvals, eigenvecs = np.linalg.eig(scale * Z)
        return np.real(eigenvecs @ np.diag(np.exp(eigenvals)) @ eigenvecs.T)


class QuantumAttention(nn.Module):
    """Quantum-enhanced attention mechanism"""
    
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Classical linear transformations
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Quantum parameters
        self.quantum_phases = nn.Parameter(torch.randn(n_heads) * 2 * np.pi)
        self.quantum_weights_real = nn.Parameter(torch.randn(n_heads, self.d_k) * 0.1)
        self.quantum_weights_imag = nn.Parameter(torch.randn(n_heads, self.d_k) * 0.1)
        
        self.dropout = nn.Dropout(dropout)
        self.quantum_matrix = QuantumMatrix()
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, d_model = query.size()
        
        # Classical attention computation
        Q = self.W_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Quantum enhancement
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply quantum transformation to attention scores
        quantum_attention = self.apply_quantum_transformation(attention_scores)
        
        if mask is not None:
            quantum_attention = quantum_attention.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(quantum_attention, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)
        
        # Concatenate heads and apply output projection
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)
        
        return self.W_o(attended_values), attention_weights
    
    def apply_quantum_transformation(self, attention_scores):
        """Apply quantum transformation to attention scores"""
        batch_size, n_heads, seq_len, seq_len = attention_scores.shape
        quantum_enhanced = attention_scores.clone()
        
        for h in range(n_heads):
            theta = self.quantum_phases[h]
            w_real = self.quantum_weights_real[h]
            w_imag = self.quantum_weights_imag[h]
            
            # Apply quantum transformation per head
            for i in range(seq_len):
                for j in range(seq_len):
                    # Create quantum matrix for this attention position
                    real_part = attention_scores[:, h, i, j]
                    imag_part = torch.sum(w_real * w_imag) * torch.sin(theta)
                    
                    # Quantum interference effect
                    quantum_phase = torch.cos(theta + real_part * 0.1)
                    quantum_enhanced[:, h, i, j] = real_part * quantum_phase + imag_part * 0.1
        
        return quantum_enhanced


class QuantumFeedForward(nn.Module):
    """Quantum-enhanced feed-forward network"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Classical layers
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Quantum parameters
        self.quantum_phases = nn.Parameter(torch.randn(d_ff) * 2 * np.pi)
        self.quantum_weights_real = nn.Parameter(torch.randn(d_ff) * 0.1)
        self.quantum_weights_imag = nn.Parameter(torch.randn(d_ff) * 0.1)
        
        self.quantum_matrix = QuantumMatrix()
    
    def forward(self, x):
        # First linear transformation
        hidden = self.linear1(x)
        
        # Apply quantum transformation
        quantum_hidden = self.apply_quantum_transformation(hidden)
        
        # Activation and dropout
        activated = F.relu(quantum_hidden)
        activated = self.dropout(activated)
        
        # Second linear transformation
        output = self.linear2(activated)
        
        return output
    
    def apply_quantum_transformation(self, hidden):
        """Apply quantum transformation to hidden states"""
        batch_size, seq_len, d_ff = hidden.shape
        quantum_enhanced = torch.zeros_like(hidden)
        
        for i in range(d_ff):
            theta = self.quantum_phases[i]
            w_real = self.quantum_weights_real[i]
            w_imag = self.quantum_weights_imag[i]
            
            # Quantum superposition
            real_part = hidden[:, :, i]
            imag_part = w_imag * torch.sin(theta + real_part * 0.1)
            
            # Quantum interference
            quantum_phase = torch.cos(theta) * w_real + torch.sin(theta) * w_imag
            quantum_enhanced[:, :, i] = real_part + quantum_phase * imag_part * 0.1
        
        return quantum_enhanced


class QuantumTransformerLayer(nn.Module):
    """Quantum-enhanced transformer layer"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.quantum_attention = QuantumAttention(d_model, n_heads, dropout)
        self.quantum_feedforward = QuantumFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Quantum phase for the entire layer
        self.layer_quantum_phase = nn.Parameter(torch.randn(1) * 2 * np.pi)
    
    def forward(self, src, src_mask=None):
        # Quantum attention with residual connection
        attn_output, attention_weights = self.quantum_attention(src, src, src, src_mask)
        src = src + self.dropout(attn_output)
        src = self.norm1(src)
        
        # Quantum feed-forward with residual connection
        ff_output = self.quantum_feedforward(src)
        
        # Apply layer-level quantum phase
        quantum_phase = torch.cos(self.layer_quantum_phase)
        ff_output = ff_output * quantum_phase
        
        src = src + self.dropout(ff_output)
        src = self.norm2(src)
        
        return src, attention_weights


class QuantumTransformer(nn.Module):
    """Full Quantum-Enhanced Transformer"""
    
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, 
                 d_ff=2048, max_seq_len=1000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Quantum transformer layers
        self.layers = nn.ModuleList([
            QuantumTransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Global quantum parameters
        self.global_quantum_phases = nn.Parameter(torch.randn(n_layers) * 2 * np.pi)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, src_mask=None):
        batch_size, seq_len = src.size()
        
        # Create position indices
        positions = torch.arange(seq_len, device=src.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_emb = self.token_embedding(src)
        pos_emb = self.position_embedding(positions)
        
        # Combine embeddings with quantum phase
        global_phase = torch.mean(torch.cos(self.global_quantum_phases))
        embeddings = (token_emb + pos_emb) * math.sqrt(self.d_model) * global_phase
        embeddings = self.dropout(embeddings)
        
        # Pass through quantum transformer layers
        hidden_states = embeddings
        attention_weights_all = []
        
        for i, layer in enumerate(self.layers):
            # Apply global quantum phase per layer
            layer_phase = torch.cos(self.global_quantum_phases[i])
            hidden_states = hidden_states * layer_phase
            
            hidden_states, attention_weights = layer(hidden_states, src_mask)
            attention_weights_all.append(attention_weights)
        
        # Output projection
        output = self.output_projection(hidden_states)
        
        return output, attention_weights_all
    
    def visualize_quantum_states(self, src):
        """Visualize quantum states in the transformer"""
        with torch.no_grad():
            output, attention_weights = self.forward(src)
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Plot 1: Global quantum phases
            ax = axes[0, 0]
            phases = self.global_quantum_phases.detach().numpy()
            ax.bar(range(len(phases)), phases)
            ax.set_title('Global Quantum Phases by Layer')
            ax.set_xlabel('Layer')
            ax.set_ylabel('Phase (radians)')
            
            # Plot 2: Attention quantum phases for first layer
            ax = axes[0, 1]
            if len(self.layers) > 0:
                attn_phases = self.layers[0].quantum_attention.quantum_phases.detach().numpy()
                ax.bar(range(len(attn_phases)), attn_phases)
                ax.set_title('Attention Head Quantum Phases (Layer 0)')
                ax.set_xlabel('Attention Head')
                ax.set_ylabel('Phase (radians)')
            
            # Plot 3: Feed-forward quantum phases
            ax = axes[0, 2]
            if len(self.layers) > 0:
                ff_phases = self.layers[0].quantum_feedforward.quantum_phases.detach().numpy()
                ax.hist(ff_phases, bins=20, alpha=0.7, color='purple')
                ax.set_title('Feed-Forward Quantum Phase Distribution')
                ax.set_xlabel('Phase (radians)')
                ax.set_ylabel('Count')
            
            # Plot 4: Attention weights heatmap
            ax = axes[1, 0]
            if attention_weights:
                attn_matrix = attention_weights[0][0, 0].detach().numpy()  # First batch, first head
                im = ax.imshow(attn_matrix, cmap='Blues')
                ax.set_title('Quantum Attention Weights (Head 0)')
                ax.set_xlabel('Key Position')
                ax.set_ylabel('Query Position')
                plt.colorbar(im, ax=ax)
            
            # Plot 5: Phase evolution across layers
            ax = axes[1, 1]
            layer_phases = []
            for layer in self.layers:
                mean_phase = torch.mean(layer.quantum_attention.quantum_phases).item()
                layer_phases.append(mean_phase)
            
            ax.plot(layer_phases, 'o-', linewidth=2, markersize=8)
            ax.set_title('Mean Quantum Phase Evolution')
            ax.set_xlabel('Layer')
            ax.set_ylabel('Mean Phase (radians)')
            ax.grid(True, alpha=0.3)
            
            # Plot 6: Quantum interference pattern
            ax = axes[1, 2]
            if src.size(1) > 1:
                # Create interference pattern based on quantum phases
                seq_len = src.size(1)
                interference = np.zeros(seq_len)
                
                for i in range(seq_len):
                    phase_sum = 0
                    for layer in self.layers:
                        for head_phase in layer.quantum_attention.quantum_phases:
                            phase_sum += torch.cos(head_phase + i * 0.1).item()
                    interference[i] = phase_sum / (len(self.layers) * self.layers[0].quantum_attention.n_heads)
                
                ax.plot(interference, 'r-', linewidth=2)
                ax.set_title('Quantum Interference Pattern')
                ax.set_xlabel('Sequence Position')
                ax.set_ylabel('Interference Strength')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    def quantum_analysis(self, src):
        """Analyze quantum properties of the transformer"""
        with torch.no_grad():
            print("QUANTUM TRANSFORMER ANALYSIS")
            print("=" * 50)
            
            # Global phase statistics
            global_phases = self.global_quantum_phases.detach().numpy()
            print(f"\nGlobal Quantum Phases:")
            print(f"  Mean: {np.mean(global_phases):.4f}")
            print(f"  Std:  {np.std(global_phases):.4f}")
            print(f"  Range: [{np.min(global_phases):.4f}, {np.max(global_phases):.4f}]")
            
            # Layer-wise analysis
            print(f"\nLayer-wise Quantum Analysis:")
            for i, layer in enumerate(self.layers):
                attn_phases = layer.quantum_attention.quantum_phases.detach().numpy()
                ff_phases = layer.quantum_feedforward.quantum_phases.detach().numpy()
                
                print(f"  Layer {i}:")
                print(f"    Attention phases - Mean: {np.mean(attn_phases):.4f}, Std: {np.std(attn_phases):.4f}")
                print(f"    FF phases - Mean: {np.mean(ff_phases):.4f}, Std: {np.std(ff_phases):.4f}")
            
            # Quantum coherence measure
            total_coherence = 0
            for layer in self.layers:
                attn_coherence = torch.mean(torch.cos(layer.quantum_attention.quantum_phases)).item()
                ff_coherence = torch.mean(torch.cos(layer.quantum_feedforward.quantum_phases)).item()
                layer_coherence = (attn_coherence + ff_coherence) / 2
                total_coherence += layer_coherence
            
            avg_coherence = total_coherence / len(self.layers)
            print(f"\nQuantum Coherence Measure: {avg_coherence:.4f}")
            print(f"  (Range: [-1, 1], higher = more coherent)")


def demonstrate_quantum_transformer():
    """Demonstrate the quantum transformer"""
    print("QUANTUM TRANSFORMER DEMONSTRATION")
    print("=" * 50)
    
    # Create model
    vocab_size = 1000
    d_model = 256
    n_heads = 8
    n_layers = 4
    
    model = QuantumTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=1024,
        max_seq_len=100
    )
    
    print(f"Created Quantum Transformer:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Model dimension: {d_model}")
    print(f"  Attention heads: {n_heads}")
    print(f"  Layers: {n_layers}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create sample input
    batch_size = 2
    seq_len = 20
    src = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"\nInput shape: {src.shape}")
    
    # Forward pass
    output, attention_weights = model(src)
    print(f"Output shape: {output.shape}")
    print(f"Number of attention weight tensors: {len(attention_weights)}")
    
    # Quantum analysis
    model.quantum_analysis(src)
    
    # Visualize quantum states
    print(f"\nGenerating quantum state visualizations...")
    model.visualize_quantum_states(src)
    
    # Compare with classical transformer behavior
    print(f"\nQuantum vs Classical Comparison:")
    
    # Save original quantum phases
    original_phases = {}
    for i, layer in enumerate(model.layers):
        original_phases[f'layer_{i}_attn'] = layer.quantum_attention.quantum_phases.clone()
        original_phases[f'layer_{i}_ff'] = layer.quantum_feedforward.quantum_phases.clone()
        original_phases[f'layer_{i}_global'] = layer.layer_quantum_phase.clone()
    
    original_global = model.global_quantum_phases.clone()
    
    # Get quantum output
    quantum_output, _ = model(src)
    
    # Set all phases to 0 (classical mode)
    with torch.no_grad():
        model.global_quantum_phases.fill_(0)
        for layer in model.layers:
            layer.quantum_attention.quantum_phases.fill_(0)
            layer.quantum_feedforward.quantum_phases.fill_(0)
            layer.layer_quantum_phase.fill_(0)
    
    # Get classical output
    classical_output, _ = model(src)
    
    # Restore quantum phases
    with torch.no_grad():
        model.global_quantum_phases.copy_(original_global)
        for i, layer in enumerate(model.layers):
            layer.quantum_attention.quantum_phases.copy_(original_phases[f'layer_{i}_attn'])
            layer.quantum_feedforward.quantum_phases.copy_(original_phases[f'layer_{i}_ff'])
            layer.layer_quantum_phase.copy_(original_phases[f'layer_{i}_global'])
    
    # Compare outputs
    output_diff = torch.mean(torch.abs(quantum_output - classical_output)).item()
    print(f"  Mean absolute difference: {output_diff:.6f}")
    print(f"  Quantum enhancement factor: {output_diff / torch.mean(torch.abs(classical_output)).item():.4f}")
    
    # Visualize difference
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(quantum_output[0].detach().numpy().T, aspect='auto', cmap='RdBu')
    plt.title('Quantum Transformer Output')
    plt.xlabel('Sequence Position')
    plt.ylabel('Vocabulary Dimension')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow((quantum_output[0] - classical_output[0]).detach().numpy().T, 
               aspect='auto', cmap='RdBu')
    plt.title('Quantum - Classical Difference')
    plt.xlabel('Sequence Position')
    plt.ylabel('Vocabulary Dimension')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nKey Quantum Features:")
    print(f"  ✓ Quantum phases in attention mechanisms")
    print(f"  ✓ Quantum superposition in feed-forward layers")
    print(f"  ✓ Global quantum coherence across layers")
    print(f"  ✓ Quantum interference patterns in representations")
    print(f"  ✓ Phase evolution during computation")

# Run the demonstration
if __name__ == "__main__":
    demonstrate_quantum_transformer()
