"""
PARAMETER-EFFICIENT QIC ON REAL DATASETS
=========================================
Tests the actual QIC implementation with learnable complex algebra
on IMDB and AG News datasets to verify parameter reduction AND performance.

Requirements:
pip install torch datasets transformers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import os
import pickle

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("ERROR: Please install datasets: pip install datasets")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Unified configuration for experiments."""
    vocab_size: int = 10000
    
    # Standard model dimensions (for ~2.5M params)
    std_embed_dim: int = 128
    std_hidden_dim: int = 512
    
    # QIC dimensions (50% for parameter reduction)
    qic_embed_dim: int = 64
    qic_hidden_dim: int = 256
    
    # Architecture
    n_heads: int = 8
    n_layers: int = 6
    max_seq_len: int = 256
    num_classes: int = 2  # Will be updated for AG News
    
    # QIC specific
    theta_init: float = 0.0  # Start near standard complex
    
    # Training
    batch_size: int = 32
    learning_rate: float = 2e-3
    epochs: int = 5
    dropout: float = 0.1
    gradient_clip: float = 1.0


# ============================================================================
# QIC ALGEBRA CORE
# ============================================================================

class QICAlgebra:
    """Core QIC algebra with J(theta)^2 = -1 + sin(2*theta)."""
    
    @staticmethod
    def j_squared(theta: torch.Tensor) -> torch.Tensor:
        return -1 + torch.sin(2 * theta)
    
    @staticmethod
    def magnitude(real: torch.Tensor, imag: torch.Tensor, 
                  theta: torch.Tensor) -> torch.Tensor:
        norm_factor = 2 - torch.sin(2 * theta)
        return torch.sqrt(real**2 + imag**2 * norm_factor + 1e-8)


# ============================================================================
# EFFICIENT QIC LAYERS
# ============================================================================

class QICLinear(nn.Module):
    """QIC Linear layer with learnable complex algebra."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        
        scale = 1.0 / math.sqrt(in_features)
        self.W_real = nn.Parameter(torch.randn(out_features, in_features) * scale)
        self.W_imag = nn.Parameter(torch.randn(out_features, in_features) * scale * 0.1)
        
        if bias:
            self.b_real = nn.Parameter(torch.zeros(out_features))
            self.b_imag = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('b_real', None)
            self.register_parameter('b_imag', None)
    
    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor, 
                theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        j_squared = QICAlgebra.j_squared(theta)
        
        y_real = F.linear(x_real, self.W_real) + F.linear(x_imag, self.W_imag) * j_squared
        y_imag = F.linear(x_imag, self.W_real) + F.linear(x_real, self.W_imag)
        
        if self.b_real is not None:
            y_real = y_real + self.b_real
            y_imag = y_imag + self.b_imag
        
        return y_real, y_imag


class QICAttention(nn.Module):
    """Efficient QIC attention with parameter sharing."""
    
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % n_heads == 0
        
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        
        # Share QK projection for efficiency
        self.qk_proj = QICLinear(embed_dim, embed_dim)
        self.v_proj = QICLinear(embed_dim, embed_dim)
        self.out_proj = QICLinear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor, 
                theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x_real.shape
        
        # Project (with QK sharing)
        qk_real, qk_imag = self.qk_proj(x_real, x_imag, theta)
        v_real, v_imag = self.v_proj(x_real, x_imag, theta)
        
        # Reshape for heads
        def reshape_heads(x):
            return x.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        q_real, q_imag = reshape_heads(qk_real), reshape_heads(qk_imag)
        k_real, k_imag = reshape_heads(qk_real), reshape_heads(qk_imag)  # Shared
        v_real, v_imag = reshape_heads(v_real), reshape_heads(v_imag)
        
        # Attention scores (use real part for simplicity)
        scores = torch.matmul(q_real, k_real.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out_real = torch.matmul(attn, v_real)
        out_imag = torch.matmul(attn, v_imag)
        
        # Reshape back
        out_real = out_real.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        out_imag = out_imag.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        return self.out_proj(out_real, out_imag, theta)


class QICLayer(nn.Module):
    """Single QIC transformer layer."""
    
    def __init__(self, embed_dim: int, hidden_dim: int, n_heads: int, dropout: float):
        super().__init__()
        
        self.attention = QICAttention(embed_dim, n_heads, dropout)
        self.fc1 = QICLinear(embed_dim, hidden_dim)
        self.fc2 = QICLinear(hidden_dim, embed_dim)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor, 
                theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Attention block
        residual_real, residual_imag = x_real, x_imag
        x_real, x_imag = self.norm1(x_real), self.norm1(x_imag)
        attn_real, attn_imag = self.attention(x_real, x_imag, theta)
        x_real = residual_real + self.dropout(attn_real)
        x_imag = residual_imag + self.dropout(attn_imag)
        
        # FFN block
        residual_real, residual_imag = x_real, x_imag
        x_real, x_imag = self.norm2(x_real), self.norm2(x_imag)
        
        # FFN with magnitude activation
        h_real, h_imag = self.fc1(x_real, x_imag, theta)
        h_mag = QICAlgebra.magnitude(h_real, h_imag, theta)
        h_activated = F.gelu(h_mag)
        scale = h_activated / (h_mag + 1e-8)  # No unsqueeze needed, element-wise
        h_real = h_real * scale
        h_imag = h_imag * scale
        
        h_real, h_imag = self.fc2(h_real, h_imag, theta)
        x_real = residual_real + self.dropout(h_real)
        x_imag = residual_imag + self.dropout(h_imag)
        
        return x_real, x_imag


# ============================================================================
# QIC TRANSFORMER MODEL
# ============================================================================

class QICTransformer(nn.Module):
    """Parameter-efficient QIC Transformer with learnable theta."""
    
    def __init__(self, config: Config):
        super().__init__()
        
        # Use reduced QIC dimensions
        embed_dim = config.qic_embed_dim
        hidden_dim = config.qic_hidden_dim
        
        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, embed_dim)
        
        # Learnable theta
        self.theta = nn.Parameter(torch.tensor(config.theta_init))
        
        # Initialize imaginary part scale
        self.imag_scale = nn.Parameter(torch.ones(1) * 0.1)
        
        # QIC layers
        self.layers = nn.ModuleList([
            QICLayer(embed_dim, hidden_dim, config.n_heads, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        # Output
        self.output_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, config.num_classes)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos_ids)
        
        x_real = self.dropout(token_emb + pos_emb)
        x_imag = x_real * self.imag_scale  # Initialize imaginary part
        
        # Pass through layers
        for layer in self.layers:
            x_real, x_imag = layer(x_real, x_imag, self.theta)
        
        # Classification (use real part)
        x = self.output_norm(x_real)
        x = x.mean(dim=1)  # Global pooling
        return self.classifier(x)
    
    def get_theta_value(self) -> float:
        return self.theta.item()


# ============================================================================
# STANDARD TRANSFORMER BASELINE
# ============================================================================

class StandardTransformer(nn.Module):
    """Standard Transformer baseline with full dimensions."""
    
    def __init__(self, config: Config):
        super().__init__()
        
        # Use full dimensions
        embed_dim = config.std_embed_dim
        hidden_dim = config.std_hidden_dim
        
        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=config.n_heads,
            dim_feedforward=hidden_dim,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.n_layers)
        
        self.classifier = nn.Linear(embed_dim, config.num_classes)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        token_emb = self.token_embedding(input_ids)
        pos_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos_ids)
        
        x = self.dropout(token_emb + pos_emb)
        x = self.transformer(x)
        x = x.mean(dim=1)
        
        return self.classifier(x)


# ============================================================================
# DATASET LOADERS
# ============================================================================

class TextDataset(Dataset):
    """Generic text dataset for IMDB/AG News."""
    
    def __init__(self, texts, labels, vocab, max_length=256):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
    
    def encode_text(self, text):
        words = text.lower().split()[:self.max_length]
        indices = [self.vocab.get(w, 1) for w in words]  # 1 is <UNK>
        
        # Pad or truncate
        if len(indices) < self.max_length:
            indices.extend([0] * (self.max_length - len(indices)))  # 0 is <PAD>
        else:
            indices = indices[:self.max_length]
        
        return torch.tensor(indices, dtype=torch.long)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.encode_text(self.texts[idx]), self.labels[idx]


def load_imdb_data(config: Config, n_train=2000, n_test=500):
    """Load IMDB dataset."""
    print("Loading IMDB dataset...")
    
    # Load data
    dataset = load_dataset('imdb')
    train_texts = dataset['train']['text'][:n_train]
    train_labels = dataset['train']['label'][:n_train]
    test_texts = dataset['test']['text'][:n_test]
    test_labels = dataset['test']['label'][:n_test]
    
    # Build vocabulary
    print("Building vocabulary...")
    word_counts = Counter()
    for text in train_texts[:5000]:
        word_counts.update(text.lower().split())
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in word_counts.most_common(config.vocab_size - 2):
        vocab[word] = len(vocab)
    
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create datasets
    train_dataset = TextDataset(train_texts, train_labels, vocab, config.max_seq_len)
    test_dataset = TextDataset(test_texts, test_labels, vocab, config.max_seq_len)
    
    return train_dataset, test_dataset


def load_agnews_data(config: Config, n_train=4000, n_test=1000):
    """Load AG News dataset."""
    print("Loading AG News dataset...")
    
    # Load data
    dataset = load_dataset('ag_news')
    train_texts = dataset['train']['text'][:n_train]
    train_labels = dataset['train']['label'][:n_train]
    test_texts = dataset['test']['text'][:n_test]
    test_labels = dataset['test']['label'][:n_test]
    
    # Build vocabulary
    print("Building vocabulary...")
    word_counts = Counter()
    for text in train_texts:
        word_counts.update(text.lower().split())
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in word_counts.most_common(config.vocab_size - 2):
        vocab[word] = len(vocab)
    
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create datasets
    train_dataset = TextDataset(train_texts, train_labels, vocab, config.max_seq_len)
    test_dataset = TextDataset(test_texts, test_labels, vocab, config.max_seq_len)
    
    return train_dataset, test_dataset


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, loader, optimizer, criterion, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()
        total += target.size(0)
        
        if batch_idx % 50 == 0 and batch_idx > 0:
            print(f"  Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}")
    
    return total_loss / len(loader), correct / total


def evaluate(model, loader):
    """Evaluate model."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
            total += target.size(0)
    
    return correct / total


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiments():
    """Run complete experiments on IMDB and AG News."""
    
    if not DATASETS_AVAILABLE:
        print("ERROR: Install datasets library: pip install datasets")
        return
    
    print("="*80)
    print("QIC TRANSFORMER EXPERIMENTS ON REAL DATASETS")
    print("="*80)
    print(f"Device: {device}\n")
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    config = Config()
    
    # Parameter comparison
    print("PARAMETER COMPARISON")
    print("-"*40)
    
    std_model = StandardTransformer(config)
    qic_model = QICTransformer(config)
    
    std_params = sum(p.numel() for p in std_model.parameters())
    qic_params = sum(p.numel() for p in qic_model.parameters())
    
    print(f"Standard Transformer: {std_params:,} parameters")
    print(f"QIC Transformer: {qic_params:,} parameters")
    print(f"Reduction: {(1 - qic_params/std_params)*100:.1f}%\n")
    
    results = {}
    
    # IMDB Experiment
    print("="*80)
    print("EXPERIMENT 1: IMDB SENTIMENT ANALYSIS")
    print("="*80)
    
    config.num_classes = 2
    train_data, test_data = load_imdb_data(config)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config.batch_size)
    
    # Train Standard
    print("\nTraining Standard Transformer...")
    std_model = StandardTransformer(config).to(device)
    optimizer = torch.optim.Adam(std_model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    std_times = []
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        start = time.time()
        train_loss, train_acc = train_epoch(std_model, train_loader, optimizer, criterion, config)
        std_times.append(time.time() - start)
        test_acc = evaluate(std_model, test_loader)
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}")
    
    std_imdb_acc = test_acc
    std_imdb_time = np.mean(std_times)
    
    # Train QIC
    print("\nTraining QIC Transformer...")
    qic_model = QICTransformer(config).to(device)
    optimizer = torch.optim.Adam(qic_model.parameters(), lr=config.learning_rate)
    
    qic_times = []
    theta_history = []
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        start = time.time()
        train_loss, train_acc = train_epoch(qic_model, train_loader, optimizer, criterion, config)
        qic_times.append(time.time() - start)
        test_acc = evaluate(qic_model, test_loader)
        theta = qic_model.get_theta_value()
        theta_history.append(theta)
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}")
        print(f"  Theta: {theta:.4f}, J²: {QICAlgebra.j_squared(torch.tensor(theta)).item():.4f}")
    
    qic_imdb_acc = test_acc
    qic_imdb_time = np.mean(qic_times)
    
    # AG News Experiment
    print("\n" + "="*80)
    print("EXPERIMENT 2: AG NEWS CLASSIFICATION")
    print("="*80)
    
    config.num_classes = 4
    train_data, test_data = load_agnews_data(config)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config.batch_size)
    
    # Train Standard
    print("\nTraining Standard Transformer...")
    std_model = StandardTransformer(config).to(device)
    optimizer = torch.optim.Adam(std_model.parameters(), lr=config.learning_rate)
    
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        train_loss, train_acc = train_epoch(std_model, train_loader, optimizer, criterion, config)
        test_acc = evaluate(std_model, test_loader)
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}")
    
    std_agnews_acc = test_acc
    
    # Train QIC
    print("\nTraining QIC Transformer...")
    qic_model = QICTransformer(config).to(device)
    optimizer = torch.optim.Adam(qic_model.parameters(), lr=config.learning_rate)
    
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        train_loss, train_acc = train_epoch(qic_model, train_loader, optimizer, criterion, config)
        test_acc = evaluate(qic_model, test_loader)
        theta = qic_model.get_theta_value()
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}")
        print(f"  Theta: {theta:.4f}, J²: {QICAlgebra.j_squared(torch.tensor(theta)).item():.4f}")
    
    qic_agnews_acc = test_acc
    
    # Final Results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    print("\nPARAMETER COMPARISON:")
    print(f"  Standard: {std_params:,} parameters")
    print(f"  QIC: {qic_params:,} parameters ({(1-qic_params/std_params)*100:.1f}% reduction)")
    
    print("\nIMDB RESULTS:")
    print(f"  Standard: {std_imdb_acc:.3f} accuracy")
    print(f"  QIC: {qic_imdb_acc:.3f} accuracy ({(qic_imdb_acc-std_imdb_acc)*100:+.1f}%)")
    print(f"  Time: Standard {std_imdb_time:.1f}s vs QIC {qic_imdb_time:.1f}s per epoch")
    
    print("\nAG NEWS RESULTS:")
    print(f"  Standard: {std_agnews_acc:.3f} accuracy")
    print(f"  QIC: {qic_agnews_acc:.3f} accuracy ({(qic_agnews_acc-std_agnews_acc)*100:+.1f}%)")
    
    print("\nOVERALL:")
    avg_std = (std_imdb_acc + std_agnews_acc) / 2
    avg_qic = (qic_imdb_acc + qic_agnews_acc) / 2
    print(f"  Average accuracy - Standard: {avg_std:.3f}")
    print(f"  Average accuracy - QIC: {avg_qic:.3f}")
    
    if avg_qic >= avg_std - 0.02:
        print(f"\n✅ SUCCESS: QIC maintains performance with {(1-qic_params/std_params)*100:.1f}% fewer parameters!")
    else:
        print(f"\n⚠️ QIC accuracy is {(avg_std-avg_qic)*100:.1f}% lower")
    
    print("\nTheta Evolution (IMDB):", theta_history)


if __name__ == "__main__":
    run_experiments() 
