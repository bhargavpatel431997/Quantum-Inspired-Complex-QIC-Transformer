# qgpt.py - Quantum Generative Pre-trained Transformer
# Based on the Matrix-Geometric Origin of Quantum Reality Framework

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('QGPT')

# =============================================================================
# QUANTUM EMERGENCE FRAMEWORK COMPONENTS
# =============================================================================

class QuantumEmergenceRFFEmbedding(nn.Module):
    """
    Random Fourier Features using the Quantum Emergence Framework
    
    Implements dual rotation matrices:
    - B(β) = cos(β)I + sin(β)J  (emergence operator)
    - H(β) = -cos(β)I + sin(β)J (complementary operator)
    
    Where J = [[0, -1], [1, 0]] represents the imaginary unit
    These operators satisfy: B(β) × H(β) = -I
    """
    def __init__(self, input_dim, output_dim, num_frequencies, gamma=1.0, trainable_beta=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_frequencies = num_frequencies
        
        # Ensure even dimensions for rotation matrices
        self.padded_input_dim = input_dim if input_dim % 2 == 0 else input_dim + 1
        self.num_pairs = self.padded_input_dim // 2
        
        # Random frequencies for rotation angles
        frequencies = torch.randn(self.num_pairs, num_frequencies) * math.sqrt(2 * gamma)
        self.register_buffer('frequencies', frequencies)
        
        # Random phase shifts
        phases = torch.rand(num_frequencies) * 2 * math.pi
        self.register_buffer('phases', phases)
        
        # Learnable rotation angle β (fundamental quantum parameter)
        if trainable_beta:
            self.beta = nn.Parameter(torch.tensor(math.pi / 6))  # Default π/6
        else:
            self.register_buffer('beta', torch.tensor(math.pi / 6))
        
        # Output projection
        self.output_projection = nn.Linear(2 * self.padded_input_dim * num_frequencies, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # Emergence strength parameter
        self.emergence_strength = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x):
        original_shape = x.shape[:-1]
        
        # Flatten batch dimensions
        if len(x.shape) > 2:
            x = x.view(-1, x.shape[-1])
        
        # Pad if necessary
        if self.input_dim % 2 == 1:
            x = torch.cat([x, torch.zeros(x.shape[0], 1, device=x.device)], dim=-1)
        
        # Reshape to pairs
        x_pairs = x.view(x.shape[0], self.num_pairs, 2)
        x1 = x_pairs[:, :, 0]
        x2 = x_pairs[:, :, 1]
        
        B_features = []  # B(β) operator features
        H_features = []  # H(β) operator features
        
        for freq_idx in range(self.num_frequencies):
            # Calculate rotation angles
            angles = torch.matmul(x1, self.frequencies[:, freq_idx]) + self.phases[freq_idx]
            angles = angles.unsqueeze(1) * self.emergence_strength
            
            # Apply quantum rotation angle β
            effective_beta = angles * self.beta
            cos_beta = torch.cos(effective_beta)
            sin_beta = torch.sin(effective_beta)
            
            # B(β) operator: [[cos(β), -sin(β)], [sin(β), cos(β)]]
            x1_B = x1 * cos_beta - x2 * sin_beta
            x2_B = x1 * sin_beta + x2 * cos_beta
            B_rotated = torch.stack([x1_B, x2_B], dim=-1).view(x.shape[0], -1)
            B_features.append(B_rotated)
            
            # H(β) operator: [[-cos(β), -sin(β)], [sin(β), -cos(β)]]
            x1_H = x1 * (-cos_beta) - x2 * sin_beta
            x2_H = x1 * sin_beta + x2 * (-cos_beta)
            H_rotated = torch.stack([x1_H, x2_H], dim=-1).view(x.shape[0], -1)
            H_features.append(H_rotated)
        
        # Combine features from both operators
        quantum_features = torch.cat(B_features + H_features, dim=-1)
        quantum_features = quantum_features * math.sqrt(1.0 / self.num_frequencies)
        
        # Project and normalize
        output = self.output_projection(quantum_features)
        output = self.layer_norm(output)
        
        # Reshape to original
        if len(original_shape) > 1:
            output = output.view(*original_shape, self.output_dim)
        
        return output
    
    def get_quantum_metrics(self):
        """Return quantum metrics for monitoring"""
        return {
            'beta': self.beta.item(),
            'emergence_strength': self.emergence_strength.item(),
            'oscillation_frequency': 2 * abs(torch.cos(self.beta).item())
        }


class QuantumMultiHeadAttention(nn.Module):
    """
    Quantum-Enhanced Multi-Head Attention
    Uses emergence operators for attention computation
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Quantum-enhanced projections using RFF
        self.q_proj = QuantumEmergenceRFFEmbedding(d_model, d_model, num_frequencies=64)
        self.k_proj = QuantumEmergenceRFFEmbedding(d_model, d_model, num_frequencies=64)
        self.v_proj = QuantumEmergenceRFFEmbedding(d_model, d_model, num_frequencies=64)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Learnable quantum temperature
        self.quantum_temp = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.shape
        
        # Quantum projections
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Quantum-scaled attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores / self.quantum_temp  # Apply quantum temperature
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax with quantum correction
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        output = self.out_proj(context)
        
        return output, attn_weights


class QuantumTransformerBlock(nn.Module):
    """
    Quantum Transformer Block with emergence operators
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Quantum attention
        self.attention = QuantumMultiHeadAttention(d_model, n_heads, dropout)
        
        # Quantum feed-forward using emergence RFF
        self.ff_quantum = nn.Sequential(
            QuantumEmergenceRFFEmbedding(d_model, d_ff, num_frequencies=128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer norms
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Quantum residual scaling
        self.quantum_residual = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x, mask=None):
        # Quantum attention with residual
        attn_out, _ = self.attention(x, x, x, mask)
        x = x + self.quantum_residual * self.dropout(attn_out)
        x = self.ln1(x)
        
        # Quantum feed-forward with residual
        ff_out = self.ff_quantum(x)
        x = x + self.quantum_residual * self.dropout(ff_out)
        x = self.ln2(x)
        
        return x


class QuantumPositionalEncoding(nn.Module):
    """
    Quantum-aware positional encoding using rotation matrices
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Standard sinusoidal encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
        # Quantum modulation
        self.quantum_modulation = QuantumEmergenceRFFEmbedding(
            d_model, d_model, num_frequencies=32, trainable_beta=True
        )
    
    def forward(self, x):
        seq_len = x.size(1)
        pos_encoding = self.pe[:, :seq_len, :]
        
        # Apply quantum modulation to positional encoding
        quantum_pos = self.quantum_modulation(pos_encoding)
        
        return x + quantum_pos


# =============================================================================
# QUANTUM GPT MODEL
# =============================================================================

class QGPT(nn.Module):
    """
    Quantum Generative Pre-trained Transformer
    
    Integrates the Matrix-Geometric Framework throughout:
    - Quantum embeddings using B(β) and H(β) operators
    - Quantum attention mechanisms
    - Emergence strength tracking
    - Geometric alignment optimization
    """
    def __init__(
        self,
        vocab_size,
        d_model=768,
        n_heads=12,
        n_layers=12,
        d_ff=3072,
        max_len=1024,
        dropout=0.1,
        tie_weights=True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token embeddings with quantum enhancement
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.quantum_token_enhance = QuantumEmergenceRFFEmbedding(
            d_model, d_model, num_frequencies=64
        )
        
        # Quantum positional encoding
        self.pos_encoding = QuantumPositionalEncoding(d_model, max_len)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Quantum transformer blocks
        self.transformer_blocks = nn.ModuleList([
            QuantumTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model)
        
        # Output projection with quantum correction
        self.output_quantum = QuantumEmergenceRFFEmbedding(
            d_model, d_model, num_frequencies=32
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights between embedding and output
        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight
        
        # Quantum metrics tracking
        self.emergence_tracker = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 64),
                nn.GELU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            ) for _ in range(n_layers)
        ])
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"Initialized QGPT with {self.count_parameters():,} parameters")
        logger.info(f"Quantum emergence framework integrated throughout architecture")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids,
        attention_mask=None,
        return_emergence_scores=False
    ):
        batch_size, seq_len = input_ids.shape
        
        # Create causal mask
        if attention_mask is None:
            attention_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=input_ids.device)
            ).unsqueeze(0).unsqueeze(1)
        
        # Token embeddings with quantum enhancement
        token_embeds = self.token_embedding(input_ids)
        token_embeds = self.quantum_token_enhance(token_embeds)
        
        # Add quantum positional encoding
        x = self.pos_encoding(token_embeds)
        x = self.dropout(x)
        
        # Track emergence scores
        emergence_scores = []
        
        # Apply quantum transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, attention_mask)
            
            # Calculate emergence score for this layer
            if return_emergence_scores:
                layer_emergence = self.emergence_tracker[i](x.mean(dim=1))
                emergence_scores.append(layer_emergence)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Apply quantum output transformation
        x = self.output_quantum(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)
        
        if return_emergence_scores:
            emergence_scores = torch.stack(emergence_scores, dim=1)
            return logits, emergence_scores
        
        return logits
    
    def generate(
        self,
        input_ids,
        max_length=100,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        num_return_sequences=1,
        use_quantum_sampling=True
    ):
        """
        Generate text with quantum-enhanced sampling
        """
        self.eval()
        
        batch_size = input_ids.shape[0] * num_return_sequences
        input_ids = input_ids.repeat_interleave(num_return_sequences, dim=0)
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                # Get logits
                logits, emergence_scores = self.forward(
                    input_ids, 
                    return_emergence_scores=True
                )
                next_token_logits = logits[:, -1, :]
                
                # Apply quantum temperature scaling
                if use_quantum_sampling:
                    # Use emergence strength to modulate temperature
                    quantum_temp = temperature * (1 + 0.5 * emergence_scores[:, -1, 0])
                    next_token_logits = next_token_logits / quantum_temp
                else:
                    next_token_logits = next_token_logits / temperature
                
                # Apply top-k and top-p filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                
                if do_sample:
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(probs, dim=-1, keepdim=True)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Stop if all sequences have generated EOS token
                if (next_token == self.token_embedding.num_embeddings - 1).all():
                    break
        
        return input_ids
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_quantum_metrics(self):
        """Get quantum metrics from all components"""
        metrics = {}
        
        # Collect metrics from quantum embeddings
        for name, module in self.named_modules():
            if isinstance(module, QuantumEmergenceRFFEmbedding):
                module_metrics = module.get_quantum_metrics()
                for key, value in module_metrics.items():
                    metrics[f"{name}.{key}"] = value
        
        return metrics


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

class QuantumTrainer:
    """
    Trainer for QGPT with quantum-aware optimization
    """
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader=None,
        learning_rate=3e-4,
        warmup_steps=1000,
        device='cuda'
    ):
        self.model = model.to(device)
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Quantum-aware optimizer with different learning rates
        quantum_params = []
        regular_params = []
        
        for name, param in model.named_parameters():
            if 'beta' in name or 'quantum' in name:
                quantum_params.append(param)
            else:
                regular_params.append(param)
        
        self.optimizer = torch.optim.AdamW([
            {'params': regular_params, 'lr': learning_rate},
            {'params': quantum_params, 'lr': learning_rate * 0.1}  # Lower LR for quantum params
        ], weight_decay=0.01)
        
        # Learning rate scheduler
        self.scheduler = self.get_linear_schedule_with_warmup(
            self.optimizer, warmup_steps, len(train_dataloader) * 10
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.step = 0
    
    @staticmethod
    def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / 
                float(max(1, num_training_steps - num_warmup_steps))
            )
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_emergence = 0
        
        progress_bar = tqdm(self.train_dataloader, desc="Training")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass with emergence tracking
            logits, emergence_scores = self.model(
                input_ids, 
                return_emergence_scores=True
            )
            
            # Compute loss
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            
            # Add quantum regularization
            emergence_reg = 0.01 * (1 - emergence_scores.mean())
            total_loss_step = loss + emergence_reg
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss_step.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            total_emergence += emergence_scores.mean().item()
            self.step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'emergence': f"{emergence_scores.mean().item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        avg_loss = total_loss / len(self.train_dataloader)
        avg_emergence = total_emergence / len(self.train_dataloader)
        
        return avg_loss, avg_emergence
    
    def evaluate(self):
        if self.val_dataloader is None:
            return None, None
        
        self.model.eval()
        total_loss = 0
        total_emergence = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits, emergence_scores = self.model(
                    input_ids,
                    return_emergence_scores=True
                )
                
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )
                
                total_loss += loss.item()
                total_emergence += emergence_scores.mean().item()
        
        avg_loss = total_loss / len(self.val_dataloader)
        avg_emergence = total_emergence / len(self.val_dataloader)
        
        return avg_loss, avg_emergence


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def create_qgpt_model(vocab_size=50257, **kwargs):
    """
    Create a QGPT model with default GPT-2 like configuration
    """
    default_config = {
        'd_model': 768,
        'n_heads': 12,
        'n_layers': 12,
        'd_ff': 3072,
        'max_len': 1024,
        'dropout': 0.1,
        'tie_weights': True
    }
    
    default_config.update(kwargs)
    
    model = QGPT(vocab_size=vocab_size, **default_config)
    
    # Log quantum metrics
    quantum_metrics = model.get_quantum_metrics()
    logger.info("Quantum Metrics at Initialization:")
    for name, value in quantum_metrics.items():
        logger.info(f"  {name}: {value:.4f}")
    
    return model


if __name__ == "__main__":
    # Example: Create and test QGPT
    logger.info("Creating Quantum GPT model...")
    
    # Create model
    model = create_qgpt_model(vocab_size=50257)
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    test_input = torch.randint(0, 50257, (batch_size, seq_len))
    
    logger.info(f"Testing forward pass with input shape: {test_input.shape}")
    
    with torch.no_grad():
        output, emergence = model(test_input, return_emergence_scores=True)
        
    logger.info(f"Output shape: {output.shape}")
    logger.info(f"Emergence scores shape: {emergence.shape}")
    logger.info(f"Mean emergence strength: {emergence.mean().item():.4f}")
    
    # Test generation
    logger.info("\nTesting quantum text generation...")
    prompt = torch.tensor([[1, 2, 3, 4, 5]])  # Example token IDs
    
    generated = model.generate(
        prompt,
        max_length=50,
        temperature=0.8,
        use_quantum_sampling=True
    )
    
    logger.info(f"Generated sequence shape: {generated.shape}")
    
    # Display quantum metrics
    logger.info("\nFinal Quantum Metrics:")
    quantum_metrics = model.get_quantum_metrics()
    for name, value in quantum_metrics.items():
        if 'beta' in name:
            logger.info(f"  {name}: {value:.4f} rad ({value*180/math.pi:.1f}°)")
        else:
            logger.info(f"  {name}: {value:.4f}")
    
    logger.info("\nQGPT successfully created and tested!")
    logger.info("The model integrates quantum emergence throughout the architecture")