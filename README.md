# QuantumComplexTransformer: A Transformer Model with a Matrix-Geometric caculation phase theta

This repository contains the PyTorch implementation of the Quantum-Algebraic Transformer (Q-Transformer), a novel Transformer architecture that operates on a tunable, two-component number system. This framework moves beyond standard real-number arithmetic to explore if a richer mathematical substrate can lead to more powerful and parameter-efficient models.
The core idea is to replace real numbers with Quantum-Complex (QC) numbers of the form z = a + bJ(θ). The behavior of the "imaginary" basis J(θ) is governed by a tunable phase θ, leading to the fundamental algebraic rule:
J(θ)² = (sin(2θ) - 1) * I
By making θ a learnable parameter, the Q-Transformer can dynamically adapt the rules of its own arithmetic during training. It can learn to operate in different algebraic regimes—behaving like standard complex numbers (J² = -1), nilpotent numbers (J² = 0), or even more exotic systems (J² = -2)—to best suit the problem at hand.
This repository provides the complete, re-derived code for an Encoder-Decoder Q-Transformer, integrating state-of-the-art techniques like RoPE, SwiGLU, and Grouped-Query Attention, all re-engineered for this new algebraic domain.

## Installation

To get started, clone the repository and install the required dependencies (primarily PyTorch).

```bash
# Clone the repository
git clone https://github.com/your-username/quantum-gpt.git
cd quantum-gpt

# Install dependencies (it's recommended to use a virtual environment)
pip install torch
```

## Usage

The main model is defined in `quantum_gpt.py`. You can import `QuantumGPT` and use it like any other PyTorch `nn.Module`.

### 1. Model Instantiation

First, define the model configuration and create an instance of `QuantumGPT`.

```python
import torch
from quantum_gpt import QuantumGPT

# --- Model Hyperparameters ---
config = {
    'vocab_size': 50257,  # Example: GPT-2 vocab size
    'd_model': 768,       # Must be even for the quantum layer
    'n_layer': 12,
    'n_head': 12,
    'd_ff': 768 * 4,
    'dropout': 0.1,
    'max_seq_len': 1024,
}

# --- Instantiate the Model ---
model = QuantumGPT(**config)
print(f"QuantumGPT model initialized with {model.count_parameters():,} parameters.")

# Move model to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
```

### 2. Forward Pass

You can run a forward pass with dummy data to see it in action.

```python
# --- Create Dummy Data and Run a Forward Pass ---
batch_size = 4
seq_len = 128
dummy_input = torch.randint(0, config['vocab_size'], (batch_size, seq_len)).to(device)
dummy_targets = torch.randint(0, config['vocab_size'], (batch_size, seq_len)).to(device)

print(f"\nRunning a forward pass with input shape: {dummy_input.shape}")

# Get logits and loss
logits, loss = model(dummy_input, targets=dummy_targets)

print(f"Output logits shape: {logits.shape}")
if loss is not None:
    print(f"Calculated loss: {loss.item():.4f}")

# Example of generating text (without targets)
logits, _ = model(dummy_input)
print(f"Generation logits shape: {logits.shape}")
```

### 3. Training Loop

The model can be integrated into a standard PyTorch training loop.

```python
# (Assuming you have a dataloader)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits, loss = model(inputs, targets=targets)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
```
## Future Work & Experiments

This implementation serves as a proof-of-concept. Potential areas for further research include:

-   **Empirical Evaluation**: Training `QuantumGPT` and a standard GPT baseline of identical size on a benchmark dataset (e.g., TinyStories, OpenWebText) to compare perplexity, convergence speed, and downstream task performance.
-   **Ablation Studies**:
    -   Compare the performance of the full `B(β)Êj + H(β)Êk` model against models using only one of the operators.
    -   Analyze the effect of making the rotation angle `β` a static hyperparameter versus a learned, data-dependent value.
-   **Exploring Different Geometries**: The paper's framework is general. One could experiment with other 2x2 matrices for `Êj` and `Êk` that satisfy the required algebraic constraints.
-   **Scaling Laws**: Investigate if the scaling laws for `QuantumGPT` differ from those of standard transformers.

## Support & Donations

If you find this project interesting or useful, please consider supporting its development. Your contribution helps in maintaining the project and exploring new research directions.

[![Sponsor](https://img.shields.io/badge/Sponsor-EA4AAA?style=for-the-badge&logo=githubsponsors&logoColor=white)](https://github.com/sponsors/bhargavpatel431997)

## Citation

If you use this work, please consider citing the original paper that inspired this architecture:

> Patel, B. (2025). *The Matrix-Geometric Origin of Quantum Reality: Universal Emergence Through Real Rotation Matrices and a New Paradigm for Quantum Computing*.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
