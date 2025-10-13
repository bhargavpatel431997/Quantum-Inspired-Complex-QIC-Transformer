# Quantum-Inspired Complex (QIC) Transformers

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS%202025-NEGEL-blue)](https://neurips.cc/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **"Quantum-Inspired Complex Transformers: Resolving the Fundamental Algebraic Ambiguity for Enhanced Neural Representations"** accepted at NeurIPS 2025 NEGEL Workshop.

**Author:** Bhargav Patel </br>
**Institution:** Independent Researcher </br>
**Email:** b.patel.physics@gmail.com </br>
**ORCID:** [0009-0004-5429-2771](https://orcid.org/0009-0004-5429-2771) </br>

---

## 🚀 Overview

QIC Transformers introduce a novel approach to neural network design by making the imaginary unit **learnable** rather than fixed. Instead of arbitrarily choosing $i$ or $-i$ as the solution to $x^2 = -1$, we treat it as a quantum superposition:

$$J(\theta) = \cos(\theta)J_+ + \sin(\theta)J_-$$

where $\theta$ is a trainable parameter. This creates a continuously parameterized family of algebras that enables:

- **47.2% parameter reduction** while maintaining or improving accuracy
- **Adaptive algebraic structures** that networks learn during training
- **Task-specific mathematical regimes** discovered automatically

### Key Results

| Dataset | Standard Transformer | QIC Transformer | Improvement |
|---------|---------------------|-----------------|-------------|
| **Parameters** | 1,466,370 | 774,407 | **-47.2%** |
| **IMDB** | 100.0% | 100.0% | 0.0% |
| **AG News** | 73.3% | **78.0%** | **+4.7%** |
| **Memory (Training)** | 1.82 GB | 1.15 GB | **-36.8%** |

---

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Experiments](#experiments)
- [Reproducing Paper Results](#reproducing-paper-results)
- [Code Structure](#code-structure)
- [Mathematical Background](#mathematical-background)
- [Citation](#citation)
- [License](#license)

---

## 🔧 Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (optional, for GPU acceleration)

### Step 1: Clone the Repository

```bash
git clone https://github.com/bhargavpatel431997/Quantum-Inspired-Complex-QIC-Transformer.git
cd Quantum-Inspired-Complex-QIC-Transformer/Neurips2025
```

### Step 2: Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install datasets numpy
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

---

## 🏃 Quick Start

### Basic Usage

Run the complete experiment with default settings:

```bash
python code.py
```

This will:
1. Compare parameter counts between Standard and QIC Transformers
2. Train both models on IMDB sentiment analysis
3. Train both models on AG News categorization
4. Display comprehensive results and statistics

### Expected Output

```
================================================================================
COMPLETE QIC EVALUATION: 2M PARAMETERS + REAL DATASETS
================================================================================

STEP 1: PARAMETER COMPARISON
Standard Transformer: 1,466,370 parameters
QIC Transformer: 774,407 parameters
Parameter Reduction: 47.2%
✅ Target 20% reduction achieved!

STEP 2: IMDB DATASET
Training Standard Transformer on IMDB...
Training QIC Transformer on IMDB...

STEP 3: AG NEWS DATASET
Training Standard Transformer on AG News...
Training QIC Transformer on AG News...

FINAL RESULTS SUMMARY
✅ SUCCESS: QIC maintains performance with 47.2% fewer parameters!
```

---

## 🏗️ Architecture

### Standard Transformer

Traditional Transformer with:
- Embedding dimension: 128
- 4 layers with 4 attention heads each
- Separate Q, K, V projection matrices per layer
- Feed-forward networks with 512 hidden units
- Total: **~1.47M parameters**

### QIC Transformer

Novel architecture featuring:
- **Shared weight matrices** across layers (key innovation)
- **Learnable phase parameter** $\theta$ for algebraic adaptation
- Complex-valued representations: $z = a + bJ(\theta)$
- Bidirectional FFN using matrix transposes
- Total: **~774K parameters** (47.2% reduction)

### Key Innovation: QIC Multiplication

For QIC numbers $z_1 = a_1 + b_1J(\theta)$ and $z_2 = a_2 + b_2J(\theta)$:

$$z_1 \cdot z_2 = [a_1a_2 + b_1b_2(-1 + \sin(2\theta))] + [a_1b_2 + b_1a_2]J(\theta)$$

The term $\sin(2\theta)$ creates learnable algebraic interactions unavailable to fixed complex networks.

---

## 🧪 Experiments

### Dataset Configuration

The code supports two real-world text classification datasets:

#### 1. IMDB Sentiment Analysis
- **Task:** Binary sentiment classification (positive/negative)
- **Vocabulary:** 5,000 most frequent words
- **Sequence Length:** 256 tokens
- **Training samples:** 2,000 (or full 25,000)
- **Test samples:** 500 (or full 25,000)

#### 2. AG News Categorization
- **Task:** 4-class news categorization (World, Sports, Business, Tech)
- **Vocabulary:** 5,000 most frequent words
- **Sequence Length:** 256 tokens
- **Training samples:** 4,000 (or full 120,000)
- **Test samples:** 1,000 (or full 7,600)

### Training Configuration

```python
config = ModelConfig2M(
    vocab_size=5000,
    embed_dim=128,
    hidden_dim=512,
    n_heads=4,
    n_layers=4,
    max_seq_len=256,
    batch_size=32,
    learning_rate=2e-3,
    epochs=5,
    dropout=0.1,
    theta_init=math.pi/4  # QIC-specific
)
```

---

## 📊 Reproducing Paper Results

### Full Dataset Training

To reproduce exact paper results with full datasets:

```python
# In code.py, set use_small_data=False
results = run_complete_experiment(use_small_data=False)
```

**Note:** Full training requires:
- ~10-20 GB RAM
- GPU recommended (NVIDIA A100 used in paper)
- ~2-4 hours total training time

### Quick Testing (Small Data)

For rapid prototyping and testing:

```python
results = run_complete_experiment(use_small_data=True)
```

This uses 2,000 IMDB samples and 4,000 AG News samples (~10-15 minutes on GPU).

### Statistical Validation

Run multiple seeds for statistical significance:

```python
seeds = [42, 123, 456, 789, 1011]
all_results = []

for seed in seeds:
    torch.manual_seed(seed)
    np.random.seed(seed)
    results = run_complete_experiment(use_small_data=False)
    all_results.append(results)
```

Paper reports:
- AG News: Standard = 73.3% ± 1.2%, QIC = 78.0% ± 0.9%
- Two-sample t-test: p < 0.001 (highly significant)
- Effect size (Cohen's d): 4.52 (very large)

---

## 📁 Code Structure

```
Quantum-Inspired-Complex-QIC-Transformer/
│
├── Neurips2025/
│   ├── code.py                 # Main implementation
│   ├── README.md               # This file
```

### Core Components

#### 1. Model Architectures (`code.py`)

- **`StandardTransformer2M`**: Baseline Transformer implementation
- **`QICTransformer2M`**: QIC Transformer with learnable algebra
- **`StandardLayer`**: Standard attention + FFN layer
- **`QICLayer`**: QIC layer with shared weights and phase parameter

#### 2. Datasets

- **`IMDBDataset`**: IMDB sentiment analysis wrapper
- **`AGNewsDataset`**: AG News categorization wrapper
- Both handle vocabulary building, tokenization, and encoding

#### 3. Training

- **`Trainer`**: Unified training and evaluation loop
- Supports gradient clipping, learning rate scheduling
- Tracks losses, accuracies, and timing metrics

#### 4. Evaluation

- **`run_complete_experiment()`**: Main experiment pipeline
- Compares Standard vs QIC across both datasets
- Reports parameter counts, accuracies, timing

---

## 📐 Mathematical Background

### The Fundamental Ambiguity

The equation $x^2 = -1$ has two solutions in any extension of real numbers:

$$x_+ = +\sqrt{-1}, \quad x_- = -\sqrt{-1}$$

Traditional mathematics arbitrarily chooses one as $i$. QIC treats this as a learnable superposition.

### QIC Algebra Properties

The QIC algebra satisfies:

1. **Closure:** $\forall z_1, z_2 \in \text{QIC}, \, z_1 \cdot z_2 \in \text{QIC}$
2. **Associativity:** $(z_1 \cdot z_2) \cdot z_3 = z_1 \cdot (z_2 \cdot z_3)$
3. **Commutativity:** $z_1 \cdot z_2 = z_2 \cdot z_1$
4. **Submultiplicativity:** $|z_1 \cdot z_2| \leq C(\theta)|z_1||z_2|$ where $C(\theta) = \sqrt{1 + \sin^2(2\theta)} \in [1, \sqrt{2}]$

### Matrix Representation

QIC numbers use 2×2 real matrices:

$$J(\theta) = \begin{pmatrix} 0 & \sin\theta - \cos\theta \\ \cos\theta - \sin\theta & 0 \end{pmatrix}$$

A general QIC number $z = a + bJ(\theta)$ becomes:

$$z = \begin{pmatrix} a & b(\sin\theta - \cos\theta) \\ b(\cos\theta - \sin\theta) & a \end{pmatrix}$$

### Non-Triviality

QIC cannot be reduced to gauge transformations of fixed complex networks because:

$$J(\theta)^2 = -1 + \sin(2\theta)$$

The deviation $\sin(2\theta)$ creates learnable algebraic interactions absent in standard $\mathbb{C}$.

---

## 🔬 Extending the Code

### Custom Datasets

Add your own dataset:

```python
class CustomDataset(Dataset):
    def __init__(self, split='train', max_length=256, vocab_size=5000):
        # Load your data
        self.texts = load_your_texts()
        self.labels = load_your_labels()
        self.vocab = self._build_vocabulary()

    def _build_vocabulary(self):
        # Build vocab from texts
        pass

    def encode_text(self, text):
        # Tokenize and encode
        pass

    def __getitem__(self, idx):
        return self.encode_text(self.texts[idx]), self.labels[idx]
```

### Hyperparameter Tuning

Modify `ModelConfig2M`:

```python
config = ModelConfig2M(
    embed_dim=256,        # Increase model capacity
    n_layers=6,           # Deeper network
    n_heads=8,            # More attention heads
    learning_rate=1e-3,   # Adjust learning rate
    theta_init=math.pi/6  # Different initialization
)
```

### Per-Head Phase Parameters

For more expressive models:

```python
# In QICTransformer2M.__init__()
self.theta = nn.Parameter(
    torch.ones(config.n_layers, config.n_heads) * config.theta_init
)

# In forward pass
x_real, x_imag = layer(x_real, x_imag, self.theta[i])
```

---

## 📈 Performance Tips

### GPU Optimization

```python
# Enable CUDA benchmarking
torch.backends.cudnn.benchmark = True

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(data)
    loss = criterion(output, target)
scaler.scale(loss).backward()
```

### Memory Efficiency

```python
# Gradient checkpointing for large models
from torch.utils.checkpoint import checkpoint

def forward_layer(x):
    return checkpoint(layer, x)
```

### Faster Data Loading

```python
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,      # Parallel data loading
    pin_memory=True,    # Faster GPU transfer
    prefetch_factor=2   # Prefetch batches
)
```

---

## 🐛 Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**
```python
# Reduce batch size
config.batch_size = 16

# Or reduce model size
config.embed_dim = 64
config.n_layers = 2
```

**2. Slow Training**
```python
# Enable GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use smaller dataset for testing
results = run_complete_experiment(use_small_data=True)
```

**3. Dataset Download Issues**
```python
# Manually specify cache directory
dataset = load_dataset('imdb', cache_dir='./my_cache')

# Or download datasets beforehand
from datasets import load_dataset
load_dataset('imdb', cache_dir='./data_cache')
load_dataset('ag_news', cache_dir='./data_cache')
```

**4. Import Errors**
```bash
# Ensure all dependencies installed
pip install --upgrade torch datasets numpy
```

---

## 📄 Citation

If you use this code or find our work helpful, please cite:

```bibtex
@inproceedings{patel2025qic,
  title={Quantum-Inspired Complex Transformers: Resolving the Fundamental Algebraic Ambiguity for Enhanced Neural Representations},
  author={Patel, Bhargav},
  booktitle={NeurIPS 2025 Workshop on Non-Euclidean Foundation Models and Geometric Learning (NEGEL)},
  year={2025}
}
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📬 Contact

**Bhargav Patel**
- Email: b.patel.physics@gmail.com
- ORCID: [0009-0004-5429-2771](https://orcid.org/0009-0004-5429-2771)
- GitHub: [@bhargavpatel431997](https://github.com/bhargavpatel431997)

For questions, issues, or collaborations, feel free to open an issue or reach out directly.

---

## 🙏 Acknowledgments

- Thanks to the NeurIPS 2025 NEGEL workshop organizers and reviewers for valuable feedback
- HuggingFace for the `datasets` library
- PyTorch team for the excellent deep learning framework
- The open-source community for inspiration and tools

---

## 📊 Additional Resources

- **Paper (arXiv):** Coming soon
- **Poster:** Available upon request
- **Presentation Slides:** Available upon request
- **Supplementary Materials:** See `main.tex` appendix

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Last Updated:** October 2025
**Version:** 1.0.0
**Status:** Active Development

