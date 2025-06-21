# QuantumComplexTransformer: A Transformer Model with a Matrix-Geometric caculation phase theta

This repository contains the PyTorch implementation of the Quantum-Algebraic Transformer (Q-Transformer), a novel Transformer architecture that operates on a tunable, two-component number system. This framework moves beyond standard real-number arithmetic to explore if a richer mathematical substrate can lead to more powerful and parameter-efficient models.
The core idea is to replace real numbers with Quantum-Complex (QC) numbers of the form z = a + bJ(θ). The behavior of the "imaginary" basis J(θ) is governed by a tunable phase θ, leading to the fundamental algebraic rule:
J(θ)² = (sin(2θ) - 1) * I
By making θ a learnable parameter, the Q-Transformer can dynamically adapt the rules of its own arithmetic during training. It can learn to operate in different algebraic regimes—behaving like standard complex numbers (J² = -1), nilpotent numbers (J² = 0), or even more exotic systems (J² = -2)—to best suit the problem at hand.
This repository provides the complete, re-derived code for an Encoder-Decoder Q-Transformer, integrating state-of-the-art techniques like RoPE, SwiGLU, and Grouped-Query Attention, all re-engineered for this new algebraic domain.

## Accuracy
#  BENCHMARKING REPORT: QUANTUM VS. STANDARD TRANSFORMER

## 1. BENCHMARK SETUP
------------------------------------------------------------
- **Objective:**  
  Compare performance of Quantum and Standard Transformers on a sequence classification task with matched parameter counts.

- **Task:**  
  Binary Classification (Is the sum of a sequence > 0?)

- **Training Device:**  
  CPU

- **Common Hyperparameters:**  
  - Epochs: 50  
  - Learning Rate: 0.001  
  - Batch Size: 32  

- **Quantum Transformer Unique Hyperparameter (θ):**  
  0.7854  

## 2. PARAMETER COUNT VERIFICATION
------------------------------------------------------------
- **Standard Transformer Parameters:** 17,665  
- **Quantum Transformer Parameters:** 16,853  
- **Parameter Count Difference:** 812 (**4.60%**)  

**Conclusion:** Parameter counts are closely matched for a fair comparison.

## 3. PERFORMANCE RESULTS
------------------------------------------------------------

| **Metric**                  | **Standard Transformer** | **Quantum Transformer** |
|----------------------------|-------------------------|------------------------|
| Final Validation Loss       | 0.0839                  | 0.0554                 |
| Final Validation Accuracy   | 96.50%                  | 98.00%                 |
| Total Training Time (sec)   | 21.17                   | 39.57                  |


## Installation

To get started, clone the repository and install the required dependencies (primarily PyTorch).

```bash
# Clone the repository
git clone https://github.com/your-username/quantum-gpt.git
cd quantum-gpt

# Install dependencies (it's recommended to use a virtual environment)
pip install torch
```

## Support & Donations

If you find this project interesting or useful, please consider supporting its development. Your contribution helps in maintaining the project and exploring new research directions.

[![Sponsor](https://img.shields.io/badge/Sponsor-EA4AAA?style=for-the-badge&logo=githubsponsors&logoColor=white)](https://github.com/sponsors/bhargavpatel431997)

## Citation

If you use this work, please consider citing the original paper that inspired this architecture:

> Patel, B. (2025). The Quantum Superposition Origin of Complex Numbers: A Foundational Framework for Quantum Mechanics and Phase-Dependent Quantum Computing

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
