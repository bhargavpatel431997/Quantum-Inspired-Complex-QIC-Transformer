# Quantum Transformer Net: A Transformer Model with a Matrix-Geometric caculation phase theta

This repository contains the PyTorch implementation of the Quantum-Algebraic Transformer (Q-Transformer), a novel Transformer architecture that operates on a tunable, two-component number system. This framework moves beyond standard real-number arithmetic to explore if a richer mathematical substrate can lead to more powerful and parameter-efficient models.
The core idea is to replace real numbers with Quantum-Complex (QC) numbers of the form z = a + bJ(θ). The behavior of the "imaginary" basis J(θ) is governed by a tunable phase θ, leading to the fundamental algebraic rule:
J(θ)² = (sin(2θ) - 1) * I
By making θ a learnable parameter, the Q-Transformer can dynamically adapt the rules of its own arithmetic during training. It can learn to operate in different algebraic regimes—behaving like standard complex numbers (J² = -1), nilpotent numbers (J² = 0), or even more exotic systems (J² = -2)—to best suit the problem at hand.
This repository provides the complete, re-derived code for an Encoder-Decoder Q-Transformer, integrating state-of-the-art techniques like RoPE, SwiGLU, and Grouped-Query Attention, all re-engineered for this new algebraic domain.

# Benchmark: Quantum-Inspired vs. Standard Transformer

This report details a comparative benchmark between a standard Transformer model and a custom Transformer implemented with **Quantum-Inspired Complex (QC) arithmetic**. The goal was to assess performance, parameter efficiency, and computational cost on a sequence classification task.

## Executive Summary

The Quantum-Inspired Transformer demonstrates **superior performance and parameter efficiency** compared to its standard counterpart. On a sequence sum classification task, the QC Transformer achieved a **higher final validation accuracy (98.50% vs. 97.75%)** while utilizing **~21% fewer parameters**.

However, this improved representational power comes at a significant computational cost, with the QC Transformer's training time being **more than double** that of the standard model due to the more complex arithmetic operations involved.

## Key Findings

-   **Higher Accuracy:** The QC Transformer consistently outperformed the standard model, achieving a higher peak accuracy and a lower final validation loss.
-   **Parameter Efficiency:** The QC model achieved its superior results with **4,522 fewer parameters** (a 20.96% reduction), highlighting its potential for creating more powerful models within a smaller parameter budget.
-   **Computational Overhead:** The QC Transformer is significantly slower, taking **98.04 seconds** to train versus the standard model's **45.24 seconds** (~2.17x slower).
-   **Faster Convergence:** Despite the slower per-epoch time, the QC model reached the 95% accuracy threshold faster in terms of epochs, suggesting more efficient learning dynamics.

## Benchmark Configuration

The experiment was designed to be a fair comparison by using identical hyperparameters where possible and matching the total parameter counts as closely as feasible.

| Parameter                      | Standard Transformer         | Quantum-Inspired Transformer   |
| ------------------------------ | ---------------------------- | ------------------------------ |
| `embed_dim`                    | 32                           | 20                             |
| `n_heads`                      | 2                            | 2                              |
| `n_layers`                     | 2                            | 2                              |
| `ffn_dim_multiplier`           | 2                            | 2                              |
| `learning_rate`                | 0.001                        | 0.001                          |
| `epochs`                       | 50                           | 50                             |
| `batch_size`                   | 32                           | 32                             |
| **Total Trainable Parameters** | **21,570**                   | **17,048 (-20.96%)**           |
| **Unique Hyperparameter**      | N/A                          | `initial_theta = 0.7854`       |

-   **Task:** Binary classification on sequences of integers (predicting if the sum > 0).
-   **Device:** All tests were run on `cpu` to ensure a fair comparison of computational steps without GPU-specific kernel optimizations.

## Performance Results

### Final Metrics

The QC Transformer finished the 50-epoch training run with better validation metrics across the board.

| Metric                    | Standard Transformer | Quantum-Inspired Transformer | Delta                     |
| ------------------------- | -------------------- | ---------------------------- | ------------------------- |
| **Final Validation Acc.** | **97.75%**           | **98.50%**                   | **+0.75%**                |
| **Final Validation Loss** | 0.0475               | 0.0361                       | **-24.0%** (Lower is better) |
| **Total Training Time**   | 45.24 sec            | 98.04 sec                    | **+116.7%** (Higher is worse) |

### Visual Comparison

The following plots illustrate the learning dynamics over 50 epochs.

![image](https://github.com/user-attachments/assets/780a1c53-23de-4dd8-8451-6ef4fc80beaf)


1.  **Validation Loss:** The QC Transformer consistently maintains a lower validation loss throughout the training process, indicating better model fit.
2.  **Validation Accuracy:** The QC Transformer's accuracy curve remains slightly but consistently above the standard model's, converging to a higher final value.
3.  **Training Time:** The bar chart clearly shows the significant computational overhead of the QC model's complex arithmetic.

## In-Depth Analysis

### 1. Representational Power

The results strongly suggest that the QC algebra, with its learnable phase parameter `θ` and hypercomplex matrix multiplications (`qc_bmm`), provides a richer representational capacity. The model was able to capture the underlying pattern in the data more effectively, even with a smaller embedding dimension (`embed_dim=20` vs. `32`).

### 2. Computational Cost Breakdown

The ~2.17x increase in training time can be attributed to the core operations in the QC model:
-   **`QComplexDense`:** A standard `F.linear` operation involves one matrix multiplication. The QC equivalent performs two matrix multiplications and several element-wise operations to compute the real and imaginary outputs.
-   **`qc_bmm` (Attention Scores):** Similarly, calculating attention scores requires four `torch.matmul` calls and additional arithmetic, compared to a single `matmul` in the standard attention mechanism.

These additional operations, while providing more modeling power, are inherently more computationally expensive on current hardware.

### 3. Learnable Phase Parameter (`θ`)

The `θ` parameters in the QC model are learnable, allowing the model to adapt the nature of its own algebra during training.
-   **Initial `θ`:** 0.7854 (π/4)
-   **Final Average `θ`:** 0.7841
-   **Final `θ` Range:** [0.7712, 0.7925]

The model made minor adjustments to `θ` from its initial value, indicating that it found the initial setting to be effective but still benefited from fine-tuning the algebraic properties of its layers and attention heads.

## Conclusion

The Quantum-Inspired Transformer is a promising architecture that demonstrates a clear trade-off: **it can achieve superior accuracy and parameter efficiency at the cost of significantly increased computation time.**

This makes it a compelling choice for scenarios where:
1.  **Model size is a critical constraint** (e.g., edge devices).
2.  **Maximum possible performance is required**, and longer training times are acceptable.

Future work could explore hardware-specific optimizations or model quantization to mitigate the computational overhead, potentially making QC Transformers a more practical alternative to standard architectures.
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
