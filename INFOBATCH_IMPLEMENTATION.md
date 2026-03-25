This `README.md` provides the implementation details for integrating **InfoBatch** into a semantic segmentation workflow, specifically optimized for benchmarks like **ADE20K**.

---

# README: InfoBatch Implementation for Semantic Segmentation

## Overview
**InfoBatch** is a plug-and-play, architecture-agnostic framework that achieves lossless training acceleration through **unbiased dynamic data pruning**. It works by randomly pruning "well-learned" samples with low loss values and rescaling the gradients of the remaining samples to maintain the original gradient expectation.

## Implementation Steps

### 1. Score Initialization and Scoring Strategy
*   **Initial Epoch:** For the first epoch ($t=0$), initialize the scores for all samples to **{1}**, as no previous loss values are available.
*   **Segmentation Loss:** Semantic segmentation involves two types of pixel-level Cross Entropy losses: one for segmentation and one for classification.
*   **Representative Score ($H(z)$):** To determine a single score for pruning, use the **weighted sum of these two losses calculated at the sample level**.

### 2. Dynamic Soft Pruning
*   **Adaptive Threshold ($H̄_t$):** At the beginning of each epoch, calculate the **mean value of all current sample scores** to serve as the pruning threshold.
*   **Pruning Condition:** Identify samples where the current score is less than the mean ($H_t(z) < H̄_t$).
*   **Random Pruning:** Apply a predefined pruning probability ($r$) to these low-score samples. Samples with scores equal to or higher than the mean ($H_t(z) \geq H̄_t$) are never pruned (probability = 0).

### 3. Expectation Rescaling
*   **Gradient Compensation:** To ensure the training objective remains a constant-rescaled version of the original, you must scale up the gradients of the **remaining** low-score samples that were not pruned.
*   **Rescale Factor:** Multiply the loss (or gradient) of these samples by **$1/(1-r)$**. Gradients for high-loss samples are left unmodified.

### 4. Score Maintenance
*   **Updating Scores:** For all samples used in the current epoch ($S_t$), update their stored scores with the latest loss values obtained during the forward pass.
*   **Unmodified Scores:** For samples that were pruned and skipped in the current epoch ($z \in D \setminus S_t$), their stored scores remain unmodified from the previous epoch.

### 5. Annealing Phase
*   **Stability Period:** To reduce performance variance and eliminate remaining bias, stop pruning and rescaling in the final stage of training.
*   **Full Dataset Training:** Use the full dataset for the remaining epochs once the training reaches the annealing ratio **$\delta$** (e.g., the last 12.5% to 15% of epochs).

---
*Reference: Qin, Z., et al. "InfoBatch: Lossless Training Speed Up by Unbiased Dynamic Data Pruning."*