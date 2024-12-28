# Bridging Generations: Comparative Performance of LSTM and Transformer in Language Translation

## Overview
This project investigates the comparative performance of two prominent deep learning architectures, LSTM (Long Short-Term Memory) and Transformer, in the domain of English-to-Hindi translation. By leveraging the **IIT Bombay English-Hindi Parallel Corpus**, the models are evaluated on their ability to generate accurate translations, their computational efficiency, and their handling of long-term dependencies in language.

The study aims to provide insights into the trade-offs between these architectures, helping researchers and practitioners make informed decisions for real-world translation tasks.

---

## Table of Contents
- [Introduction](#introduction)
- [Motivation](#motivation)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [LSTM Model](#lstm-model)
  - [Transformer Model](#transformer-model)
- [Comparison Metrics](#comparison-metrics)
- [Results](#results)
- [Challenges Faced](#challenges-faced)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [References](#references)

---

## Introduction
Language translation systems have undergone a paradigm shift, thanks to advancements in machine learning. Traditionally, **Recurrent Neural Networks (RNNs)** and their variant **LSTMs** have been the cornerstone of sequence-to-sequence tasks. However, the advent of the **Transformer** architecture, as introduced in Vaswani et al.’s seminal paper *“Attention is All You Need”*, has significantly enhanced the capabilities of natural language processing systems.

This project, titled **“Bridging Generations: Comparative Performance of LSTM and Transformer in Language Translation,”** explores:
1. The strengths and limitations of LSTM in handling sequential data.
2. The superior capabilities of Transformers in capturing long-range dependencies and parallelizing computations.

---

## Motivation
Despite the rapid adoption of Transformer models, many applications still rely on LSTM due to its simplicity and suitability for small-scale tasks. This project addresses the following:
- **Understanding trade-offs:** When should one use LSTM over Transformer?
- **Performance benchmarks:** Establishing metrics to compare these models in real-world tasks like English-to-Hindi translation.
- **Advancing research:** Offering a detailed analysis to guide future work in neural machine translation.

---

## Dataset
The **IIT Bombay English-Hindi Parallel Corpus**, sourced from the **Hugging Face datasets library**, is utilized for this study. This dataset is renowned for its quality and relevance in English-to-Hindi translation.

### Dataset Details
- **Structure:** Each entry consists of parallel sentences in English and Hindi under the `"translation"` key.
- **Preprocessing:** Sentences are tokenized and converted to numerical representations using embedding techniques.
- **Splits:**
  - **Training Set:** Used to train models on input-output sentence pairs.
  - **Validation Set:** Used to monitor and adjust performance during training.
  - **Test Set:** Held out for final evaluation.

---

## Methodology

### LSTM Model
#### Key Features:
1. **Encoder-Decoder Architecture:**
   - The encoder processes the input sentence token by token, creating a hidden vector.
   - The decoder uses this hidden vector to generate output translations.
2. **Attention Mechanism:**
   - Helps focus on relevant parts of the input sentence during decoding.
3. **Beam Search:**
   - A heuristic search algorithm to find the most probable sentence.

#### Training Details:
- **Optimizer:** Adam.
- **Loss Function:** Categorical Cross-Entropy Loss.
- **Hyperparameters:** Learning rate of 0.001; beta_1 = 0.9, beta_2 = 0.999.

---

### Transformer Model
#### Key Features:
1. **Self-Attention Mechanism:**
   - Captures contextual relationships between words efficiently.
2. **Multi-Headed Attention:**
   - Processes input through multiple attention mechanisms for robust feature extraction.
3. **Parallel Processing:**
   - Unlike LSTMs, the Transformer processes entire sequences simultaneously, making it computationally efficient.

#### Training Details:
- **Optimizer:** AdamW (Adam with Weight Decay).
- **Loss Function:** Label Smoothed Cross-Entropy Loss.
- **Hyperparameters:** Learning rate of \(2 \times 10^{-5}\); weight decay = 0.01.

---

## Comparison Metrics
To evaluate the performance of both models, the following metrics were employed:
1. **BLEU Score:** Measures n-gram overlap between predicted translations and reference sentences.
2. **Efficiency:** Training time and computational resource usage.
3. **Contextual Understanding:** Ability to handle long-range dependencies in sequences.
4. **Scalability:** Performance on increasing dataset sizes.

---

## Results

| Metric                | LSTM         | Transformer   |
|------------------------|--------------|---------------|
| **BLEU Score**        | 6.3          | 6.9           |
| **Training Time**     | Higher       | Lower         |
| **Memory Efficiency** | Moderate     | High          |
| **Context Handling**  | Short-term   | Long-term     |
| **Scalability**       | Limited      | Excellent     |

### Observations:
- Transformers outperform LSTMs in handling long sentences and achieving higher BLEU scores.
- LSTMs are more suitable for smaller datasets or tasks with low computational resources.

---

## Challenges Faced
1. **Data Preprocessing:** Aligning tokenization methods for both models.
2. **Hyperparameter Tuning:** Finding optimal configurations for Transformer training.
3. **Computational Resources:** High GPU memory requirements for Transformer models.

---

## Conclusion
The **Transformer architecture** demonstrated significant advantages over LSTM in this comparative study:
- **Higher translation accuracy** as measured by BLEU scores.
- **Faster training times** due to parallel processing.
- **Better scalability** to larger datasets and longer sentences.

LSTMs, while effective for simpler tasks, are limited in their ability to process long-term dependencies and parallelize computations.

---

## Future Work
To build upon this research, the following directions are suggested:
1. **Domain-Specific Fine-Tuning:** Apply models to specialized domains (e.g., medical or legal translations).
2. **Alternative Metrics:** Use ROUGE or METEOR for a more comprehensive evaluation.
3. **Optimization Techniques:** Explore pruning or quantization to reduce Transformer model size.
4. **Hybrid Architectures:** Combine the strengths of LSTMs and Transformers.

---

## References
1. **[The IIT Bombay English-Hindi Parallel Corpus](https://scholar.google.co.in/citations?view_op=view_citation&hl=en&user=jnoUuGcAAAAJ&citation_for_view=jnoUuGcAAAAJ:bnK-pcrLprsC)**
2. Laskar, S.R., et al. Neural Machine Translation: English to Hindi. *DOI: [Link](https://doi.org/10.1109/cict48419.2019.9066238)*
3. Singh, A., et al. Machine Translation Systems. *DOI: [Link](https://doi.org/10.1051/itmconf/20224403004)*
4. Joshi, N., et al. Human and Automatic Evaluation. *DOI: [Link](https://doi.org/10.1007/978-3-642-30157-5_42)*

---
[Download Project Report](./https://drive.google.com/file/d/1krvdqcNTB8K9Gy4EboEQDPozIK60bns9/view?usp=sharing)
