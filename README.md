# Turkish GRPO Reward Ensembles for Multiple-Choice Question Answering

This project investigates reward engineering, GRPO fine-tuning and model merging for Turkish multiple-choice question answering using GPT-2-based language models.

The main focus is not only improving answer accuracy, but also analyzing model behavior under different reward strategies, especially mode collapse, output diversity and format compliance.

---

# Project Overview

Large language models can follow answer-format instructions after supervised fine-tuning, but they may still suffer from limited answer diversity and mode collapse.

In this project, a Turkish multiple-choice question answering task was used to evaluate how different reward functions affect model behavior.

The workflow includes:

* Supervised Fine-Tuning (SFT)
* GRPO-based reward optimization
* Reward function design
* Diversity and entropy-based behavior analysis
* Model merging with MergeKit
* Comparative evaluation of output distributions

---

# Methodology

The project pipeline consists of the following stages:

1. Dataset subset selection
2. Prompt construction for A–E multiple-choice questions
3. Data cleaning and answer-label formatting
4. Supervised Fine-Tuning (SFT)
5. GRPO fine-tuning with multiple reward functions
6. Model merging using MergeKit
7. Evaluation with accuracy, format rate, dominant answer share and prediction entropy

---

# Reward Strategies

The following reward strategies were evaluated:

## Baseline Reward

* Accuracy reward
* Format compliance reward

## Alternative 1: Diversity Reward

Encourages different answer choices across multiple generations for the same prompt.

## Alternative 2: Entropy-Based Reward

Uses the entropy of the generated answer distribution to encourage balanced predictions.

## Alternative 3: Format + Diversity Reward

Combines format compliance with diversity encouragement.

## Alternative 4: Behavior-Balancing Reward

Reduces repetitive answer behavior without applying direct negative penalties.

---

# Model Merging

After training multiple GRPO variants, the models were merged using MergeKit.

The merging strategy was designed to combine:

* baseline model stability,
* diversity-oriented behavior,
* entropy-based distributional balance,
* and format-aware reward behavior.

The merged model was used to analyze whether parameter-level merging could reduce mode collapse more effectively than reward engineering alone.

---

# Results Summary

The experiments showed that SFT successfully taught the model the required output format.

However, GRPO reward optimization alone was not sufficient to fully resolve mode collapse.

The merged model reduced the dominant answer share and produced a more diverse prediction distribution compared to individual GRPO models.

## Key Findings

* SFT achieved strong format compliance.
* GRPO variants maintained high format rate.
* Individual reward strategies could not fully eliminate mode collapse.
* MergeKit-based model merging improved prediction diversity.
* Output diversity improved more clearly than accuracy.

---

# Evaluation Metrics

The models were evaluated using:

* Accuracy
* Format Rate
* Dominant Answer Share
* Prediction Entropy
* A–E Prediction Distribution

These metrics were selected to evaluate not only correctness, but also behavioral characteristics of the generated outputs.

---

# Repository Structure

```text
turkish-grpo-reward-ensembles/
│
├── README.md
├── requirements.txt
├── RewardEnsembles.ipynb
├── merge_multi.yaml
│
└── results/
```

---

# Libraries and Tools

* PyTorch
* Hugging Face Transformers
* TRL
* PEFT
* MergeKit
* Hugging Face Hub
* Pandas
* NumPy
* Matplotlib
* scikit-learn

---

# Key Topics

* Large Language Models
* GRPO
* Reward Engineering
* Supervised Fine-Tuning
* Model Merging
* MergeKit
* Mode Collapse Analysis
* Turkish NLP
* Multiple-Choice Question Answering

---

# Author

Emir Kaan SAIT

Yıldız Technical University
M.Sc. Computer Engineering
