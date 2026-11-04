# Margin Regularisation Against Batch-Induced Token Flips

This repository contains the experimental materials for a research project investigating how batching during greedy decoding can induce token flips in large language models, and how logit-margin regularisation can mitigate that effect.

The project studies the interaction between:

- batching and autoregressive decoding,
- precision-related perturbations (fp32, fp16, bf16),
- logit margin structure and top-1/top-2 separation,
- model-dependent sensitivity to batch-induced perturbations.

The repository is centered around a documented analysis notebook and the corresponding research paper, along with experiment artifacts used to evaluate the observed effects.

## Overview

In greedy decoding, a token is chosen by the maximum logit. However, when sequences are processed in a batch, perturbations in the model's internal computations can change the local ordering of logits and cause the selected token to flip even when the model is otherwise unchanged.

This project looks at how those flips arise and whether regularising the top-1/top-2 logit margin can reduce the likelihood of such flips. The work evaluates this both from a precision-scaling perspective and through model-wise empirical sweeps.

Key themes explored in the repository include:

- batch-induced perturbation analysis,
- top-1/top-2 logit margin regularisation,
- precision scaling across fp32, fp16 and bf16,
- comparison of model families and decoding conditions,
- evaluation of whether a universal invariant exists or whether effects are model-specific.

## Repository contents

- `Margin_FineTuning_documented.ipynb` – the main notebook containing the documented experimental analysis, training/measurement logic, and summary results.
- `ResearchPaper.pdf` – the paper associated with the project.
- `*.json` artifact files – experiment result dumps for precision sweeps, regime comparisons, model comparisons, and related measurements.

Examples of included artifacts:

- `boundary.json`
- `frontier.json`
- `pooled_evidence.json`
- `ratio_sweep.json`
- `ratio_sweep3.json`
- `regime_transfer.json`
- `instruct_gsm8k.json`
- `big_model_qgap.json`
- `sysA.json`
- `sysB.json`

## Main findings

The notebook and experiments highlight several important conclusions:

1. Batch-induced perturbations can be non-zero even when the control setting (single-sequence decoding) is stable.
2. Precision strongly affects the magnitude of the perturbation; the observed scaling is consistent with representational precision differences.
3. The measured effect is not fully explained by a universal, number-format-only rule; the model itself matters.
4. A model-specific margin calibration is often necessary rather than a single closed-form formula.

The analysis also notes that behaviour can differ significantly across model families and precisions, which suggests that margin regularisation must be understood alongside model architecture and numerical behaviour.

## Setup and requirements

This repository is primarily designed for research workflows in Python notebooks, typically on Google Colab or a GPU-enabled local environment.

### Recommended environment

- Python 3.10+
- CUDA-capable GPU (L4, A100, or similar)
- bf16 support is preferred for the precision experiments
- Jupyter or Google Colab

### Python dependencies

```bash
pip install -q transformers datasets accelerate scipy
```

The notebook also uses Hugging Face `transformers` and `datasets` libraries to load models and benchmark data.

## Getting started

1. Open `Margin_FineTuning_documented.ipynb` in Jupyter or Colab.
2. Ensure the environment has a compatible GPU.
3. Run the notebook cells in order.
4. Review the experimental outputs, summary tables, and saved JSON artifacts.

## Data and models

The analysis uses a mix of standard research datasets and open-source LLMs, including:

- WikiText-2 for characterisation and perturbation measurement,
- GSM8K for systems-level experiments,
- MBPP for out-of-domain checks,
- Qwen, SmolLM, and Pythia model families for cross-model evaluation.

## Notes

- This project is research-oriented and intended for experimental analysis.
- The notebook is the clearest entry point for understanding the methodology and conclusions.
- The JSON files are useful for comparing experiment runs and summarising sweep behaviour.

## Contact / authorship

The repository is associated with the author listed in the notebook:

- Charan Teja Nagisettygari

## Summary

This repository documents a research effort to understand and mitigate batch-induced token flips in greedy decoding through logit-margin analysis and precision-aware experimentation. It combines a documented notebook, paper, and experimental artifacts to provide a reproducible foundation for the study.
