---
title: "Semantic Plausibility & Hallucination Detection"
tier: supporting
summary: "Reward-model fine-tuning and synthetic labels for detecting implausible generations (SHROOM-inspired)."
tags: ["NLP", "Evaluation"]
order: 2
repo: https://github.com/Meloneneis/SemanticPlausibilitySS24
---

## Problem
Fluent models still emit statements that are locally coherent and globally wrong. Detecting semantic implausibility is a practical guardrail.

## Approach
SHROOM-inspired pipeline: dataset analysis, fine-tuning from `OpenAssistant/reward-model-deberta-v3-base`, synthetic label expansion, and skip/preload flags so synthesis, training, and evaluation stay separable. Validated on consumer GPU (RTX 3070 Ti mobile).

## Artifacts
- [SemanticPlausibilitySS24](https://github.com/Meloneneis/SemanticPlausibilitySS24)
