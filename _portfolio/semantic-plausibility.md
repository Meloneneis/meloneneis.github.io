---
title: "Semantic Plausibility & Hallucination Detection"
excerpt: "Reward-model pipelines and synthetic labels for detecting implausible model generations (SHROOM-inspired)."
collection: portfolio
permalink: /portfolio/semantic-plausibility/
---

## Problem
Fluent language models still emit statements that are locally coherent and globally wrong. Detecting *semantic implausibility* is a practical guardrail for generation systems.

## Approach
Project work around SHROOM-style plausibility / hallucination signals:

- Dataset analysis scripts for class balance and failure modes
- Fine-tuning starting from `OpenAssistant/reward-model-deberta-v3-base`
- Synthetic label generation to expand scarce supervision
- Reproducible skip/preload flags so training, synthesis, and evaluation can be isolated

Validated on consumer GPU hardware (RTX 3070 Ti mobile, 8GB) with a multi-hour full pipeline.

## Why it matters
Shows comfort with **evaluation-centric NLP**: not only training a classifier, but building the data machinery that makes the metric trustworthy.

## Links
- Repository: [SemanticPlausibilitySS24](https://github.com/Meloneneis/SemanticPlausibilitySS24)
