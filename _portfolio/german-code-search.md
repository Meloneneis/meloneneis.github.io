---
title: "German Code Search through Knowledge Transfer"
excerpt: "Bachelor thesis adapting GraphCodeBERT for German–code retrieval — tokenizer adaptation, MLM pre-training, and a public Hugging Face checkpoint."
collection: portfolio
permalink: /portfolio/german-code-search/
---

<p class="work-feature__index" style="margin-top:0">Thesis · NLP · Code models</p>

## Problem
Most code-search models assume English documentation. German developer communities leave a gap: queries and comments in German paired with source that still needs structural understanding.

## Approach
End-to-end knowledge transfer on top of GraphCodeBERT:

1. **Tokenizer adaptation** — reshape the vocabulary toward German + code domains.
2. **Continued MLM pre-training** — interleave corpora so the encoder absorbs German linguistic signal without forgetting code structure.
3. **Dataset construction** — scrape and convert German GitHub Java repositories into documentation–function pairs.
4. **Fine-tuning** — train for retrieval so natural-language intents surface the right functions.

## Result
A reusable checkpoint, [`meloneneneis/graphcodebert-base-german`](https://huggingface.co/meloneneneis/graphcodebert-base-german), published on Hugging Face for fill-mask / further fine-tuning in German code-search pipelines.

## Links
- Repository: [german-code-retrieval](https://github.com/Meloneneis/german-code-retrieval)
- Model: [Hugging Face](https://huggingface.co/meloneneneis/graphcodebert-base-german)
