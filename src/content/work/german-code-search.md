---
title: "German Code Search through Knowledge Transfer"
tier: flagship
summary: "Bachelor thesis adapting GraphCodeBERT for German–code retrieval — tokenizer adaptation, continued MLM pre-training, dataset construction, and a public Hugging Face checkpoint."
tags: ["Transformers", "GraphCodeBERT", "MLM", "Code retrieval"]
order: 1
repo: https://github.com/Meloneneis/german-code-retrieval
model: https://huggingface.co/meloneneneis/graphcodebert-base-german
---

## Problem
Most neural code-search models assume English documentation. German developer communities leave a gap: natural-language intent in German paired with source that still needs structural understanding.

## Constraints
Limited labeled German doc–function pairs; need to transfer from English-centric GraphCodeBERT without destroying code structure priors; produce something others can run.

## Approach
1. **Tokenizer adaptation** toward German + code domains  
2. **Continued MLM pre-training** on interleaved corpora  
3. **Dataset construction** from German GitHub Java repositories into documentation–function pairs  
4. **Retrieval fine-tuning** so NL intents surface the right functions  

## Artifacts
- Repository: [german-code-retrieval](https://github.com/Meloneneis/german-code-retrieval)  
- Public model: [`meloneneneis/graphcodebert-base-german`](https://huggingface.co/meloneneneis/graphcodebert-base-german)  

## What I’d do next
Publish retrieval metrics on a fixed held-out German split and add a minimal demo notebook that embeds a query and returns top-k functions.
