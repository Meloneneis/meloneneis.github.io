---
title: "CV Agent — Spec-driven LLM application"
excerpt: "Modular Flask + React platform that analyzes job descriptions and generates tailored CVs with local or cloud LLMs."
collection: portfolio
permalink: /portfolio/cv-agent/
---

## Problem
Generic resume tools ignore the job. Useful AI assistance needs structured user profiles, job understanding, controlled generation, and export — without a monolith of prompt spaghetti.

## Approach
Two iterations (`cv_agent`, `cv_agent_v2`) culminating in a **specification-driven modular architecture**:

- Independent modules: profile, job analyzer, CV generator, templates, PDF, auth, LLM service
- REST boundaries between modules
- Support for **Ollama** (local) and cloud providers (OpenAI, Anthropic, DeepSeek)
- React frontend with test suites and migration docs

## Why it matters
Evidence of product-minded ML engineering: treating LLM features as software with contracts, tests, and deployable seams — not a single chat box.

## Links
- [cv_agent](https://github.com/Meloneneis/cv_agent)
- [cv_agent_v2](https://github.com/Meloneneis/cv_agent_v2)
