---
title: "CV Agent — Spec-driven LLM application"
tier: supporting
summary: "Modular Flask + React system that tailors CVs to job descriptions with local or cloud LLMs."
tags: ["Agents", "Full stack"]
order: 3
repo: https://github.com/Meloneneis/cv_agent_v2
---

## Problem
Useful resume assistance needs structured profiles, job understanding, controlled generation, and export — without prompt spaghetti.

## Approach
Specification-driven modules (profile, job analyzer, generator, templates, PDF, LLM service) with REST boundaries. Supports Ollama and cloud providers. React frontend with tests and migration docs across `cv_agent` → `cv_agent_v2`.

## Artifacts
- [cv_agent_v2](https://github.com/Meloneneis/cv_agent_v2)  
- [cv_agent](https://github.com/Meloneneis/cv_agent)
