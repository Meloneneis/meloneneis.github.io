---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

Kevin Van Le
======
Machine learning engineer — NLP, code models, reinforcement learning, applied LLM systems.

- Email: [lekevinvan@gmail.com](mailto:lekevinvan@gmail.com)
- GitHub: [Meloneneis](https://github.com/Meloneneis)
- Hugging Face: [meloneneneis](https://huggingface.co/meloneneneis)
- Portfolio: [meloneneis.github.io](https://meloneneis.github.io)

Education
======
* Bachelor's thesis: **German Code Search through Knowledge Transfer** — adapting GraphCodeBERT via tokenizer adaptation, continued MLM pre-training, German GitHub dataset construction, and retrieval fine-tuning. Public model: [`graphcodebert-base-german`](https://huggingface.co/meloneneneis/graphcodebert-base-german).

Selected projects
======
* **German Code Retrieval** — End-to-end knowledge transfer for German–code search ([repo](https://github.com/Meloneneis/german-code-retrieval)).
* **Semantic Plausibility (SHROOM-inspired)** — Reward-model fine-tuning and synthetic labeling for hallucination / implausibility detection ([repo](https://github.com/Meloneneis/SemanticPlausibilitySS24)).
* **DQN Racing** — Configurable vision DQN with dueling option, mixed-precision benchmarks, W&B tracking ([repo](https://github.com/Meloneneis/DQN_Training)).
* **CV Agent** — Spec-driven modular LLM app (Flask + React) for job-tailored CVs with local/cloud providers ([v2](https://github.com/Meloneneis/cv_agent_v2)).
* **Legal OCR** — Fine-tuning / inference for German legal document scans ([repo](https://github.com/Meloneneis/glm_ocr)).

Writing
======
* [Evolutionary-scale prediction of atomic-level protein structure with a language model]({{ base_path }}/posts/2024/02/DL-in-the-Science/) — technical deep dive into ESM-2 / ESMFold with original animated figures.

Skills
======
* **ML / DL:** PyTorch, Transformers, representation learning, MLM, retrieval, RL (DQN)
* **NLP:** code search, semantic plausibility, OCR, evaluation pipelines
* **Systems:** Python, Flask, React, experiment tracking (W&B), GPU training
* **Languages:** German, English

Portfolio
======
<ul>{% for post in site.portfolio %}
  {% include archive-single.html %}
{% endfor %}</ul>
