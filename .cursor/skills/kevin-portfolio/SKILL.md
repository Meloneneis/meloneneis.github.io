---
name: kevin-portfolio
description: Maintain Kevin Van Le's GitHub Pages portfolio (meloneneis.github.io) — design system, case studies, 3D hero, blog motion. Use when editing the homepage, portfolio CSS/JS, projects, CV, or blog presentation.
---

# Kevin Van Le Portfolio Skill

## When to use
Editing `meloneneis.github.io` visual identity, homepage, case studies, CV, or blog UX.

## Hard constraints
- Follow `.cursor/rules/portfolio-design.mdc`.
- Do not invent employers, degrees, or metrics.
- Keep the protein-backbone Three.js metaphor (not neon neural particles).
- Prefer Jekyll + static assets over introducing a SPA unless explicitly requested.

## Content sources of truth
- GitHub: https://github.com/Meloneneis
- HF: https://huggingface.co/meloneneneis
- Email: lekevinvan@gmail.com
- Flagship: German code search thesis + `graphcodebert-base-german`

## Verify
```bash
bundle exec jekyll build
```
Check `/`, `/portfolio/*`, `/posts/2024/02/DL-in-the-Science/`, `/cv/`.
