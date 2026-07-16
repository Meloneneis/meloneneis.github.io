---
layout: portfolio
permalink: /
title: "Kevin Van Le — Machine Learning Engineer"
author_profile: false
redirect_from:
  - /about/
  - /about.html
---

<a class="skip-link" href="#work">Skip to work</a>

<nav class="p-nav" aria-label="Primary">
  <a class="p-nav__brand" href="/">Kevin Van Le</a>
  <ul class="p-nav__links">
    <li><a href="#work">Work</a></li>
    <li class="hide-sm"><a href="#proof">Results</a></li>
    <li><a href="#about">About</a></li>
    <li class="hide-sm"><a href="#writing">Writing</a></li>
    <li><a href="#contact">Contact</a></li>
  </ul>
</nav>

<header class="hero">
  <div id="protein-canvas" class="hero__canvas" aria-hidden="true"></div>
  <div class="hero__veil"></div>
  <div class="hero__content">
    <h1 class="hero__brand">Kevin Van&nbsp;Le</h1>
    <p class="hero__headline">I build models that find structure in language, code, and control.</p>
    <p class="hero__lede">
      Machine learning engineer focused on representation learning, NLP, and applied RL —
      from German code search to agents that turn messy goals into working systems.
    </p>
    <div class="hero__cta">
      <a class="btn btn--solid" href="#work">Selected work</a>
      <a class="btn btn--ghost" href="mailto:lekevinvan@gmail.com">Email me</a>
    </div>
  </div>
  <p class="hero__hint">Move to explore · Protein backbone</p>
</header>

<section class="section" id="work">
  <div class="section__inner">
    <p class="section__label reveal">Selected work</p>
    <h2 class="section__title reveal">Projects worth opening in an interview</h2>
    <p class="section__lede reveal">
      Flagship pieces that show end-to-end ownership: data, training, evaluation, and product thinking.
    </p>

    <article class="work-feature reveal">
      <div>
        <p class="work-feature__index">01 — Thesis</p>
        <h3 class="work-feature__title">German Code Search through Knowledge Transfer</h3>
        <p class="work-feature__body">
          Bachelor thesis on adapting GraphCodeBERT for German–code retrieval: tokenizer adaptation,
          continued MLM pre-training, GitHub-scale dataset construction, and fine-tuning for
          documentation-to-function search.
        </p>
        <ul class="work-feature__meta">
          <li>Transformers</li>
          <li>GraphCodeBERT</li>
          <li>MLM pre-training</li>
          <li>Code retrieval</li>
        </ul>
        <a class="btn btn--ink" href="/portfolio/german-code-search/">Case study</a>
        <a class="btn btn--outline-ink" href="https://huggingface.co/meloneneneis/graphcodebert-base-german" target="_blank" rel="noopener">Hugging Face model</a>
      </div>
      <p class="work-feature__result">
        <strong>Outcome</strong>
        Published <em>graphcodebert-base-german</em> on Hugging Face — a reusable German-adapted code encoder for retrieval workflows.
      </p>
    </article>

    <div class="work-list">
      <a class="work-row reveal" href="/portfolio/semantic-plausibility/">
        <span class="work-row__num">02</span>
        <div>
          <h3 class="work-row__title">Semantic Plausibility &amp; Hallucination Detection</h3>
          <p class="work-row__blurb">Reward-model fine-tuning on SHROOM-style signals with synthetic label pipelines for detecting implausible generations.</p>
        </div>
        <span class="work-row__tag">NLP · Evaluation</span>
      </a>
      <a class="work-row reveal" href="/portfolio/dqn-racing/">
        <span class="work-row__num">03</span>
        <div>
          <h3 class="work-row__title">Deep Q-Network Racing Agent</h3>
          <p class="work-row__blurb">Configurable DQN / dueling architectures, mixed-precision benchmarks, and W&amp;B-tracked training for vision-based control.</p>
        </div>
        <span class="work-row__tag">RL · PyTorch</span>
      </a>
      <a class="work-row reveal" href="/portfolio/cv-agent/">
        <span class="work-row__num">04</span>
        <div>
          <h3 class="work-row__title">CV Agent — Spec-driven LLM application</h3>
          <p class="work-row__blurb">Modular Flask + React system that tailors CVs to job descriptions with local/cloud LLMs and specification-first modules.</p>
        </div>
        <span class="work-row__tag">Agents · Full stack</span>
      </a>
      <a class="work-row reveal" href="/portfolio/glm-ocr/">
        <span class="work-row__num">05</span>
        <div>
          <h3 class="work-row__title">German Legal Document OCR</h3>
          <p class="work-row__blurb">Fine-tuning and inference tooling for extracting structured text from complex scanned legal PDFs.</p>
        </div>
        <span class="work-row__tag">OCR · Fine-tuning</span>
      </a>
    </div>
  </div>
</section>

<section class="section section--stage" id="proof">
  <div class="section__inner proof-grid">
    <div>
      <p class="section__label reveal">Signal, not slogans</p>
      <h2 class="section__title reveal">What I optimize for</h2>
      <p class="section__lede reveal">
        Clear problem framing, reproducible training loops, and artifacts other people can actually run —
        models on Hugging Face, scripts with skip flags, specs before glue code.
      </p>
    </div>
    <div>
      <div class="proof-stat reveal">
        <p class="proof-stat__value"><span data-count="1" data-decimals="0"></span></p>
        <p class="proof-stat__label">Public German GraphCodeBERT checkpoint for code search transfer</p>
      </div>
      <div class="proof-stat reveal">
        <p class="proof-stat__value"><span data-count="5" data-decimals="0"></span></p>
        <p class="proof-stat__label">Shipped case studies spanning retrieval, evaluation, RL, agents, and OCR</p>
      </div>
      <div class="proof-stat reveal">
        <p class="proof-stat__value">End-to-end</p>
        <p class="proof-stat__label">From scraper → tokenizer → pre-train → fine-tune → packaged model</p>
      </div>
    </div>
  </div>
</section>

<section class="section section--band" id="about">
  <div class="section__inner about-grid">
    <div class="about-portrait reveal">
      <img src="/images/profile.png" alt="Portrait of Kevin Van Le" width="720" height="720" loading="lazy">
    </div>
    <div class="about-copy reveal">
      <p class="section__label">About</p>
      <h2 class="section__title">Curious about how models latch onto structure</h2>
      <p>
        I’m Kevin Van Le — I work where representation learning meets practical systems:
        code that has to retrieve, text that has to be trustworthy, and agents that have to ship.
      </p>
      <p>
        My bachelor thesis asked a simple question with hard consequences: can knowledge transfer
        make code search work when the natural language side is German? That led me through tokenizer
        surgery, continued pre-training, and building training data from real repositories.
      </p>
      <p>
        Since then I’ve explored semantic plausibility for generated text, vision-based RL for racing control,
        OCR for dense legal scans, and LLM apps designed module-by-module so behavior stays inspectable.
        I also write long-form technical notes — including a visual deep dive into ESMFold.
      </p>
      <ul class="skill-rail">
        <li>PyTorch</li>
        <li>Transformers</li>
        <li>NLP / Code LLMs</li>
        <li>Reinforcement Learning</li>
        <li>Python</li>
        <li>Flask / React</li>
        <li>Experiment tracking</li>
        <li>German ↔ English</li>
      </ul>
    </div>
  </div>
</section>

<section class="section" id="writing">
  <div class="section__inner">
    <p class="section__label reveal">Writing</p>
    <h2 class="section__title reveal">Notes from the lab notebook</h2>
    <p class="section__lede reveal">Long-form explanations with diagrams I animate myself — not slide-deck summaries.</p>

    <a class="writing-item reveal" href="/posts/2024/02/DL-in-the-Science/">
      <span class="writing-item__date">Feb 2024</span>
      <div>
        <h3 class="writing-item__title">Evolutionary-scale protein structure with a language model</h3>
        <p class="writing-item__excerpt">
          A walkthrough of ESM-2 / ESMFold: why MSA-free folding matters, how attention echoes contact maps,
          and what the Metagenomic Atlas unlocked — with original animated figures.
        </p>
      </div>
    </a>
  </div>
</section>

<section class="section section--band" id="contact">
  <div class="section__inner contact-block">
    <div class="reveal">
      <p class="section__label">Contact</p>
      <h2 class="section__title">Let’s build something exacting</h2>
      <p class="section__lede" style="margin-bottom:0">
        Open to ML engineering, applied research engineering, and NLP / LLM roles.
        Prefer concrete problems over vague “AI transformation.”
      </p>
      <ul class="contact-links">
        <li><a href="mailto:lekevinvan@gmail.com">lekevinvan@gmail.com</a></li>
        <li><a href="https://github.com/Meloneneis" target="_blank" rel="noopener">GitHub</a></li>
        <li><a href="https://huggingface.co/meloneneneis" target="_blank" rel="noopener">Hugging Face</a></li>
        <li><a href="/cv/">CV</a></li>
      </ul>
    </div>
    <div class="reveal">
      <a class="btn btn--ink" href="mailto:lekevinvan@gmail.com">Start a conversation</a>
    </div>
  </div>
</section>

<footer class="p-footer">
  <span>© <span id="year-stamp">2026</span> Kevin Van Le</span>
  <span>Built for clarity · Hosted on GitHub Pages</span>
</footer>
