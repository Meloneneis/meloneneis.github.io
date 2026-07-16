# PLAN — Kevin Van Le Portfolio (FAANG-grade rebuild)

> Status: **APPROVED v4** — visual · IA · tech all greenlit. Build authorized.  
> Owner: Cloud agent · Customer: Kevin Van Le  

---

## 0. Critique log

### Round 1 → v2
Fixed: Geist/zinc, contrast, brand/headline split, particles, light-SaaS, proof strip, chips, fade-ups, hero copy, tiers, CV shape, writing bridge, single deploy-pages path, redirect stubs, Jekyll kill intent, SVG-first.

### Round 2 → v3
| Item | Fix |
|------|-----|
| Delete forbidden terracotta token | Removed entirely from palette |
| Cream-adjacent `#F7F6F3` | Cool instrument paper `#F2F2F0` |
| Mint `--stage-accent` too loud | Desaturated `#8AA99A` |
| `font-display: optional` on brand | `swap` + `size-adjust` fallback |
| About copy unlocked | Locked 3 paragraphs |
| CV Education lonely heading | Fold into Background |
| Cutover stranding window | Non-negotiable runbook in §6 |
| Incomplete Jekyll kill list | Full inventory |
| Workflow hardening | permissions, concurrency, LH gate |

---

## 1. Product goal

Staff-engineer product page: calm, expensive, specific.  
Jobs: identity &lt;5s → flagship case → supporting work → thinking (essay) → contact.

Success: brand test; WCAG AA; Lighthouse mobile Perf ≥ 90 (CI); no cards/chips/stats; no academic CSS; GitHub+HF+email within first scroll.

---

## 2. Design strategy

**Instrument-grade systems craft** (tool manual energy, not SaaS cosplay).  
Metaphor via grid/hairlines/tabular meta — never neural graphs.  
Expensive = spacing + type fidelity + state contrast.

---

## 3. Visual system (locked v3)

### Color

Light:
- `--bg: #F2F2F0` (cool instrument paper — not cream `#F4F1EA`)
- `--bg-elev: #FAFAF8`
- `--bg-mute: #E8E8E4`
- `--fg: #14141A`
- `--fg-2: #3F3F4A`
- `--fg-3: #5C5C66` (≥4.5:1 on bg)
- `--line: rgba(20,20,26,0.14)` (work rows must boost to ≥0.28 on hover)
- `--accent: #1F4B3F` (sole accent — CTA fill + focus)
- `--accent-fg: #FAFAF8`

Stage (hero/footer):
- `--stage: #0E0F12`
- `--stage-fg: #F2F0EA`
- `--stage-fg-2: #B8B6AE`
- `--stage-line: rgba(242,240,234,0.14)`
- `--stage-accent: #8AA99A` (desaturated; links/ghost CTA on dark only)

**No second accent token exists.**

### Type
IBM Plex Sans 400/500/600 + IBM Plex Mono 500. Self-host latin woff2.  
Brand: clamp(56–88px)/0.95/600/−0.045em (loudest).  
Headline: clamp(18–24px)/1.35/500 (deck under brand — short wraps).  
H2: clamp(28–40px)/1.15/600. Body 16/1.65. Meta 12–13 mono.  
No uppercase eyebrow spam.  
Fonts: `font-display: swap` + `size-adjust` on fallback face; preload Sans 600+400 only above fold.

### Layout / components / motion / motif
As v2: 1080 max; ink hero; 7/5 flagship; hairline list; no cards/chips/stats; **2 motions** (optional grid pulse + row hover); **SVG grid motif only**; article 700px; figures aspect-boxed, stack on mobile.

---

## 4. IA & copy (IA APPROVED — locked)

**Hero**
- Brand: Kevin Van Le  
- Headline: Bachelor thesis on German code search — shipped as a public GraphCodeBERT model.  
- Lede: I build NLP and applied ML systems end-to-end: retrieval, evaluation, agents, and document understanding.  
- CTAs: View work · GitHub · Email  

**Order:** Hero → Flagship → Supporting/brief list → Writing → About → Contact  

**About (locked)**
1. I build NLP and applied ML systems — retrieval, evaluation, agents, and document pipelines — with a bias toward artifacts other people can run.  
2. My strongest public work is my bachelor thesis on German code search, released as [`graphcodebert-base-german`](https://huggingface.co/meloneneneis/graphcodebert-base-german) on Hugging Face.  
3. I’m looking for ML / NLP engineering roles where representation learning meets production constraints.

**Writing bridge:** “How I read a research system — ESMFold’s MSA-free bet — with original animated figures.”

**Work tiers:** flagship german-code-search; supporting semantic-plausibility + cv-agent; brief dqn-racing + glm-ocr (list labels include scope).

**CV:** Name + line + links; **Background** (thesis + HF, no lonely Education void); Selected projects; Skills as definition list; Writing link. Print CSS.

**Homepage flagship:** teaser only (~120 words + CTA to case). Full narrative on `/work/german-code-search/`.

---

## 5. Content inventory

Unchanged truthful sources (repos, HF, email, GIFs). Migrate `images/` → `public/images/` in cutover PR.

---

## 6. Technical architecture + cutover runbook

### Stack
Astro 5 at repo root · collections `work`, `writing` · vanilla CSS · `public/.nojekyll` · `site: https://meloneneis.github.io` · `trailingSlash: 'always'`.

### Deploy (only)
`.github/workflows/pages.yml`:
- `permissions: pages: write, id-token: write, contents: read`
- `concurrency: { group: pages, cancel-in-progress: false }`
- Node 22 from `.nvmrc`
- `npm ci` → `npm run build` → assert redirect stubs exist in `dist` → `upload-pages-artifact` → `deploy-pages`
- Lighthouse CI mobile on `/` must pass Perf ≥ 90 before artifact upload (or blocking follow-up job on PR)

### Redirect stubs (CI-enforced)
`/posts/2024/02/DL-in-the-Science/` → `/writing/esmfold/`  
`/portfolio/` + each old slug → matching `/work/.../` or `/#work`  
`/year-archive/` → `/writing/`  
`/resume` **and** `/resume/` → `/cv/`  
`/about/` → `/`  
Also emit `/sitemap.xml` (Astro) ; optional `/feed.xml` stub or Atom from content.

### Jekyll kill inventory (move to `_legacy/` or delete in cutover PR)
`Gemfile`, `Gemfile.lock`, `_config.yml`, `_config.dev.yml`, `_layouts/`, `_includes/`, `_sass/`, `_pages/`, `_data/`, `_posts/`, `_portfolio/`, `_publications/`, `_talks/`, `_teaching/`, `_drafts/`, `assets/` (theme), `talkmap*`, `markdown_generator/`, old theme `package.json` scripts, `vendor/` if present.  
**Keep** migrating content out first. **Do not** leave publishable Jekyll entrypoints on `master` after cutover.

### NON-NEGOTIABLE CUTOVER RUNBOOK (single unambiguous order)

**Principle:** Prefer a short “waiting for first Actions deploy” window over a broken Jekyll rebuild. GitHub Pages keeps serving the **last successful** deployment until a new one succeeds.

1. Develop on branch until `astro build` + stub check + **Lighthouse on local `astro preview` / `dist`** ≥ 90 mobile.  
2. Open **one** replacement PR: Astro + workflow + redirects + `public/images` migration + Jekyll → `_legacy/` (or delete) + new `package.json` + `.nvmrc` + `_site/` in `.gitignore`.  
3. **Before merge:** GitHub → Settings → Pages → Source → **GitHub Actions only** (disable branch / `/docs` publish). Site continues serving the previous Jekyll build until the first Actions artifact deploys.  
4. Merge PR to `master` (triggers workflow).  
5. Wait for `deploy-pages` green (same session; do not walk away).  
6. Smoke: `/`, `/cv/`, `/writing/esmfold/`, `/work/german-code-search/`, `/posts/2024/02/DL-in-the-Science/`, `/portfolio/german-code-search/`, `/year-archive/`, `/resume/`.  
7. Network panel: zero academicpages `main.css`.  
8. Rollback: redeploy prior Actions artifact from the Actions UI; do **not** flip back to branch deploy unless restoring `_legacy` Jekyll intentionally.

**Concurrency:** `cancel-in-progress: false` on the production `pages` group so a follow-up push cannot cancel the first cutover deploy.

**Do not** flip back to “Deploy from branch” after Jekyll entrypoints are gone — that strands the site.

---

## 7. Build sequence (on feature branch; single merge)

1. Scaffold Astro + tokens + fonts + layout  
2. Redirect stubs + stub CI script  
3. Homepage  
4. Work pages  
5. Writing + figures  
6. CV  
7. SVG motif  
8. LH local gate  
9. Quarantine Jekyll in same PR  
10. Flip Pages → Actions → merge → smoke  

---

## 8. Anti-patterns

Geist/zinc; purple/cyan neon; cream+terracotta; particles; blend nav; fade-up reveals; cards/chips/stat strips; fake employers; WebGL v1; second accent token.

---

## 9. Gate

Round 3 critique on visual + tech must return **APPROVED** (no blockers). Then implement immediately and ship.
