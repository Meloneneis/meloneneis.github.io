# Kevin Van Le — portfolio

Astro site for [meloneneis.github.io](https://meloneneis.github.io).

Design & architecture: [`docs/PORTFOLIO_PLAN.md`](docs/PORTFOLIO_PLAN.md) (APPROVED after critique loops).

## Develop

```bash
npm ci
npm run dev
```

## Build

```bash
npm run build
node scripts/check-redirects.mjs
npm run preview
```

## Deploy

GitHub Actions → Pages (`deploy-pages`).  
Cutover: set Pages source to **GitHub Actions**, then merge to `master`.

Legacy Jekyll academicpages is quarantined in `_legacy/` (not deployed).
