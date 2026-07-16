# Kevin Van Le — Portfolio site

Personal GitHub Pages site for **Kevin Van Le** (`meloneneis.github.io`).

Built on a Jekyll academicpages base, with a custom portfolio homepage:

- Interactive **Three.js protein-backbone** hero
- Case studies for thesis + applied ML projects
- Blog post motion (progress bar + figure reveals)
- Design system documented in `.cursor/rules/portfolio-design.mdc`

## Local development

```bash
bundle install
bundle exec jekyll serve
```

Open `http://localhost:4000`.

## Key paths

| Path | Role |
|------|------|
| `_pages/about.md` | Homepage (portfolio layout) |
| `_layouts/portfolio.html` | Homepage shell |
| `_layouts/portfolio-case.html` | Project case studies |
| `assets/css/portfolio.css` | Design tokens + layout |
| `assets/js/protein-scene.js` | 3D hero |
| `assets/js/portfolio.js` | Scroll reveals / counters |
| `assets/js/blog-motion.js` | Blog animations |
| `_portfolio/` | Case study markdown |
| `_posts/` | Blog |

## Design intent

Archival lab / structural reading — cool stone paper, prussian stage, teal + oxide accents, Newsreader + Source Sans 3 + IBM Plex Mono. Avoid purple neon “AI portfolio” tropes.
