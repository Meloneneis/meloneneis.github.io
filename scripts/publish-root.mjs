#!/usr/bin/env node
/**
 * Publish Astro dist/ into repo root for branch-based GitHub Pages
 * (when Actions deploy is not yet enabled).
 */
import { cpSync, mkdirSync, rmSync, existsSync, readdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const root = process.cwd();
const dist = join(root, 'dist');
if (!existsSync(dist)) {
  console.error('dist/ missing — run npm run build first');
  process.exit(1);
}

const publishDirs = [
  '_astro',
  'work',
  'writing',
  'cv',
  'posts',
  'portfolio',
  'year-archive',
  'resume',
  'about',
  'fonts',
  'images',
];

for (const d of publishDirs) {
  const target = join(root, d);
  if (existsSync(target)) rmSync(target, { recursive: true, force: true });
}

for (const entry of readdirSync(dist, { withFileTypes: true })) {
  const from = join(dist, entry.name);
  const to = join(root, entry.name);
  if (['src', 'node_modules', '_legacy', 'docs', '.git', '.github'].includes(entry.name)) continue;
  if (entry.isDirectory()) {
    if (existsSync(to)) rmSync(to, { recursive: true, force: true });
    cpSync(from, to, { recursive: true });
  } else if (entry.name === 'index.html' || entry.name === '404.html' || entry.name === 'favicon.svg' || entry.name === 'favicon.ico' || entry.name === '.nojekyll') {
    cpSync(from, to);
  }
}

writeFileSync(join(root, '.nojekyll'), '');
console.log('Published dist/ → repo root for GitHub Pages branch deploy');
