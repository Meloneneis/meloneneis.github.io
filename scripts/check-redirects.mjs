import { existsSync } from 'node:fs';
import { join } from 'node:path';

const dist = join(process.cwd(), 'dist');
const required = [
  'posts/2024/02/DL-in-the-Science/index.html',
  'portfolio/index.html',
  'portfolio/german-code-search/index.html',
  'portfolio/semantic-plausibility/index.html',
  'portfolio/cv-agent/index.html',
  'portfolio/dqn-racing/index.html',
  'portfolio/glm-ocr/index.html',
  'year-archive/index.html',
  'resume/index.html',
  'about/index.html',
  'index.html',
  'cv/index.html',
  'writing/esmfold/index.html',
  'work/german-code-search/index.html',
];

let failed = false;
for (const rel of required) {
  const p = join(dist, rel);
  if (!existsSync(p)) {
    console.error('Missing:', rel);
    failed = true;
  }
}
if (failed) {
  process.exit(1);
}
console.log(`OK — ${required.length} paths present in dist/`);
