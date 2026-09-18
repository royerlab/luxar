import { chromium } from '@playwright/test';
const OUT = '/Users/loic.royer/.claude/jobs/9b9f99f0/tmp/cards';
const b = await chromium.launch();
const p = await b.newPage();
await p.goto(`file://${OUT}/cards.html`, { waitUntil: 'load', timeout: 300000 });
await p.waitForTimeout(4000);
await p.pdf({
  path: `${OUT}/protein_universe_stories.pdf`,
  format: 'A4',
  printBackground: true,
  margin: { top: '14mm', bottom: '14mm', left: '13mm', right: '13mm' },
});
console.log('pdf written');
await b.close();
