// Boot: load the exported data, build the stage, wire the UI and the scroll layer, then lift the loading sheet.
import { loadCore, loadEpisode } from './data.js';
import { bindData, renderEvaluate, renderFallback } from './sections.js';
import { initTilt } from './tilt.js';
import { initTheme, initToTop, initSteps, initFilm } from './chrome.js';
import { initMotion } from './motion.js';
import { prefersReducedMotion, isCoarse, watchMedia, $ } from './util.js';

const TOTAL_STEPS = 7;
let done = 0;
const loaderFill = $('#loaderFill'), loaderMsg = $('#loaderMsg'), loader = $('#loader'), status = $('#status');
const step = () => { done++; loaderFill.style.transform = `scaleX(${Math.min(done / TOTAL_STEPS, 1)})`; loaderMsg.textContent = done >= TOTAL_STEPS ? 'Ready' : 'Loading the kitchen'; };
const announce = (() => { let t = 0; return (msg) => { clearTimeout(t); t = setTimeout(() => { status.textContent = msg; }, 250); }; })();
const html = document.documentElement;

// The plain page: recorded camera frames instead of the live stage. Also what a lost context falls back to if it never returns.
function showFallback(data) {
  html.classList.add('no-webgl');
  $('#fallback').hidden = false; renderFallback(data.stats);
  $('#journey').style.height = 'auto';
  loaderFill.style.transform = 'scaleX(1)'; loader.classList.add('is-done');
  window.ScrollTrigger?.refresh();
}

async function boot() {
  const reduced = prefersReducedMotion(), coarse = isCoarse();
  // chrome that has to work on every path: theme toggle, spec sheet that follows the pipeline steps
  let stage = null;
  initFilm();
  const motion = initMotion({ reduced, coarse });
  const theme = initTheme((t) => { stage?.setTheme(t); });
  initSteps();

  // all four faces the page uses above the fold, raced against a timeout so a slow font never holds the loader
  const faces = ['300 48px Fraunces', 'italic 340 48px Fraunces', '400 12px "DM Mono"', '500 12px "DM Mono"'];
  const fonts = Promise.race([Promise.all(faces.map((f) => document.fonts.load(f))), new Promise((r) => setTimeout(r, 1500))]).then(step);
  // the first episode downloads in parallel with the core data instead of after it
  const first = 'open_drawer';
  const firstEp = loadEpisode(first);
  const data = await loadCore(step);
  await fonts;
  await firstEp; step();

  bindData(data);
  renderEvaluate(data.stats);

  try {
    const { Stage } = await import('./stage.js');
    stage = new Stage({ canvas: $('#gl'), wrap: $('#stage'), data, reduced, coarse });
  } catch (err) {
    console.warn('WebGL stage unavailable, showing the recorded frames instead:', err);
    showFallback(data);
    initToTop((y) => scrollTo({ top: y, behavior: reduced ? 'auto' : 'smooth' }));
    initTilt(document, { reduced, coarse });
    motion.loaderDone();
    return;
  }
  step();
  watchMedia('(prefers-reduced-motion: reduce)', (v) => { stage.reduced = v; });
  const { initUI } = await import('./ui.js');
  const { initCompare } = await import('./compare.js');
  const { initScroll } = await import('./scroll.js');
  const ui = initUI({ stage, data, announce });
  theme();                                                         // paint the stage in the current theme
  initTilt(document, { reduced, coarse });                         // after initUI: the task cards are created there
  await ui.selectTask(first, { silent: true });
  const compare = initCompare({ stage, reduced });
  const scroll = initScroll({ stage, ui, reduced, coarse, announce, compare, motion });
  initToTop((y) => scroll.goTo(y, 'cross'));
  window.arrival = { stage, ui, scroll, data };      // handle for debugging and automated checks

  // first frame, then lift the loading sheet
  stage.resize(); stage.tick(0.016);
  step();
  await motion.introHold();
  requestAnimationFrame(() => { loader.classList.add('is-done'); motion.loaderDone(); });
  data.prefetchEpisodes();                           // the other five tasks load while the reader reads, so a click is instant

  // a lost WebGL context pauses the stage and says so; three restores its own state when the browser gives the context back
  const note = $('#glNote');
  let giveUp = 0;
  stage.canvas.addEventListener('webglcontextlost', (e) => {
    e.preventDefault(); stage.lost = true; note.hidden = false;
    giveUp = setTimeout(() => { if (stage.lost) { note.hidden = true; showFallback(data); } }, 5000);   // it did not come back
  });
  stage.canvas.addEventListener('webglcontextrestored', () => {
    clearTimeout(giveUp); stage.lost = false; stage.poseDirty = true; stage._feedDirty = true; note.hidden = true; stage.resize();
  });
}

boot().catch((err) => { console.error(err); loaderMsg.textContent = 'Could not open the survey data. Reload the page.'; });
