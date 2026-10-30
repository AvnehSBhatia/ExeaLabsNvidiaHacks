// Page chrome that must work on every path, with or without WebGL: the theme toggle, the "back to the plan" button and the
// sticky spec sheet that follows the active pipeline step. main.js imports this before it tries to start the stage.
import { safeStorage, $, $$ } from './util.js';

export function readTheme() {
  const cs = getComputedStyle(document.documentElement), v = (k) => cs.getPropertyValue(k).trim();
  const paper = v('--paper');
  const m = /^#?([0-9a-f]{6})$/i.exec(paper);
  const n = m ? parseInt(m[1], 16) : 0xefe9dc;
  const lum = (((n >> 16) & 255) * 0.299 + ((n >> 8) & 255) * 0.587 + (n & 255) * 0.114) / 255;
  return { paper, ink: v('--ink'), signal: v('--signal'), dark: lum < 0.5 };
}

// onChange(theme) is called synchronously after the DOM flips, so the WebGL clear colour changes in the same frame.
export function initTheme(onChange = () => {}) {
  const html = document.documentElement, btn = $('#themeBtn');
  const isNight = () => html.dataset.theme === 'night' || (html.dataset.theme === 'auto' && matchMedia('(prefers-color-scheme: dark)').matches);
  const sync = () => {
    onChange(readTheme());
    btn.setAttribute('aria-pressed', String(isNight()));
    btn.setAttribute('aria-label', isNight() ? 'Switch to day plan' : 'Switch to night plan');
  };
  let fade = 0;
  btn.addEventListener('click', () => {
    html.classList.add('theme-fade');                       // ease the colour change instead of a hard flip
    html.dataset.theme = isNight() ? 'day' : 'night';
    safeStorage.set('arrival-theme', html.dataset.theme);
    sync();
    clearTimeout(fade); fade = setTimeout(() => html.classList.remove('theme-fade'), 400);
  });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { if (html.dataset.theme === 'auto') sync(); });
  sync();
  return sync;
}

export function initToTop(goTo) {
  $('#toTop').addEventListener('click', () => {
    goTo(0);
    const top = $('#top'); top.setAttribute('tabindex', '-1'); top.focus({ preventScroll: true });
  });
}

// Sticky-scroll-reveal: the spec sheet on the right follows whichever step crosses a thin band at 58% of the viewport.
export function initSteps() {
  const steps = $$('.step'), panels = $$('.spec__panel'), FILES = ['scan', 'rooms.json', 'mujoco', 'expert', 'lerobot v2.1', 'loader check'];
  const activate = (i) => {
    steps.forEach((s, k) => s.classList.toggle('is-active', k === i));
    panels.forEach((p, k) => p.classList.toggle('is-active', k === i));
    $('#specStep').textContent = `Step ${i + 1}`; $('#specFile').textContent = FILES[i];
  };
  activate(0);
  if (!('IntersectionObserver' in window)) return;
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) if (e.isIntersecting) activate(steps.indexOf(e.target));
  }, { rootMargin: '-57% 0px -42% 0px', threshold: 0 });
  steps.forEach((s) => io.observe(s));
}

// The film: every [data-film] button opens it in a dialog; closing it (button, Esc or a click outside) pauses it.
export function initFilm() {
  const dlg = $('#film'), video = $('#filmVideo'), close = $('#filmClose');
  if (!dlg || !video) return;
  const open = () => {
    if (typeof dlg.showModal === 'function') dlg.showModal(); else dlg.setAttribute('open', '');
    video.play().catch(() => {});                       // a browser that blocks autoplay leaves the controls to the viewer
  };
  document.querySelectorAll('[data-film]').forEach((b) => b.addEventListener('click', open));
  close.addEventListener('click', () => dlg.close());
  dlg.addEventListener('click', (e) => { if (e.target === dlg) dlg.close(); });
  dlg.addEventListener('close', () => video.pause());
}
