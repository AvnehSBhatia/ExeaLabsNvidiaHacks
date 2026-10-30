// Fills every number on the page from the exported data. Nothing here is typed in by hand except wording.
import { FAMILIES, frameUrl } from './data.js';
import { fmt } from './util.js';

const LABEL = { open_drawer: 'Open drawer', open_door: 'Open cabinet door', open_fridge: 'Open fridge', put_in_drawer: 'Put in drawer', wipe_counter: 'Wipe counter', counter_to_sink: 'Counter to sink' };
export const familyLabel = (f) => LABEL[f] || f;

const FORMATS = {
  int: (v) => fmt.n(v),
  len: (v) => (Array.isArray(v) ? v.length : Object.keys(v || {}).length),
  keys: (v) => Object.keys(v || {}).length + ' kinds',
  hz: (v) => `${fmt.n(v)} Hz`,
  okword: (v) => (v ? 'ok' : 'not ok'),
  source: (v) => (v === 'authored_sample' ? 'authored sample' : v === 'roomplan' ? 'iPhone RoomPlan' : String(v)),
  bounds: (v) => `${(v.max[0] - v.min[0]).toFixed(1)} x ${(v.max[1] - v.min[1]).toFixed(1)} x ${(v.max[2] - v.min[2]).toFixed(1)} m`,
  dims: (v) => `${v.state_dim} / ${v.action_dim}`,
  cams: (v) => `${v.cameras.join(' + ')} at ${v.resolution.join('x')}`,
  res: (v) => v.join('×'),
};

function resolve(root, path) { return path.split('.').reduce((o, k) => (o == null ? o : o[k]), root); }

export function bindData(data) {
  const roots = { rooms: data.rooms, stats: data.stats, scan: { n: data.scan.n, ...data.scan.meta } };
  document.querySelectorAll('[data-bind]').forEach((el) => {
    const [path, f] = el.dataset.bind.split('|');
    let v = resolve(roots, path);
    if (v === undefined) return;
    if (f && f.startsWith('count:')) { const kind = f.slice(6); v = v.filter((p) => p.kind === kind).length; }
    else if (f && FORMATS[f]) v = FORMATS[f](v);
    el.textContent = String(v);
  });
}

export function renderEvaluate(stats) {
  const ev = stats.eval, rows = document.getElementById('evalRows'), total = document.getElementById('evalTotal');
  if (!rows || !total) return;                       // the results section shows the headline numbers instead
  rows.innerHTML = '';
  const pct = (x) => (x * 100).toFixed(3) + '%';
  const ci = (lo, hi, pt) => `<div class="axis" role="img" aria-label="95% Wilson interval ${lo.toFixed(2)} to ${hi.toFixed(2)}, observed ${pt.toFixed(2)}"><span class="axis__ci" style="left:${pct(lo)};width:${pct(hi - lo)}"><b></b></span><span class="axis__pt" style="left:${pct(pt)}"></span></div>`;
  for (const f of FAMILIES) {
    const e = ev.families[f]; if (!e) continue;
    const arm = e.arm_contact_episodes ? ` · arm touched furniture ${(e.arm_contact_max_depth_m * 1000).toFixed(1)} mm` : '';
    const row = document.createElement('div');
    row.className = 'erow';
    row.innerHTML = `<div class="erow__name">${familyLabel(f)}<small>${stats.families[f].instruction}</small></div>${ci(e.wilson_95[0], e.wilson_95[1], e.success_rate)}
      <div class="erow__fig"><strong>${e.successes} / ${e.episodes}</strong><small>closest person ${e.min_human_distance_m.toFixed(2)} m · ${e.human_contacts} contacts${arm}</small></div>`;
    rows.appendChild(row);
  }
  const t = ev.totals, per = ev.episodes_per_family;
  // the lead is built from the data, so "every episode succeeded" and "one per family" can never drift from stats.json
  const lead = document.getElementById('evLead');
  if (lead) lead.innerHTML = `The evaluation harness ran <b>${per}</b> ${per === 1 ? 'episode' : 'episodes'} per task family with the <b>${ev.policy}</b> expert and <b>${ev.people}</b> people walking. ${t.successes === t.episodes ? 'Every episode succeeded.' : `${t.successes} of ${t.episodes} episodes succeeded.`} ${per === 1 ? 'One episode per family shows that the pipeline runs and says nothing about a success rate.' : `${per} episodes per family still give wide intervals.`}`;
  total.innerHTML = `<div class="erow__name">All families<small>pooled, still only ${t.episodes} runs</small></div>${ci(t.wilson_95[0], t.wilson_95[1], t.success_rate)}
    <div class="erow__fig"><strong>${t.successes} / ${t.episodes}</strong><small>${t.human_contacts} human contacts · ${t.arm_contact_episodes} runs touched furniture</small></div>`;

}

export function renderFallback(stats) {
  const grid = document.getElementById('fallbackGrid'); grid.innerHTML = '';
  for (const f of FAMILIES) {
    const fig = document.createElement('figure');
    fig.innerHTML = `<div>${['ego_view_mid', 'ego_view_end', 'wrist_mid', 'wrist_end'].map((k) => { const i = k.lastIndexOf('_'); return `<img loading="lazy" width="128" height="128" alt="${k.replace(/_/g, ' ')} frame of ${familyLabel(f)}" src="${frameUrl(f, k.slice(0, i), k.slice(i + 1))}">`; }).join('')}</div>
      <figcaption>${stats.families[f].instruction}<small>${familyLabel(f)} · ${stats.families[f].duration_s} s · ${stats.families[f].success ? 'success' : 'failed'}</small></figcaption>`;
    grid.appendChild(fig);
  }
}
