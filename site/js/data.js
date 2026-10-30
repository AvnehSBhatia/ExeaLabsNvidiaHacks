// Loads the exported demo data and turns it into typed arrays once, so the render loop never parses JSON.
const base = new URL('../data/', import.meta.url);
const j = async (p) => { const r = await fetch(new URL(p, base)); if (!r.ok) throw new Error(`${p}: ${r.status}`); return r.json(); };

export const FAMILIES = ['open_drawer', 'open_door', 'open_fridge', 'put_in_drawer', 'wipe_counter', 'counter_to_sink'];

export const frameUrl = (family, cam, which) => new URL(`frames/${family}_${cam}_${which}.png`, base).href;

function parseScan(buf, meta) {
  const n = meta.count, dv = new DataView(buf);
  const pos = new Float32Array(n * 3), col = new Uint8Array(n * 3);
  for (let i = 0; i < n; i++) {
    const o = i * 9;
    pos[i * 3] = meta.origin[0] + dv.getUint16(o, true) * meta.scale;
    pos[i * 3 + 1] = meta.origin[1] + dv.getUint16(o + 2, true) * meta.scale;
    pos[i * 3 + 2] = meta.origin[2] + dv.getUint16(o + 4, true) * meta.scale;
    col[i * 3] = dv.getUint8(o + 6); col[i * 3 + 1] = dv.getUint8(o + 7); col[i * 3 + 2] = dv.getUint8(o + 8);
  }
  return { n, pos, col, meta };
}

export function parseEpisode(raw) {
  const nDyn = raw.nodes.dynamic.length, n = raw.frames.length, stride = nDyn * 7;
  const flat = new Float32Array(n * stride);
  for (let i = 0; i < n; i++) flat.set(raw.frames[i], i * stride);
  const joints = new Float32Array(n * 8);
  for (let i = 0; i < n; i++) joints.set(raw.joint_states[i], i * 8);
  const ep = { ...raw, flat, joints, nDyn, stride, n };
  delete ep.frames; delete ep.joint_states;
  return ep;
}

// Episodes are cached as promises, so a double click or a prefetch never fetches the same file twice and an episode can be
// requested before the core data has finished. A failed fetch is dropped from the cache so a later click can retry.
const episodeCache = {};
export function loadEpisode(family) {
  if (!episodeCache[family]) episodeCache[family] = j(`episodes/${family}.json`).then(parseEpisode).catch((err) => { delete episodeCache[family]; throw err; });
  return episodeCache[family];
}

export async function loadCore(step) {
  const out = {};
  const tasks = [
    ['rooms', () => j('rooms.json')],
    ['geoms', () => j('geoms.json')],
    ['stats', () => j('stats.json')],
    ['scan', async () => {
      const meta = await j('scan_points.json');
      const r = await fetch(new URL('scan_points.bin', base));
      if (!r.ok) throw new Error('scan_points.bin');
      return parseScan(await r.arrayBuffer(), meta);
    }],
  ];
  await Promise.all(tasks.map(async ([k, f]) => { out[k] = await f(); step?.(k); }));
  out.episodes = episodeCache;
  out.loadEpisode = loadEpisode;
  out.prefetchEpisodes = () => {
    const go = () => { for (const f of FAMILIES) loadEpisode(f).catch(() => {}); };
    if ('requestIdleCallback' in window) requestIdleCallback(go, { timeout: 4000 }); else setTimeout(go, 1500);
  };
  return out;
}
