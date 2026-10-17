"""Twin JSON -> MJCF: walls, floor, counters with drawers, hinged cabinet doors and upper cabinets, sink basin
with a faucet lever, stove with an oven door, fridge, dishwasher and microwave with hinged doors, free-body
objects, people, Panda-like arm, cameras. Every moving part is a named joint with a handle site (part_names)."""

from __future__ import annotations

import math

from arrival.catalog.objects import object_spec
from arrival.sim import people, robot
from arrival.twin.compile import to_local

LIGHT = {"dim": 0.35, "neutral": 0.7, "bright": 1.0}
TOP_T = 0.04
CARCASS_H = 0.70  # cabinet body below the counter top (sink basin sits between this and the top)
DRAWER = 'contype="2" conaffinity="2"'  # moving parts collide with the robot, not with their carcass
APPLIANCES = ("fridge", "stove", "dishwasher")  # floor-standing: counters are trimmed where they run into one
VIS = 'contype="0" conaffinity="0" group="2"'
FRONT, GAP = "0.70 0.55 0.40 1", "0.10 0.09 0.08 1"
STEEL, DARK, SPLASH = "0.75 0.77 0.80 1", "0.08 0.08 0.09 1", "0.88 0.86 0.81 1"
REVEAL, INSET, FRONT_T, KICK = 0.003, 0.015, 0.018, 0.10
UPPER_Z0, UPPER_H, UPPER_D, UPPER_DOOR_MAX = 1.56, 0.60, 0.32, 0.60  # wall cabinets over a counter run; the wrist
# reaches 1.50 m over the back of the counter when it drops into the sink, so the cabinets start above that
DOOR_FLAT = 1.5  # bottom-hinged doors (oven, dishwasher) stop just short of horizontal


def _box(name, pos, size, rgba, extra=""):
    label = f'name="{name}" ' if name else ""
    return (f'<geom {label}type="box" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" '
            f'size="{size[0] / 2:.4f} {size[1] / 2:.4f} {size[2] / 2:.4f}" rgba="{rgba}" {extra}/>')


def _vbox(pos, size, rgba):
    return _box(None, pos, size, rgba, VIS)


def _vr(x0, x1, y0, y1, z0, z1, rgba, o=(0.0, 0.0, 0.0)):
    """Visual box from coordinate ranges, expressed relative to origin o."""
    return _vbox(((x0 + x1) / 2 - o[0], (y0 + y1) / 2 - o[1], (z0 + z1) / 2 - o[2]),
                 (x1 - x0, y1 - y0, z1 - z0), rgba)


def _vcyl(rgba, r, a, b, kind="cylinder"):
    f = " ".join(f"{v:.4f}" for v in (*a, *b))
    return f'<geom type="{kind}" fromto="{f}" size="{r}" rgba="{rgba}" {VIS}/>'


def _sinks_in(counter: dict, sinks: list[dict]) -> list[dict]:
    (cx, cy, _), (sx, sy, _) = counter["position"], counter["size"]
    return [s for s in sinks if abs(s["position"][0] - cx) < sx / 2 and abs(s["position"][1] - cy) < sy / 2]


def _span(c: dict, fixtures: list[dict]) -> tuple[float, float]:
    """Local x-extent of a counter, trimmed at an end that runs into a fridge/stove footprint.
    Appliances wholly inside the span are left alone (the twin data should not place them there; TWN-6)."""
    sx, sy = c["size"][0], c["size"][1]
    x0, x1 = -sx / 2, sx / 2
    for f in fixtures:
        if not f["type"].startswith(APPLIANCES):
            continue
        hx, hy = f["size"][0] / 2, f["size"][1] / 2
        yaw = f.get("yaw", 0.0)
        co, si = math.cos(yaw), math.sin(yaw)
        pts = [to_local((f["position"][0] + co * dx - si * dy, f["position"][1] + si * dx + co * dy), c)
               for dx in (-hx, hx) for dy in (-hy, hy)]
        ax0, ax1 = min(p[0] for p in pts), max(p[0] for p in pts)
        ay0, ay1 = min(p[1] for p in pts), max(p[1] for p in pts)
        if ay1 <= -sy / 2 or ay0 >= sy / 2:
            continue
        if ax0 <= x0 < ax1:
            x0 = ax1
        elif ax0 < x1 <= ax1:
            x1 = ax0
    return x0, x1


def drawer_x(c: dict, fixtures: list[dict]) -> float:
    """Local x of the drawer stack: left quarter, or the middle if a fridge/stove overlaps the left quarter."""
    x = c["position"][0] - c["size"][0] / 4
    appl = [f for f in fixtures if f["type"].startswith(APPLIANCES)]
    if any(abs(f["position"][0] - x) < f["size"][0] / 2 + DRAWER_W / 2 for f in appl):
        return 0.0
    return -c["size"][0] / 4


DRAWER_W = 0.5


def door_w(c: dict, fixtures: list[dict]) -> float:
    """Cabinet door width: right half of the counter, shrunk so it never covers the drawer stack."""
    sx = c["size"][0]
    left = max(0.02, drawer_x(c, fixtures) + DRAWER_W / 2 + 0.02)
    return sx / 2 - 0.02 - left


def _sink_visuals(hx, hw, hd, sy, h) -> list[str]:
    """Stainless rim lip, drain and faucet for a sink hole centred at local x=hx."""
    g, lip, w = [], 0.003, 0.03
    xa, xb = hx - hw / 2 - (w - 0.01), hx + hw / 2 + (w - 0.01)
    for y0, y1 in ((hd / 2 - 0.01, hd / 2 + w - 0.01), (-hd / 2 - w + 0.01, -hd / 2 + 0.01)):
        g.append(_vr(xa, xb, y0, y1, h, h + lip, STEEL))
    for x0, x1 in ((xa, hx - hw / 2 + 0.01), (hx + hw / 2 - 0.01, xb)):
        g.append(_vr(x0, x1, -hd / 2 + 0.01, hd / 2 - 0.01, h, h + lip, STEEL))
    zb = CARCASS_H + 0.02
    g.append(_vcyl(DARK, 0.035, (hx, 0, zb + 0.0005), (hx, 0, zb + 0.0015)))
    g.append(_vcyl(STEEL, 0.025, (hx, 0, zb + 0.0025), (hx, 0, zb + 0.0035)))
    y0 = -sy / 2 + 0.04
    g.append(_vcyl(STEEL, 0.025, (hx, y0, h), (hx, y0, h + 0.02)))
    g.append(_vcyl(STEEL, 0.012, (hx, y0, h + 0.02), (hx, y0, h + 0.21)))
    g.append(_vcyl(STEEL, 0.011, (hx, y0, h + 0.205), (hx, y0 + 0.15, h + 0.205), "capsule"))
    g.append(_vcyl(STEEL, 0.011, (hx, y0 + 0.15, h + 0.205), (hx, y0 + 0.15, h + 0.175), "capsule"))
    return g


def _faucet_lever_xml(sid: str, hx: float, y0: float, h: float) -> str:
    """Single-lever mixer handle on the faucet column: hinge `{sid}_faucet`, q > 0 lifts the lever (water on)."""
    return (f'<body name="{sid}_faucet_lever" pos="{hx + 0.012:.4f} {y0:.4f} {h + 0.19:.4f}">'
            f'<joint name="{sid}_faucet" type="hinge" axis="0 -1 0" range="0 1.2" damping="0.05" frictionloss="0.05"/>'
            f'<inertial pos="0.025 0 0.005" mass="0.05" diaginertia="1e-5 1e-5 1e-5"/>'
            f'<geom name="{sid}_faucet_handle" type="capsule" size="0.006" fromto="0 0 0 0.05 0 0.01" rgba="{STEEL}" '
            f'{DRAWER}/><site name="{sid}_faucet_handle" pos="0.04 0 0.008" size="0.005"/></body>')


def upper_doors(c: dict, fixtures: list[dict] = ()) -> list[tuple[float, float, bool]]:
    """(x_left, x_right, hinge_on_left) per wall-cabinet door over a counter run. Tables, islands and low
    counters get none."""
    if c.get("default_config_name") == "dining" or c.get("island") or c["size"][2] < 0.8:
        return []
    x0, x1 = _span(c, list(fixtures))
    x0, x1 = x0 + 0.03, x1 - 0.03  # a reveal at each end keeps the run inside the walls (wall boxes are 5 cm thick)
    n = max(1, math.ceil((x1 - x0) / UPPER_DOOR_MAX))
    w = (x1 - x0) / n
    return [(x0 + k * w, x0 + (k + 1) * w, k % 2 == 0) for k in range(n)]


def _upper_cabinet_xml(cid: str, c: dict, fixtures: list[dict]) -> str:
    doors = upper_doors(c, fixtures)
    if not doors:
        return ""
    sy = c["size"][1]
    yb, yf = -sy / 2 + 0.01, -sy / 2 + 0.01 + UPPER_D
    x0, x1 = doors[0][0], doors[-1][1]
    zc = UPPER_Z0 + UPPER_H / 2
    g = [_box(f"{cid}_upper", ((x0 + x1) / 2, (yb + yf - 0.02) / 2, zc), (x1 - x0, yf - 0.02 - yb, UPPER_H),
              "0.55 0.42 0.3 1")]
    for k, (xa, xb, left) in enumerate(doors):
        dw = xb - xa - 0.04 - REVEAL
        hx = xa + 0.02 if left else xb - 0.02
        sgn = 1.0 if left else -1.0
        px, hxh = sgn * dw / 2, sgn * (dw - 0.05)
        za, zb = -UPPER_H / 2 + 0.06, -UPPER_H / 2 + 0.20
        g.append(
            f'<body name="{cid}_upper_door{k}" pos="{hx:.4f} {yf - 0.01:.4f} {zc:.4f}">'
            f'<joint name="{cid}_upper_door{k}" type="hinge" axis="0 0 {sgn:.0f}" range="0 1.9" damping="2" '
            f'frictionloss="0.2"/><inertial pos="{px:.4f} 0 0" mass="1.5" diaginertia="0.03 0.03 0.03"/>'
            + _box(f"{cid}_upper_door{k}_panel", (px, 0.0, 0), (dw, 0.02, UPPER_H - 0.02), FRONT, DRAWER)
            + f'<geom name="{cid}_upper_door{k}_handle" type="capsule" size="0.012" fromto="{hxh:.4f} 0.05 {za:.3f} '
              f'{hxh:.4f} 0.05 {zb:.3f}" rgba="{STEEL}" {DRAWER}/>'
            + "".join(_vcyl(STEEL, 0.005, (hxh, 0.009, z), (hxh, 0.05, z)) for z in (za + 0.01, zb - 0.01))
            + f'<site name="{cid}_upper_door{k}_handle" pos="{hxh:.4f} 0.05 {(za + zb) / 2:.3f}" size="0.01"/></body>')
    return "".join(g)


def counter_xml(c: dict, sinks: list[dict], fixtures: list[dict] = ()) -> str:
    """Counter in its local frame (+y local = front). Top slab is split around any sink hole; the counter
    is trimmed where it runs into a fridge/stove. Visible fronts are recessed visual twins of the physical
    drawer/door panels (hidden, group 3) with small reveals, slim handles, backer, end panels and backsplash."""
    fixtures = list(fixtures)
    (cx, cy, _), (sx, sy, h) = c["position"], c["size"]
    cid, yaw = c["id"], c.get("yaw", 0.0)
    x0, x1 = _span(c, fixtures)
    xm, xw = (x0 + x1) / 2, x1 - x0
    g = [_box(f"{cid}_carcass", (xm, 0, CARCASS_H / 2), (xw, sy - 0.04, CARCASS_H), "0.55 0.42 0.3 1")]
    holes = [(s["position"][0] - cx, s["size"][0], s["size"][1], s) for s in _sinks_in(c, sinks)]
    zt = h - TOP_T / 2
    top = "0.82 0.82 0.8 1"
    if not holes:
        g.append(_box(f"{cid}_top", (xm, 0, zt), (xw, sy, TOP_T), top))
    for hx, hw, hd, s in holes[:1]:
        lx0, lx1 = x0, hx - hw / 2
        rx0, rx1 = hx + hw / 2, x1
        g.append(_box(f"{cid}_top_l", ((lx0 + lx1) / 2, 0, zt), (lx1 - lx0, sy, TOP_T), top))
        g.append(_box(f"{cid}_top_r", ((rx0 + rx1) / 2, 0, zt), (rx1 - rx0, sy, TOP_T), top))
        rim = (sy - hd) / 2
        g.append(_box(f"{cid}_top_f", (hx, sy / 2 - rim / 2, zt), (hw, rim, TOP_T), top))
        g.append(_box(f"{cid}_top_b", (hx, -sy / 2 + rim / 2, zt), (hw, rim, TOP_T), top))
        # basin: bottom + 4 thin walls, from carcass top to counter top
        bz0, bt = CARCASS_H + 0.01, 0.01
        wall_h = h - bz0
        steel = "0.6 0.62 0.65 1"
        sid = s["id"]
        g.append(_box(f"{sid}_bottom", (hx, 0, bz0), (hw, hd, 0.02), steel))
        for k, (px, py, wx, wy) in enumerate([(hx - hw / 2 + bt / 2, 0, bt, hd), (hx + hw / 2 - bt / 2, 0, bt, hd),
                                              (hx, hd / 2 - bt / 2, hw, bt), (hx, -hd / 2 + bt / 2, hw, bt)]):
            g.append(_box(f"{sid}_wall{k}", (px, py, bz0 + wall_h / 2), (wx, wy, wall_h), steel))
        g += _sink_visuals(hx, hw, hd, sy, h)
        g.append(_faucet_lever_xml(sid, hx, -sy / 2 + 0.04, h))
    # ---- visual shell: backer, end panels, fronts ----
    front = sy / 2 - 0.02
    yf, zf = sy / 2 - INSET, h - TOP_T
    xl = x0 if x0 > -sx / 2 + 1e-6 else x0 + FRONT_T
    xr = x1 if x1 < sx / 2 - 1e-6 else x1 - FRONT_T
    g.append(_vr(xl, xr, front, front + 0.002, 0, zf, GAP))
    if xl > x0:
        g.append(_vr(x0 - 0.001, xl, -sy / 2 + 0.021, yf, 0.001, zf, FRONT))
    if xr < x1:
        g.append(_vr(xr, x1 + 0.001, -sy / 2 + 0.021, yf, 0.001, zf, FRONT))
    g.append(_vr(x0, x1, -sy / 2 - 0.06, -sy / 2 + 0.012, h, h + 0.45, SPLASH))
    ya, yb, r2 = yf - FRONT_T, yf, REVEAL / 2
    zlo, zhi = KICK + REVEAL, zf - REVEAL
    dx = drawer_x(c, fixtures)
    da, db = max(dx - DRAWER_W / 2, xl), min(dx + DRAWER_W / 2, xr)
    dz0 = CARCASS_H - 0.01
    yb_bar = front + 0.07

    def bar_standoffs(x, za, zb, o=(0.0, 0.0, 0.0), face=yb):
        return "".join(
            _vcyl(STEEL, 0.005, (x - o[0], face - 0.001 - o[1], z - o[2]), (x - o[0], yb_bar - o[1], z - o[2]))
            for z in (za, zb))

    def vbar(x, za, zb):
        """Fixed-front handle: steel capsule on two standoffs at za, zb."""
        return (f'<geom type="capsule" fromto="{x:.4f} {yb_bar:.4f} {za - 0.01:.4f} {x:.4f} {yb_bar:.4f} '
                f'{zb + 0.01:.4f}" size="0.012" rgba="{STEEL}" {VIS}/>' + bar_standoffs(x, za, zb))

    g.append(_vr(da + r2, db - r2, ya, yb, KICK + REVEAL, KICK + 0.227, FRONT))
    g.append(vbar(dx, KICK + 0.08, KICK + 0.14))
    if da - xl > 0.03:
        g.append(_vr(xl + r2, da - r2, ya, yb, zlo, zhi, FRONT))
        if da - xl >= 0.25:
            g.append(vbar((xl + da) / 2, 0.46, 0.54))
    g.append(_upper_cabinet_xml(cid, c, fixtures))
    body = [f'<body name="{cid}" pos="{cx} {cy} 0" euler="0 0 {yaw}">' + "".join(g)]
    # drawers (slide along local +y) in the left half; hinged door cabinets in the right half
    hid = f'{DRAWER} group="3"'
    for k in range(2):
        dz = dz0 - k * 0.24  # top drawer sits in the band just under the counter top
        o = (dx, front, dz)
        z0, z1 = (0.600, zf - REVEAL) if k == 0 else (KICK + 0.23, 0.597)
        body.append(
            f'<body name="{cid}_drawer{k}" pos="{dx:.3f} {front:.3f} {dz:.3f}">'
            f'<joint name="{cid}_drawer{k}" type="slide" axis="0 1 0" range="0 0.4" damping="5" frictionloss="0.5"/>'
            f'<inertial pos="0 -0.2 0" mass="1.5" diaginertia="0.02 0.02 0.02"/>'
            + _box(f"{cid}_drawer{k}_front", (0, 0.01, 0), (DRAWER_W, 0.02, 0.18), "0.65 0.5 0.36 1", hid)
            + _box(f"{cid}_drawer{k}_tray", (0, -0.2, -0.07), (DRAWER_W - 0.04, 0.38, 0.02), "0.6 0.46 0.33 1", DRAWER)
            + f'<geom name="{cid}_drawer{k}_handle" type="capsule" size="0.012" fromto="0 0.07 -0.05 0 0.07 0.05" '
              f'rgba="{STEEL}" {DRAWER}/>'
            + _box(None, (0, 0.035, 0.04), (0.02, 0.07, 0.02), "0.3 0.3 0.3 1", hid)
            + _box(None, (0, 0.035, -0.04), (0.02, 0.07, 0.02), "0.3 0.3 0.3 1", hid)
            + _vr(da + r2, db - r2, ya, yb, z0, z1, FRONT, o)
            + bar_standoffs(dx, dz - 0.04, dz + 0.04, o)
            + f'<site name="{cid}_drawer{k}_handle" pos="0 0.07 0.0" size="0.01"/></body>')
    dw = door_w(c, fixtures)
    hinge = (sx / 2 - 0.02, front + 0.01, 0.36)
    hx_ = -dw + 0.06
    body.append(
        f'<body name="{cid}_door" pos="{hinge[0]:.3f} {hinge[1]:.3f} {hinge[2]}">'
        f'<joint name="{cid}_door" type="hinge" axis="0 0 -1" range="0 1.9" damping="2" frictionloss="0.2"/>'
        f'<inertial pos="{-dw / 2:.3f} 0 0" mass="2" diaginertia="0.05 0.05 0.05"/>'
        + _box(f"{cid}_door_panel", (-dw / 2, 0.01, 0), (dw, 0.02, 0.6), "0.65 0.5 0.36 1", hid)
        + f'<geom name="{cid}_door_handle" type="capsule" size="0.012" fromto="{hx_:.3f} 0.05 0.1 '
          f'{hx_:.3f} 0.05 0.25" rgba="{STEEL}" {DRAWER}/>'
        + _vr(db + r2, xr - r2, ya, yb, zlo, zf - REVEAL, FRONT, hinge)
        + "".join(_vcyl(STEEL, 0.005, (hx_, yb - hinge[1] - 0.001, z), (hx_, 0.05, z)) for z in (0.11, 0.24))
        + f'<site name="{cid}_door_handle" pos="{hx_:.3f} 0.05 0.17" size="0.01"/></body>')
    return "".join(body) + "</body>"


def stove_xml(f: dict) -> str:
    (cx, cy, _), (sx, sy, sz) = f["position"], f["size"]
    sid = f["id"]
    ctop, ys = sz + 0.02, sy / 2
    burners = [(-0.17, -0.13), (0.17, -0.13), (-0.17, 0.13), (0.17, 0.13)]
    g = [_box(f"{sid}_box", (0, 0, sz / 2), (sx, sy, sz), "0.15 0.15 0.16 1")]
    g += [_box(f"{sid}_burner{k}", (bx, by, sz + 0.003), (0.16, 0.16, 0.006), "0.35 0.1 0.08 1", 'group="3"')
          for k, (bx, by) in enumerate(burners)]
    g.append(_vr(-sx / 2, sx / 2, -ys, ys, sz, ctop, DARK))
    for bx, by in burners:
        g.append(_vcyl("0.45 0.46 0.48 1", 0.085, (bx, by, ctop + 0.0005), (bx, by, ctop + 0.0015)))
        g.append(_vcyl("0.02 0.02 0.02 1", 0.06, (bx, by, ctop + 0.0015), (bx, by, ctop + 0.0030)))
        g.append(_vcyl("0.45 0.46 0.48 1", 0.03, (bx, by, ctop + 0.0030), (bx, by, ctop + 0.0045)))
        g.append(_vbox((bx, by, ctop + 0.0045), (0.19, 0.012, 0.011), "0.06 0.06 0.065 1"))
        g.append(_vbox((bx, by, ctop + 0.0050), (0.012, 0.19, 0.012), "0.06 0.06 0.065 1"))
    g.append(_vr(-sx / 2, sx / 2, ys, ys + 0.012, sz - 0.10, sz - 0.02, "0.2 0.2 0.21 1"))
    for kx in (-0.27, -0.09, 0.09, 0.27):
        g.append(_vcyl(STEEL, 0.018, (kx, ys + 0.011, sz - 0.06), (kx, ys + 0.03, sz - 0.06)))
    g.append(_flap_door_xml(f"{sid}_oven_door", (0, ys + 0.016, 0.12), sx - 0.06, sz - 0.24, 6.0, "0.3 0.31 0.33 1",
                            window=(sx - 0.16, 0.12, sz - 0.44)))
    g.append(_vr(-sx / 2, sx / 2, ys, ys + 0.003, 0, 0.10, DARK))
    g.append(_vr(-sx / 2 - 0.05, sx / 2 + 0.05, -ys - 0.059, -ys + 0.010, ctop - 0.001, ctop + 0.448, SPLASH))
    return f'<body name="{sid}" pos="{cx} {cy} 0" euler="0 0 {f.get("yaw", 0)}">' + "".join(g) + "</body>"


def _flap_door_xml(name: str, pivot, width: float, height: float, mass: float, rgba: str, window=None,
                   handle_dz: float = 0.03) -> str:
    """Bottom-hinged door (oven, dishwasher) in the parent's frame: body `name`, joint `name`, handle bar and
    site `name_handle` near the top edge. q > 0 swings the top edge forward (+y) and down to DOOR_FLAT. The
    panel sits 8 mm behind the hinge line so gravity holds it shut at q = 0."""
    zh = height - handle_dz
    g = [f'<body name="{name}" pos="{pivot[0]:.4f} {pivot[1]:.4f} {pivot[2]:.4f}">'
         f'<joint name="{name}" type="hinge" axis="-1 0 0" range="0 {DOOR_FLAT}" damping="4" frictionloss="0.8"/>'
         f'<inertial pos="0 -0.008 {height / 2:.4f}" mass="{mass}" diaginertia="{mass / 20:.3f} {mass / 20:.3f} '
         f'{mass / 60:.3f}"/>'
         + _box(f"{name}_panel", (0, -0.0085, height / 2), (width, 0.015, height), rgba, DRAWER)]
    if window:
        ww, z0, z1 = window
        g.append(_vr(-ww / 2, ww / 2, 0.0, 0.002, z0, z1, DARK))
    for x in (-0.22, 0.22):
        g.append(_vcyl(STEEL, 0.005, (x, 0.0, zh), (x, 0.041, zh)))
    g.append(f'<geom name="{name}_handle" type="capsule" size="0.012" fromto="-0.26 0.041 {zh:.4f} 0.26 0.041 {zh:.4f}" '
             f'rgba="{STEEL}" {DRAWER}/><site name="{name}_handle" pos="0 0.041 {zh:.4f}" size="0.01"/></body>')
    return "".join(g)


def dishwasher_xml(f: dict) -> str:
    """Floor-standing dishwasher: carcass box and a bottom-hinged front door with the handle bar near the top."""
    (cx, cy, _), (sx, sy, h) = f["position"], f["size"]
    fid, ys = f["id"], sy / 2
    g = [_box(f"{fid}_box", (0, -0.01, h / 2), (sx, sy - 0.02, h), "0.55 0.56 0.58 1"),
         _vr(-sx / 2, sx / 2, ys - 0.012, ys + 0.004, h - 0.08, h - 0.01, "0.2 0.2 0.21 1"),
         _vr(-sx / 2, sx / 2, ys - 0.01, ys + 0.003, 0, 0.08, DARK),
         _flap_door_xml(f"{fid}_door", (0, ys + 0.006, 0.08), sx - 0.04, h - 0.18, 5.0, STEEL, handle_dz=0.05)]
    return f'<body name="{fid}" pos="{cx} {cy} 0" euler="0 0 {f.get("yaw", 0)}">' + "".join(g) + "</body>"


def microwave_xml(f: dict) -> str:
    """Counter-top microwave: the fixture position is its centre; the door is hinged on the left edge, the
    control panel takes the right 30% of the front."""
    (cx, cy, cz), (sx, sy, h) = f["position"], f["size"]
    fid, ys, dw = f["id"], sy / 2, sx * 0.7 - 0.02
    g = [_box(f"{fid}_box", (0, -0.01, h / 2), (sx, sy - 0.02, h), "0.25 0.25 0.27 1"),
         _vr(sx / 2 - sx * 0.3 + 0.01, sx / 2 - 0.01, ys - 0.012, ys - 0.004, 0.03, h - 0.03, "0.12 0.12 0.13 1"),
         f'<body name="{fid}_door" pos="{-sx / 2 + 0.01:.4f} {ys - 0.008:.4f} {h / 2:.4f}">'
         f'<joint name="{fid}_door" type="hinge" axis="0 0 1" range="0 1.9" damping="0.5" frictionloss="0.1"/>'
         f'<inertial pos="{dw / 2:.4f} 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>'
         + _box(f"{fid}_door_panel", (dw / 2, 0.0, 0), (dw, 0.016, h - 0.02), "0.2 0.2 0.22 1", DRAWER)
         + _vr(0.03, dw - 0.05, 0.008, 0.010, -h / 2 + 0.04, h / 2 - 0.04, DARK)
         + f'<geom name="{fid}_door_handle" type="capsule" size="0.008" fromto="{dw - 0.025:.4f} 0.035 {-h / 2 + 0.04:.3f} '
           f'{dw - 0.025:.4f} 0.035 {h / 2 - 0.04:.3f}" rgba="{STEEL}" {DRAWER}/>'
         + "".join(_vcyl(STEEL, 0.004, (dw - 0.025, 0.008, z), (dw - 0.025, 0.035, z)) for z in (-h / 2 + 0.05, h / 2 - 0.05))
         + f'<site name="{fid}_door_handle" pos="{dw - 0.025:.4f} 0.035 0" size="0.008"/></body>']
    return f'<body name="{fid}" pos="{cx} {cy} {cz - h / 2:.4f}" euler="0 0 {f.get("yaw", 0)}">' + "".join(g) + "</body>"


def fridge_xml(f: dict) -> str:
    (cx, cy, _), (sx, sy, h) = f["position"], f["size"]
    fid, yaw = f["id"], f.get("yaw", 0.0)
    yd = sy / 2 - 0.03
    zf = h * 0.25 - 0.006
    fx = sx / 2 - 0.005
    zb = zf - 0.05
    vis = (_vr(-fx, fx, yd, sy / 2 + 0.02, 0.02, zf, "0.9 0.9 0.92 1")
           + "".join(_vcyl(STEEL, 0.005, (x, sy / 2 + 0.019, zb), (x, sy / 2 + 0.06, zb)) for x in (-0.22, 0.22))
           + _vcyl(STEEL, 0.012, (-0.26, sy / 2 + 0.06, zb), (0.26, sy / 2 + 0.06, zb), "capsule")
           + _vr(-sx / 2 + 0.01, sx / 2 - 0.01, sy / 2 - 0.06, yd - 0.002, 0.01, h - 0.01, DARK))
    return (f'<body name="{fid}" pos="{cx} {cy} 0" euler="0 0 {yaw}">'
            + _box(f"{fid}_box", (0, -0.03, h / 2), (sx, sy - 0.06, h), "0.85 0.86 0.88 1") + vis
            + f'<body name="{fid}_door" pos="{-sx / 2:.3f} {yd:.3f} 0">'
              f'<joint name="{fid}_door" type="hinge" axis="0 0 1" range="0 2.0" damping="3" frictionloss="0.3"/>'
              f'<inertial pos="{sx / 2:.3f} 0.02 {h * 0.6:.3f}" mass="4" diaginertia="0.3 0.3 0.3"/>'
            + _box(f"{fid}_door_panel", (sx / 2, 0.025, h * 0.62), (sx - 0.01, 0.05, h * 0.74), "0.9 0.9 0.92 1", DRAWER)
            + _box(f"{fid}_freezer_panel", (sx / 2, 0.025, h * 0.12), (sx - 0.01, 0.05, h * 0.22), "0.9 0.9 0.92 1",
                   f'{DRAWER} group="3"')
            + f'<geom name="{fid}_door_handle" type="capsule" size="0.015" fromto="{sx - 0.07:.3f} 0.09 {h * 0.45:.3f} '
              f'{sx - 0.07:.3f} 0.09 {h * 0.7:.3f}" rgba="{STEEL}" {DRAWER}/>'
            + "".join(_vcyl(STEEL, 0.005, (sx - 0.07, 0.049, h * k), (sx - 0.07, 0.09, h * k)) for k in (0.46, 0.69))
            + f'<site name="{fid}_door_handle" pos="{sx - 0.07:.3f} 0.09 {h * 0.57:.3f}" size="0.01"/>'
            + "</body></body>")


BOWL_WALL = 0.006  # wall thickness; the closed fingers (12 mm apart at the pads) pinch the rim


def _ring_geoms(oid: str, radius: float, height: float, wall: float, mass: float, rgba: str, col: str) -> list[str]:
    """Open-top container: a bottom disc and an octagon of thin wall panels."""
    g = [f'<geom name="{oid}" type="cylinder" size="{radius - wall:.4f} {wall / 2:.4f}" '
         f'pos="0 0 {-height / 2 + wall / 2:.4f}" mass="{0.4 * mass:.4f}" rgba="{rgba}" {col}/>']
    r = radius - wall / 2
    half = r * math.tan(math.pi / 8) + 0.002
    for k in range(8):
        a = k * math.pi / 4
        g.append(f'<geom name="{oid}_wall{k}" type="box" size="{half:.4f} {wall / 2:.4f} {height / 2:.4f}" '
                 f'pos="{r * math.cos(a):.4f} {r * math.sin(a):.4f} 0" euler="0 0 {a + math.pi / 2:.4f}" '
                 f'mass="{0.6 * mass / 8:.4f}" rgba="{rgba}" {col}/>')
    return g


def object_xml(o: dict, pos) -> str:
    """Free body for a catalog object. Objects collide with the robot, the fixtures and the moving parts
    (conaffinity 3), so they rest on drawer trays and open oven doors. A bowl is hollow so the gripper
    can pinch its rim (MAN-4)."""
    oid, cid = o["id"], o["catalog_id"]
    sx, sy, sz = o["size"]
    spec = object_spec(cid, o["size"])
    m, rgba, shape = spec["mass"], spec["rgba"], spec["shape"]
    col = f'friction="{spec["friction"]} 0.02 0.001" condim="4" conaffinity="3"'
    r = min(sx, sy) / 2
    if cid == "bowl":
        g = _ring_geoms(oid, r, sz, BOWL_WALL, m, rgba, col)
    elif shape == "cylinder":
        if cid in ("mug", "cup"):
            r = min(r, 0.035)  # the body has to fit the 10 cm gripper opening with room for the pads
        g = [f'<geom name="{oid}" type="cylinder" size="{r:.4f} {sz / 2:.4f}" mass="{m}" rgba="{rgba}" {col}/>']
        if cid == "mug":
            g += [_box(None, p, sz_, rgba, f'mass="0" {VIS}') for p, sz_ in  # handle: looks only, weighs nothing
                  (((r + 0.014, 0, 0), (0.01, 0.012, sz * 0.5)), ((r + 0.007, 0, sz * 0.25), (0.016, 0.012, 0.01)),
                   ((r + 0.007, 0, -sz * 0.25), (0.016, 0.012, 0.01)))]
    elif shape == "sphere":
        g = [f'<geom name="{oid}" type="sphere" size="{r:.4f}" mass="{m}" rgba="{rgba}" {col}/>']
    elif shape == "bottle":
        g = [f'<geom name="{oid}" type="cylinder" size="{r:.4f} {0.375 * sz:.4f}" pos="0 0 {-0.125 * sz:.4f}" '
             f'mass="{0.9 * m:.4f}" rgba="{rgba}" {col}/>',
             f'<geom name="{oid}_neck" type="cylinder" size="{0.42 * r:.4f} {0.125 * sz:.4f}" pos="0 0 {0.375 * sz:.4f}" '
             f'mass="{0.1 * m:.4f}" rgba="{rgba}" {col}/>']
    elif shape == "pan":
        g = _ring_geoms(oid, r, sz, 0.005, 0.8 * m, rgba, col)
        g.append(_box(f"{oid}_handle", (r + 0.08, 0, sz / 2 - 0.008), (0.16, 0.025, 0.012), rgba, f'mass="{0.2 * m:.4f}" {col}'))
    else:
        g = [f'<geom name="{oid}" type="box" size="{sx / 2:.4f} {sy / 2:.4f} {sz / 2:.4f}" mass="{m}" rgba="{rgba}" {col}/>']
    return (f'<body name="{oid}" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}"><freejoint name="{oid}"/>'
            + "".join(g) + "</body>")


def part_names(twin: dict) -> list[dict]:
    """Every moving part the scene builds for this twin: joint name, handle site, kind and an open position a
    policy can target. Order follows the fixtures."""
    fx = twin["fixtures"]
    sinks = [f for f in fx if f["type"] == "sink"]
    parts = []

    def add(fid, joint, kind, open_q):
        parts.append({"fixture": fid, "joint": joint, "site": f"{joint}_handle", "kind": kind, "open": open_q})

    for f in fx:
        t, fid = f["type"], f["id"]
        if t == "counter":
            for k in range(2):
                add(fid, f"{fid}_drawer{k}", "slide", 0.3)
            add(fid, f"{fid}_door", "hinge", 1.2)
            for k in range(len(upper_doors(f, fx))):
                add(fid, f"{fid}_upper_door{k}", "hinge", 1.2)
            for s in _sinks_in(f, sinks)[:1]:
                add(s["id"], f'{s["id"]}_faucet', "hinge", 1.0)
        elif t.startswith("fridge"):
            add(fid, f"{fid}_door", "hinge", 1.2)
        elif t == "stove":
            add(fid, f"{fid}_oven_door", "hinge", 1.3)
        elif t == "dishwasher":
            add(fid, f"{fid}_door", "hinge", 1.3)
        elif t == "microwave":
            add(fid, f"{fid}_door", "hinge", 1.2)
    return parts




def arm_parts(arm: str, base_pos, base_yaw_deg: float) -> dict[str, str]:
    """MJCF fragments for the chosen arm: `procedural` (robot.py) or `menagerie` (the Franka asset)."""
    if arm == "procedural":
        return {"meshdir": "", "defaults": "", "assets": "", "body": robot.arm_xml(tuple(base_pos), base_yaw_deg),
                "actuators": robot.actuators_xml(), "contact": ""}
    if arm == "menagerie":
        from arrival.sim import menagerie

        return menagerie.arm_fragments(tuple(base_pos), base_yaw_deg)
    raise ValueError(f"unknown arm {arm!r} (procedural or menagerie)")


def build_mjcf(twin: dict, base_pos, base_yaw_deg: float = -90.0, object_pos: dict | None = None,
               lighting: str = "neutral", n_people: int = 0, arm: str = "procedural") -> str:
    """Returns an MJCF string. object_pos overrides {object_id: [x, y, z]} start positions."""
    object_pos = object_pos or {}
    a = arm_parts(arm, base_pos, base_yaw_deg)
    fx = twin["fixtures"]
    sinks = [f for f in fx if f["type"] == "sink"]
    world = []
    for w in twin["walls"]:
        (x0, y0), (x1, y1) = w["start"], w["end"]
        L = math.hypot(x1 - x0, y1 - y0)
        yaw = math.atan2(y1 - y0, x1 - x0)
        world.append(f'<geom name="{w["id"]}" type="box" pos="{(x0 + x1) / 2} {(y0 + y1) / 2} {w["height"] / 2}" '
                     f'size="{L / 2} {max(w["thickness"], 0.05) / 2} {w["height"] / 2}" euler="0 0 {yaw}" '
                     f'rgba="0.93 0.91 0.86 1"/>')
    for f in fx:
        t = f["type"]
        if t == "counter":
            world.append(counter_xml(f, sinks, fx))
        elif t.startswith("fridge"):
            world.append(fridge_xml(f))
        elif t == "stove":
            world.append(stove_xml(f))
        elif t == "dishwasher":
            world.append(dishwasher_xml(f))
        elif t == "microwave":
            world.append(microwave_xml(f))
    for o in twin["objects"]:
        world.append(object_xml(o, object_pos.get(o["id"], o["position"])))
    world += [people.person_xml(i) for i in range(n_people)]
    world.append(a["body"])
    xs = [p for w in twin["walls"] for p in (w["start"][0], w["end"][0])]
    ys = [p for w in twin["walls"] for p in (w["start"][1], w["end"][1])]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    li = LIGHT[lighting]
    meshdir = f' meshdir="{a["meshdir"]}"' if a["meshdir"] else ""
    return f"""<mujoco model="{twin['room_id']}_sim">
<compiler angle="radian" autolimits="true"{meshdir}/>
<option timestep="0.002" cone="elliptic" impratio="10" integrator="implicitfast"/>
<visual><global offwidth="640" offheight="480"/><headlight ambient="{0.25 * li} {0.25 * li} {0.25 * li}" diffuse="{0.3 * li} {0.3 * li} {0.3 * li}"/><quality shadowsize="2048"/></visual>
<asset><texture name="floor" type="2d" builtin="checker" rgb1="0.62 0.55 0.47" rgb2="0.56 0.5 0.43" width="256" height="256"/>
<material name="floor" texture="floor" texrepeat="8 8"/>{a['assets']}</asset>
<default><geom friction="1.0 0.01 0.001"/>{a['defaults']}</default>
<worldbody>
<light name="ceiling" pos="{cx} {cy} 2.35" dir="0 0 -1" diffuse="{li} {li} {0.95 * li}" castshadow="true"/>
<light name="fill" pos="{cx} {max(ys) - 0.3} 2.0" dir="0 -1 -1" diffuse="{0.4 * li} {0.4 * li} {0.4 * li}"/>
<geom name="floor" type="plane" pos="{cx} {cy} 0" size="{max(xs)} {max(ys)} 0.1" material="floor"/>
<camera name="overview" pos="{cx} {max(ys) - 0.1} 2.2" xyaxes="-1 0 0 0 -0.6 0.8" fovy="80"/>
{''.join(world)}
</worldbody>{a['contact']}
<actuator>{a['actuators']}</actuator>
</mujoco>"""
