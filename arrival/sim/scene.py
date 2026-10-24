"""Twin JSON -> MJCF: walls with their scanned doors, windows and openings (visual only), floor, counters with
drawers, hinged cabinet doors and upper cabinets, sink basin with a faucet lever, stove with an oven door, fridge,
dishwasher and microwave with hinged doors, free-body objects, people, Panda-like arm, cameras. Every moving part is
a named joint with a handle site (part_names)."""

from __future__ import annotations

import math
import random

from arrival.catalog.objects import object_spec
from arrival.scan.bundle import ScanBundleError
from arrival.scan.roomplan import PORTALS
from arrival.sim import people, robot
from arrival.twin.compile import COUNTER_HEIGHT, is_counter_host, to_local

LIGHT = {"dim": 0.35, "neutral": 0.7, "bright": 1.0}
LIGHT_JITTER = 0.15  # per-task intensity jitter inside a label; the three bands never overlap
WARM, COOL = (1.0, 0.93, 0.82), (0.86, 0.93, 1.0)  # key light colour runs between these
WALL = "0.93 0.91 0.86 1"
TOP_T = 0.04
MIN_HALF = 0.0005  # smallest box half-extent written: MuJoCo refuses a size of 0 or less
CARCASS_H = 0.70  # cabinet body below the counter top (sink basin sits between this and the top)
DRAWER = 'contype="2" conaffinity="2"'  # moving parts collide with the robot, not with their carcass
APPLIANCES = ("fridge", "stove", "dishwasher")  # floor-standing: counters are trimmed where they run into one
VIS = 'contype="0" conaffinity="0" group="2"'
FRONT, GAP = "0.70 0.55 0.40 1", "0.10 0.09 0.08 1"
STEEL, DARK, SPLASH = "0.75 0.77 0.80 1", "0.08 0.08 0.09 1", "0.88 0.86 0.81 1"
TOP, FURNITURE = "0.82 0.82 0.8 1", "0.46 0.44 0.42 1"  # counter top; the solid box a scanned fixture without a model gets
REVEAL, INSET, FRONT_T, KICK = 0.003, 0.015, 0.018, 0.10
UPPER_Z0, UPPER_H, UPPER_D, UPPER_DOOR_MAX = 1.56, 0.60, 0.32, 0.60  # wall cabinets over a counter run; the wrist
# reaches 1.50 m over the back of the counter when it drops into the sink, so the cabinets start above that
DOOR_FLAT = 1.5  # bottom-hinged doors (oven, dishwasher) stop just short of horizontal
TOP = "0.82 0.82 0.8 1"
WALL_MIN_T = 0.05  # a wall is drawn at least this thick
# doors, windows and openings (portals): white trim, a wood leaf, sky-tinted glass, a dark recess with shaded returns
TRIM, LEAF, GLASS, RECESS, RETURN = ("0.96 0.95 0.92 1", "0.58 0.43 0.30 1", "0.62 0.74 0.84 1", "0.16 0.15 0.14 1",
                                     "0.80 0.78 0.73 1")
# attached by colour in build_mjcf
MATERIALS = {FRONT: "wood", TOP: "stone", SPLASH: "panel", WALL: "paint", LEAF: "wood"}
RACK_Z, RACK_OUT = 0.52, 0.45  # dishwasher upper rack: tray height and how far it slides out over the open door
PORTAL_BAND = 0.05  # jamb, head and sill width, inside the scanned rectangle
PORTAL_MIN = 0.15  # a portal narrower or lower than this once clipped to its wall is not drawn
PORTAL_SNAP = 0.25  # a portal whose parent is not a scanned wall goes on the nearest wall this close to its centre
FLOOR_SNAP = 0.05  # a door or opening whose bottom is this close to the floor starts on it
PROUD = {"frame": 0.015, "sill": 0.03, "leaf": 0.006, "pane": 0.004, "recess": 0.002, "knob": 0.06}  # past each face
KNOB_R, KNOB_Z, KNOB_IN = 0.025, 1.0, 0.07  # door knob: radius, height above the door bottom, inset from the latch jamb


def _box(name, pos, size, rgba, extra=""):
    label = f'name="{name}" ' if name else ""
    half = [max(s / 2, MIN_HALF) for s in size]  # a counter squeezed by neighbours can leave a zero-width panel
    return (f'<geom {label}type="box" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" '
            f'size="{half[0]:.4f} {half[1]:.4f} {half[2]:.4f}" rgba="{rgba}" {extra}/>')


def _vbox(pos, size, rgba):
    return _box(None, pos, size, rgba, VIS)


def _rbox(name, x0, x1, y0, y1, z0, z1, rgba, extra="", o=(0.0, 0.0, 0.0)):
    """Box from coordinate ranges, expressed relative to origin o."""
    return _box(name, ((x0 + x1) / 2 - o[0], (y0 + y1) / 2 - o[1], (z0 + z1) / 2 - o[2]),
                (x1 - x0, y1 - y0, z1 - z0), rgba, extra)


def _vr(x0, x1, y0, y1, z0, z1, rgba, o=(0.0, 0.0, 0.0)):
    """Visual box from coordinate ranges, expressed relative to origin o."""
    return _rbox(None, x0, x1, y0, y1, z0, z1, rgba, VIS, o)


def _vcyl(rgba, r, a, b, kind="cylinder"):
    f = " ".join(f"{v:.4f}" for v in (*a, *b))
    return f'<geom type="{kind}" fromto="{f}" size="{r}" rgba="{rgba}" {VIS}/>'


def _sinks_in(counter: dict, sinks: list[dict]) -> list[dict]:
    sx, sy, _ = counter["size"]
    inside = []
    for sink in sinks:
        x, y = to_local(sink["position"], counter)
        if abs(x) < sx / 2 and abs(y) < sy / 2:
            inside.append(sink)
    return inside


def _host_of(s: dict, fixtures: list[dict]) -> dict | None:
    """The counter or counter-height cabinet a sink is set into, if any."""
    return next((f for f in fixtures if is_counter_host(f) and _sinks_in(f, [s])), None)


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
    """Local x of the drawer stack: left quarter, or the middle if a fridge/stove in the same run overlaps the
    left quarter. Judged in the counter's frame, so a run on any wall gets the same layout."""
    x = -c["size"][0] / 4
    for f in fixtures:
        if not f["type"].startswith(APPLIANCES):
            continue
        rel = float(f.get("yaw", 0.0)) - float(c.get("yaw", 0.0))
        co, si = abs(math.cos(rel)), abs(math.sin(rel))
        lx, ly = to_local(f["position"], c)
        if abs(ly) > (c["size"][1] + si * f["size"][0] + co * f["size"][1]) / 2:
            continue  # across the room, not in this run
        if abs(lx - x) < (co * f["size"][0] + si * f["size"][1]) / 2 + DRAWER_W / 2:
            return 0.0
    return x


DRAWER_W = 0.5


def front_stand(c: dict, family: str, fixtures: list[dict]) -> tuple[float, float]:
    """Where the base stands for a drawer or door task: local x along the run, and distance out from the front."""
    if family in ("open_drawer", "close_drawer"):
        return drawer_x(c, fixtures), 0.72
    # just left of the handle (away from the hinge), outside the door's swing, far enough out that the far end of
    # the handle arc stays within reach of the tilted-down grasp
    return c["size"][0] / 2 - 0.02 - door_w(c, fixtures) + 0.06 - 0.1, 0.65


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
            f'<joint name="{sid}_faucet" type="hinge" axis="0 -1 0" range="0 1.2" damping="0.5" frictionloss="0.05"/>'
            f'<inertial pos="0 0 0" mass="0.05" diaginertia="1e-5 1e-5 1e-5"/>'  # balanced on its cartridge: it stays
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


TABLE_LEG = 0.05  # square leg side (m)


def is_table(f: dict) -> bool:
    """A scanned table: compiled as a counter for RoboCasa (default_config_name "dining") but built as a table."""
    return f.get("type") == "counter" and f.get("default_config_name") == "dining"


def table_xml(f: dict) -> str:
    """A top slab on four legs, no carcass, drawers or doors. Objects rest on the top; the arm collides with it."""
    (cx, cy, _), (sx, sy, h) = f["position"], f["size"]
    fid, yaw = f["id"], f.get("yaw", 0.0)
    wood = "0.62 0.47 0.33 1"
    g = [_box(f"{fid}_top", (0, 0, h - TOP_T / 2), (sx, sy, TOP_T), wood)]
    lx, ly, lh = sx / 2 - 0.06 - TABLE_LEG / 2, sy / 2 - 0.06 - TABLE_LEG / 2, h - TOP_T
    g += [_box(f"{fid}_leg{k}", (px, py, lh / 2), (TABLE_LEG, TABLE_LEG, lh), "0.35 0.27 0.2 1")
          for k, (px, py) in enumerate(((-lx, -ly), (lx, -ly), (-lx, ly), (lx, ly)))]
    return f'<body name="{fid}" pos="{cx} {cy} 0" euler="0 0 {yaw}">' + "".join(g) + "</body>"


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
    holes = [(to_local(s["position"], c)[0], s["size"][0], s["size"][1], s) for s in _sinks_in(c, sinks)]
    zt = h - TOP_T / 2
    if not holes:
        g.append(_box(f"{cid}_top", (xm, 0, zt), (xw, sy, TOP_T), TOP))
    for hx, hw, hd, s in holes[:1]:
        g += _split_top_xml(cid, x0, x1, hx, hw, hd, sy, zt)
        g += _basin_xml(s["id"], hx, hw, hd, sy, h)
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
    # drawers (slide along local +y) in the left half; hinged door cabinets in the right half.
    # Each hidden collision box uses the same coordinate ranges as its visible front, so the arm touches what
    # the cameras show (MAN-2).
    hid = f'{DRAWER} group="3"'
    for k in range(2):
        dz = dz0 - k * 0.24  # top drawer sits in the band just under the counter top
        o = (dx, front, dz)
        z0, z1 = (0.600, zf - REVEAL) if k == 0 else (KICK + 0.23, 0.597)
        body.append(
            f'<body name="{cid}_drawer{k}" pos="{dx:.3f} {front:.3f} {dz:.3f}">'
            f'<joint name="{cid}_drawer{k}" type="slide" axis="0 1 0" range="0 0.4" damping="5" frictionloss="0.5"/>'
            f'<inertial pos="0 -0.2 0" mass="1.5" diaginertia="0.02 0.02 0.02"/>'
            + _rbox(f"{cid}_drawer{k}_front", da + r2, db - r2, ya, yb, z0, z1, "0.65 0.5 0.36 1", hid, o)
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
        + _rbox(f"{cid}_door_panel", db + r2, xr - r2, ya, yb, zlo, zf - REVEAL, "0.65 0.5 0.36 1", hid, hinge)
        + f'<geom name="{cid}_door_handle" type="capsule" size="0.012" fromto="{hx_:.3f} 0.05 0.1 '
          f'{hx_:.3f} 0.05 0.25" rgba="{STEEL}" {DRAWER}/>'
        + _vr(db + r2, xr - r2, ya, yb, zlo, zf - REVEAL, FRONT, hinge)
        + "".join(_vcyl(STEEL, 0.005, (hx_, yb - hinge[1] - 0.001, z), (hx_, 0.05, z)) for z in (0.11, 0.24))
        + f'<site name="{cid}_door_handle" pos="{hx_:.3f} 0.05 0.17" size="0.01"/></body>')
    return "".join(body) + "</body>"


def _split_top_xml(cid: str, x0: float, x1: float, hx: float, hw: float, hd: float, sy: float, zt: float) -> list[str]:
    """Top slab in four pieces around a sink hole centred at local x=hx."""
    lx0, lx1 = x0, hx - hw / 2
    rx0, rx1 = hx + hw / 2, x1
    rim = (sy - hd) / 2
    g = []
    if lx1 - lx0 > 1e-4:  # a hole flush with the counter end leaves no slab on that side
        g.append(_box(f"{cid}_top_l", ((lx0 + lx1) / 2, 0, zt), (lx1 - lx0, sy, TOP_T), TOP))
    if rx1 - rx0 > 1e-4:
        g.append(_box(f"{cid}_top_r", ((rx0 + rx1) / 2, 0, zt), (rx1 - rx0, sy, TOP_T), TOP))
    g.append(_box(f"{cid}_top_f", (hx, sy / 2 - rim / 2, zt), (hw, rim, TOP_T), TOP))
    g.append(_box(f"{cid}_top_b", (hx, -sy / 2 + rim / 2, zt), (hw, rim, TOP_T), TOP))
    return g


def _basin_xml(sid: str, hx: float, hw: float, hd: float, sy: float, h: float) -> list[str]:
    """Basin: bottom + 4 thin walls from carcass top to counter top, rim and drain visuals, faucet lever."""
    bz0, bt = CARCASS_H + 0.01, 0.01
    wall_h = h - bz0
    steel = "0.6 0.62 0.65 1"
    g = [_box(f"{sid}_bottom", (hx, 0, bz0), (hw, hd, 0.02), steel)]
    for k, (px, py, wx, wy) in enumerate([(hx - hw / 2 + bt / 2, 0, bt, hd), (hx + hw / 2 - bt / 2, 0, bt, hd),
                                          (hx, hd / 2 - bt / 2, hw, bt), (hx, -hd / 2 + bt / 2, hw, bt)]):
        g.append(_box(f"{sid}_wall{k}", (px, py, bz0 + wall_h / 2), (wx, wy, wall_h), steel))
    g += _sink_visuals(hx, hw, hd, sy, h)
    g.append(_faucet_lever_xml(sid, hx, -sy / 2 + 0.04, h))
    return g


def sink_unit_xml(s: dict) -> str:
    """A sink with no counter or cabinet under it: a carcass the sink's footprint plus a 5 cm rim, a top at
    counter height and the same basin and faucet a counter gets, so a sink task still has a basin to reach."""
    (cx, cy, _), (hw, hd, _) = s["position"], s["size"]
    sid, yaw, h = s["id"], s.get("yaw", 0.0), COUNTER_HEIGHT
    sx, sy = hw + 0.1, hd + 0.1
    g = [_box(f"{sid}_carcass", (0, 0, CARCASS_H / 2), (sx, sy - 0.04, CARCASS_H), "0.55 0.42 0.3 1")]
    g += _split_top_xml(sid, -sx / 2, sx / 2, 0.0, hw, hd, sy, h - TOP_T / 2)
    g += _basin_xml(sid, 0.0, hw, hd, sy, h)
    return f'<body name="{sid}" pos="{cx} {cy} 0" euler="0 0 {yaw}">' + "".join(g) + "</body>"


def furniture_xml(f: dict) -> str:
    """Any scanned fixture without its own model (tall or wall-hung storage, chair, bed, sofa, TV, ...): one
    solid box at its scanned size and height, so the arm, the base and the walkers cannot pass through it."""
    (cx, cy, cz), (sx, sy, sz) = f["position"], f["size"]
    return (f'<body name="{f["id"]}" pos="{cx} {cy} {cz}" euler="0 0 {f.get("yaw", 0.0)}">'
            + _box(f'{f["id"]}_box', (0, 0, 0), (sx, sy, sz), FURNITURE) + "</body>")


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
    """Floor-standing dishwasher: a hollow carcass (bottom, top, back and two sides), a bottom-hinged front door
    with the handle bar near the top, and an upper rack (`{fid}_rack`, slide along +y) that rolls out over the
    open door so an arm on a pedestal can reach it; the lower rack would sit below a fixed-base arm's reach."""
    (cx, cy, _), (sx, sy, h) = f["position"], f["size"]
    fid, ys = f["id"], sy / 2
    shell, t = "0.55 0.56 0.58 1", 0.03
    g = [_box(f"{fid}_bottom", (0, -0.01, 0.04), (sx, sy - 0.02, 0.08), shell),
         _box(f"{fid}_top", (0, -0.01, h - t / 2), (sx, sy - 0.02, t), shell),
         _box(f"{fid}_back", (0, -ys + t / 2, h / 2), (sx, t, h), shell),
         _box(f"{fid}_left", (-sx / 2 + t / 2, -0.01, h / 2), (t, sy - 0.02, h), shell),
         _box(f"{fid}_right", (sx / 2 - t / 2, -0.01, h / 2), (t, sy - 0.02, h), shell),
         _vr(-sx / 2, sx / 2, ys - 0.012, ys + 0.004, h - 0.08, h - 0.01, "0.2 0.2 0.21 1"),
         _vr(-sx / 2, sx / 2, ys - 0.01, ys + 0.003, 0, 0.08, DARK),
         _flap_door_xml(f"{fid}_door", (0, ys + 0.006, 0.08), sx - 0.04, h - 0.18, 5.0, STEEL, handle_dz=0.05)]
    rw, rd, rail = sx - 2 * t - 0.06, sy - 0.12, 0.04
    rack = [f'<body name="{fid}_rack" pos="0 0 {RACK_Z}">'
            f'<joint name="{fid}_rack" type="slide" axis="0 1 0" range="0 {RACK_OUT}" damping="5" frictionloss="0.5"/>'
            f'<inertial pos="0 0 0.01" mass="1.2" diaginertia="0.02 0.02 0.02"/>'
            + _box(f"{fid}_rack_tray", (0, 0, 0), (rw, rd, 0.02), "0.7 0.72 0.75 1", DRAWER)]
    for k, (px, py, wx, wy) in enumerate([(0, rd / 2 - 0.005, rw, 0.01), (0, -rd / 2 + 0.005, rw, 0.01),
                                          (rw / 2 - 0.005, 0, 0.01, rd), (-rw / 2 + 0.005, 0, 0.01, rd)]):
        rack.append(_box(f"{fid}_rack_rail{k}", (px, py, 0.01 + rail / 2), (wx, wy, rail), "0.7 0.72 0.75 1", DRAWER))
    g.append("".join(rack) + "</body>")
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


def fridge_mirror(f: dict) -> int:
    """+1 for the default left-hinged fridge, -1 when the fixture says `hinge: "right"` (the door is the mirror
    image in the fixture's x: hinge on the right edge, free edge on the left, swinging the other way)."""
    return -1 if f.get("hinge") == "right" else 1


def fridge_door_frame(f: dict) -> dict:
    """The fridge door in the fixture's frame: hinge xy (left edge, just behind the front), the panel's length
    along local +x when closed, the handle in the door's frame, and the front face's local y. people.py drives
    the door from these numbers, so they live in one place."""
    sx, sy, h = f["size"]
    m = fridge_mirror(f)
    return {"hinge": (-m * sx / 2, sy / 2 - 0.03), "length": sx, "handle": (m * (sx - 0.07), 0.09, h * 0.57),
            "front": sy / 2, "mirror": m}


def fridge_xml(f: dict) -> str:
    (cx, cy, _), (sx, sy, h) = f["position"], f["size"]
    fid, yaw = f["id"], f.get("yaw", 0.0)
    yd = fridge_door_frame(f)["hinge"][1]
    zf = h * 0.25 - 0.006
    fx = sx / 2 - 0.005
    zb = zf - 0.05
    hx, hy, hz = fridge_door_frame(f)["handle"]
    m = fridge_mirror(f)
    px = m * sx / 2  # panel centre along x from the hinge
    vis = (_vr(-fx, fx, yd, sy / 2 + 0.02, 0.02, zf, "0.9 0.9 0.92 1")
           + "".join(_vcyl(STEEL, 0.005, (x, sy / 2 + 0.019, zb), (x, sy / 2 + 0.06, zb)) for x in (-0.22, 0.22))
           + _vcyl(STEEL, 0.012, (-0.26, sy / 2 + 0.06, zb), (0.26, sy / 2 + 0.06, zb), "capsule")
           + _vr(-sx / 2 + 0.01, sx / 2 - 0.01, sy / 2 - 0.06, yd - 0.002, 0.01, h - 0.01, DARK))
    return (f'<body name="{fid}" pos="{cx} {cy} 0" euler="0 0 {yaw}">'
            + _box(f"{fid}_box", (0, -0.03, h / 2), (sx, sy - 0.06, h), "0.85 0.86 0.88 1") + vis
            + f'<body name="{fid}_door" pos="{-m * sx / 2:.3f} {yd:.3f} 0">'
              f'<joint name="{fid}_door" type="hinge" axis="0 0 {m}" range="0 2.0" damping="3" frictionloss="0.3"/>'
              f'<inertial pos="{px:.3f} 0.02 {h * 0.6:.3f}" mass="4" diaginertia="0.3 0.3 0.3"/>'
            + _box(f"{fid}_door_panel", (px, 0.025, h * 0.62), (sx - 0.01, 0.05, h * 0.74), "0.9 0.9 0.92 1", DRAWER)
            + _box(f"{fid}_freezer_panel", (px, 0.025, h * 0.12), (sx - 0.01, 0.05, h * 0.22), "0.9 0.9 0.92 1",
                   f'{DRAWER} group="3"')
            + f'<geom name="{fid}_door_handle" type="capsule" size="0.015" fromto="{hx:.3f} {hy} {h * 0.45:.3f} '
              f'{hx:.3f} {hy} {h * 0.7:.3f}" rgba="{STEEL}" {DRAWER}/>'
            + "".join(_vcyl(STEEL, 0.005, (hx, 0.049, h * k), (hx, hy, h * k)) for k in (0.46, 0.69))
            + f'<site name="{fid}_door_handle" pos="{hx:.3f} {hy} {hz:.3f}" size="0.01"/>'
            + "</body></body>")


BOWL_WALL = 0.006  # wall thickness; the closed fingers (12 mm apart at the pads) pinch the rim
PLATE_RIM, PLATE_FLOOR = 0.018, 0.005  # a thick rolled rim the closed pads (12 mm apart) can squeeze; thin floor


def _ring_geoms(oid: str, radius: float, height: float, wall: float, mass: float, rgba: str, col: str,
                floor: float | None = None) -> list[str]:
    """Open-top container: a bottom disc (`floor` thick, default the wall) and an octagon of wall panels."""
    floor = wall if floor is None else floor
    g = [f'<geom name="{oid}" type="cylinder" size="{radius - wall:.4f} {floor / 2:.4f}" '
         f'pos="0 0 {-height / 2 + floor / 2:.4f}" mass="{0.4 * mass:.4f}" rgba="{rgba}" {col}/>']
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
    (conaffinity 3), so they rest on drawer trays and open oven doors. A bowl and a plate are hollow so the
    gripper can pinch their rim (MAN-4)."""
    oid, cid = o["id"], o["catalog_id"]
    sx, sy, sz = o["size"]
    spec = object_spec(cid, o["size"])
    m, rgba, shape = spec["mass"], spec["rgba"], spec["shape"]
    col = f'friction="{spec["friction"]} 0.02 0.001" condim="4" conaffinity="3"'
    r = min(sx, sy) / 2
    if cid == "bowl":
        g = _ring_geoms(oid, r, sz, BOWL_WALL, m, rgba, col)
    elif cid == "plate":  # a shallow dish: a thin floor and a thick 18 mm rim the pads pinch (MAN-4)
        g = _ring_geoms(oid, r, sz, PLATE_RIM, m, rgba, col, floor=PLATE_FLOOR)
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
        if is_counter_host(f) and not is_table(f):  # a table has no moving parts
            for k in range(2):
                add(fid, f"{fid}_drawer{k}", "slide", 0.3)
            add(fid, f"{fid}_door", "hinge", 1.2)
            for k in range(len(upper_doors(f, fx))):
                add(fid, f"{fid}_upper_door{k}", "hinge", 1.2)
            for s in _sinks_in(f, sinks)[:1]:
                add(s["id"], f'{s["id"]}_faucet', "hinge", 1.0)
        elif t == "sink":
            if _host_of(f, fx) is None:
                add(fid, f"{fid}_faucet", "hinge", 1.0)
        elif t.startswith("fridge"):
            add(fid, f"{fid}_door", "hinge", 1.2)
        elif t == "stove":
            add(fid, f"{fid}_oven_door", "hinge", 1.3)
        elif t == "dishwasher":
            add(fid, f"{fid}_door", "hinge", 1.3)
            parts.append({"fixture": fid, "joint": f"{fid}_rack", "site": None, "kind": "slide", "open": RACK_OUT})
        elif t == "microwave":
            add(fid, f"{fid}_door", "hinge", 1.2)
    return parts


def _seg_dist(p, w: dict) -> float:
    """Horizontal distance from point p to a wall's segment."""
    (x0, y0), (x1, y1) = w["start"], w["end"]
    dx, dy = x1 - x0, y1 - y0
    n = dx * dx + dy * dy
    t = 0.0 if n < 1e-12 else min(max(((p[0] - x0) * dx + (p[1] - y0) * dy) / n, 0.0), 1.0)
    return math.hypot(p[0] - x0 - t * dx, p[1] - y0 - t * dy)


def portal_wall(p: dict, walls: list[dict]) -> dict | None:
    """The wall a door, window or opening is drawn on: the one its `wall_id` names, else (an export without
    `parentIdentifier`, or a parent that is not one of the walls) the wall whose segment passes nearest its centre,
    within PORTAL_SNAP. None: no wall qualifies and the portal is not drawn."""
    if p.get("wall_id") is not None:
        for w in walls:
            if w["id"] == p["wall_id"]:
                return w
    d, i = min(((_seg_dist(p["position"], w), i) for i, w in enumerate(walls)), default=(math.inf, -1))
    return walls[i] if d <= PORTAL_SNAP else None


def portal_xml(kind: str, k: int, p: dict, wall: dict) -> list[str]:
    """Visual-only geoms for the k-th door, window or opening, named `{kind}{k}_{part}`, in the plane of `wall`: the
    scanned rectangle is put on the wall's line and clipped to its length and height, and the frame sits inside it.
    Every box runs through the wall and stands proud of both faces (PROUD), so it shows from either side without
    knowing which one is the room. A closed door is two jambs, a head, a wood leaf and a knob; an open door and an
    opening are jambs and a head around a dark recess (RoomPlan gives no hinge side, and a leaf swung into the room
    would be something the arm and the walkers pass through); a window is jambs, a head, a sill, a pane and a mullion;
    an opening that starts above the floor gets a sill too."""
    (x0, y0), (x1, y1) = wall["start"], wall["end"]
    length = math.hypot(x1 - x0, y1 - y0)
    if length < PORTAL_MIN:
        return []
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    (px, py, pz), w, h = p["position"], p["size"][0], p["size"][2]
    u = (px - x0) * ux + (py - y0) * uy
    u0, u1 = max(u - w / 2, 0.0), min(u + w / 2, length)
    z0, z1 = max(pz - h / 2, 0.0), min(pz + h / 2, wall["height"])
    if kind != "window" and z0 <= FLOOR_SNAP:
        z0 = 0.0
    if u1 - u0 < PORTAL_MIN or z1 - z0 < PORTAL_MIN:
        return []
    b, closed = PORTAL_BAND, kind == "door" and not p.get("is_open")
    sill = kind == "window" or (kind == "opening" and z0 > 0.0)
    trim = RETURN if kind == "opening" else TRIM
    zs = z0 + b if sill else z0  # foot of the jambs and of the infill
    parts = [("jamb0", u0, u0 + b, zs, z1, "frame", trim), ("jamb1", u1 - b, u1, zs, z1, "frame", trim),
             ("head", u0 + b, u1 - b, z1 - b, z1, "frame", trim)]
    if sill:
        parts.append(("sill", u0, u1, z0, zs, "sill" if kind == "window" else "frame", trim))
    if kind == "window":
        um = (u0 + u1) / 2
        parts += [("pane", u0 + b, u1 - b, zs, z1 - b, "pane", GLASS),
                  ("mullion", um - b / 4, um + b / 4, zs, z1 - b, "frame", TRIM)]
    elif closed:
        parts.append(("leaf", u0 + b, u1 - b, z0, z1 - b, "leaf", LEAF))
    else:
        parts.append(("recess", u0 + b, u1 - b, zs, z1 - b, "recess", RECESS))
    half_t, yaw = max(wall["thickness"], WALL_MIN_T) / 2, math.atan2(uy, ux)

    def on_line(uc: float) -> tuple[float, float]:
        return x0 + ux * uc, y0 + uy * uc

    g = []
    for part, ua, ub, za, zb, proud, rgba in parts:
        g.append(_box(f"{kind}{k}_{part}", (*on_line((ua + ub) / 2), (za + zb) / 2),
                      (ub - ua, 2 * (half_t + PROUD[proud]), zb - za), rgba, f'euler="0 0 {yaw}" {VIS}'))
    if closed:  # one bar through the wall makes a knob on each face, by the latch jamb
        (kx, ky), r = on_line(max(u1 - b - KNOB_IN, (u0 + u1) / 2)), half_t + PROUD["knob"]
        kz = z0 + min(KNOB_Z, (z1 - b - z0) / 2)
        g.append(f'<geom name="{kind}{k}_knob" type="cylinder" fromto="{kx + uy * r:.4f} {ky - ux * r:.4f} {kz:.4f} '
                 f'{kx - uy * r:.4f} {ky + ux * r:.4f} {kz:.4f}" size="{KNOB_R}" rgba="{STEEL}" {VIS}/>')
    return g


def portals_xml(twin: dict) -> list[str]:
    """The twin's doors, windows and openings (a RoomPlan export carries them; an authored scan has none), each on
    its wall (portal_wall). Visual only: they change renders, never contacts, and a twin without any builds as
    before."""
    g = []
    for key, kind in PORTALS:
        for k, p in enumerate(twin.get(key) or []):
            wall = portal_wall(p, twin["walls"])
            if wall is not None:
                g += portal_xml(kind, k, p, wall)
    return g


def assets_xml() -> str:
    """Builtin textures only (no files), white so the geom rgba sets the colour: tile floor with grout,
    plank-grain wood for cabinet fronts, speckled stone for counter tops, glossy boards for the backsplash
    (a 2d texture is projected along a box's local z, so a vertical face only varies along x)."""
    return (
        '<asset>'
        '<texture name="floor_tile" type="2d" builtin="flat" mark="edge" rgb1="0.68 0.61 0.53" '
        'markrgb="0.42 0.38 0.34" width="128" height="128"/>'
        '<material name="floor" texture="floor_tile" texuniform="true" texrepeat="3 3" '
        'specular="0.1" shininess="0.2"/>'
        '<texture name="wood_grain" type="2d" builtin="checker" rgb1="1 1 1" rgb2="0.9 0.87 0.84" '
        'mark="random" random="0.03" markrgb="0.72 0.66 0.6" width="64" height="64"/>'
        '<material name="wood" texture="wood_grain" texuniform="true" texrepeat="1.5 40" '
        'specular="0.2" shininess="0.2"/>'
        '<texture name="stone_speckle" type="2d" builtin="flat" rgb1="1 1 1" mark="random" random="0.05" '
        'markrgb="0.55 0.55 0.53" width="128" height="128"/>'
        '<material name="stone" texture="stone_speckle" texuniform="true" texrepeat="6 6" '
        'reflectance="0.1" specular="0.7" shininess="0.7"/>'
        '<texture name="panel_lines" type="2d" builtin="flat" mark="edge" rgb1="1 1 1" '
        'markrgb="0.8 0.8 0.8" width="64" height="64"/>'
        '<material name="panel" texture="panel_lines" texuniform="true" texrepeat="8 8" '
        'specular="0.5" shininess="0.5"/>'
        '<material name="paint" specular="0.05" shininess="0.05"/>'
        '</asset>')


def lights_xml(lighting: str, cx: float, cy: float, y_front: float, seed: int | None = None) -> tuple[str, float]:
    """Key spot (shadows), cool fill from the front, and a low window light from one side. The label sets
    the level; the seed jitters level, colour temperature, key position and window side within the label."""
    rng = random.Random(seed)
    u = (lambda a: rng.uniform(-a, a)) if seed is not None else (lambda a: 0.0)
    li = LIGHT[lighting] * (1 + u(LIGHT_JITTER))
    t = 0.5 + u(0.5)
    kc = [w + (c - w) * t for w, c in zip(WARM, COOL)]
    kx, ky = cx + u(0.4), cy + u(0.4)
    side = rng.choice((-1, 1)) if seed is not None else 1
    key = " ".join(f"{li * c:.3f}" for c in kc)
    spec = " ".join(f"{0.4 * li * c:.3f}" for c in kc)
    fill = " ".join(f"{0.3 * li * c:.3f}" for c in COOL)
    window = " ".join(f"{0.25 * li * c:.3f}" for c in WARM)
    return (f'<light name="key" pos="{kx:.3f} {ky:.3f} 2.35" dir="0 0 -1" cutoff="80" exponent="1" '
            f'diffuse="{key}" specular="{spec}" castshadow="true"/>'
            f'<light name="fill" directional="true" pos="{cx:.3f} {y_front:.3f} 2.0" dir="0 -0.6 -1" '
            f'diffuse="{fill}" specular="0 0 0" castshadow="false"/>'
            f'<light name="window" directional="true" pos="{cx + side * 2:.3f} {cy:.3f} 1.6" '
            f'dir="{-side * 0.8:.1f} 0 -0.45" diffuse="{window}" specular="0 0 0" castshadow="false"/>'), li


def arm_parts(arm: str, base_pos, base_yaw_deg: float, mobile: bool = False) -> dict[str, str]:
    """MJCF fragments for the chosen arm: `procedural` (robot.py) or `menagerie` (the Franka asset, menagerie.py)."""
    if arm == "procedural":
        return {"meshdir": "", "defaults": "", "assets": "", "body": robot.arm_xml(tuple(base_pos), base_yaw_deg, mobile=mobile),
                "actuators": robot.actuators_xml(mobile=mobile), "contact": ""}
    if arm == "menagerie":
        from arrival.sim import menagerie

        return menagerie.arm_fragments(tuple(base_pos), base_yaw_deg, mobile)
    raise ValueError(f"unknown arm {arm!r} (procedural or menagerie)")


OVERVIEW_BACK = 2.4  # m the overview camera stands behind the robot (a north-wall run keeps its old camera)


def overview_camera_xml(xs, ys, base_pos, base_yaw_deg: float, fixtures=()) -> str:
    """The overview camera 2.2 m up, behind the robot and looking where it faces, tilted down (forward 0.8, down
    0.6): OVERVIEW_BACK behind the base, pulled in to stay 0.2 m inside the walls and clear of every fixture
    footprint (a cabinet behind the robot would fill the frame). A robot facing -y (every north-wall run, the hero
    kitchen) keeps the original camera line byte for byte; a run on another wall used to film empty floor."""
    fx, fy = math.cos(math.radians(base_yaw_deg)), math.sin(math.radians(base_yaw_deg))
    if abs(fx) < 1e-9 and fy < 0:
        return f'<camera name="overview" pos="{(min(xs) + max(xs)) / 2} {max(ys) - 0.1} 2.2" xyaxes="-1 0 0 0 -0.6 0.8" fovy="80"/>'
    lo, hi = (min(xs) + 0.2, min(ys) + 0.2), (max(xs) - 0.2, max(ys) - 0.2)

    def clear(x, y):
        inside = lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1]
        return inside and all(not (abs(lx) < f["size"][0] / 2 + 0.15 and abs(ly) < f["size"][1] / 2 + 0.15)
                              for f in fixtures for lx, ly in [to_local((x, y), f)])

    d = OVERVIEW_BACK
    while d > 0.6 and not clear(base_pos[0] - fx * d, base_pos[1] - fy * d):
        d -= 0.1
    x, y = base_pos[0] - fx * d, base_pos[1] - fy * d
    xa = (fy, -fx, 0.0)  # forward x up, normalised: horizontal, to the right of the view
    ya = (0.6 * fx, 0.6 * fy, 0.8)  # xa x forward, with forward = (0.8 fx, 0.8 fy, -0.6)
    axes = " ".join(f"{v:.4f}" for v in (*xa, *ya))
    return f'<camera name="overview" pos="{x:.4f} {y:.4f} 2.2" xyaxes="{axes}" fovy="80"/>'


def build_mjcf(twin: dict, base_pos, base_yaw_deg: float = -90.0, object_pos: dict | None = None,
               lighting: str = "neutral", n_people: int = 0, seed: int | None = None, people_seed: int = 0,
               base: str = "fixed", arm: str = "procedural") -> str:
    """Returns an MJCF string. object_pos overrides {object_id: [x, y, z]} start positions.
    seed jitters the lights within the lighting label (None gives the nominal lights); people_seed picks the walkers' bodies.
    base="mobile" gives the pedestal planar joints and actuators (MAN-3); "fixed" is the string main always built.
    arm="menagerie" swaps in the Franka asset from MuJoCo Menagerie (needs `python scripts/fetch_menagerie.py`)."""
    object_pos = object_pos or {}
    if not twin["walls"]:
        raise ScanBundleError("the twin has no walls, so there is no room to simulate; rescan with the walls captured")
    a = arm_parts(arm, base_pos, base_yaw_deg, mobile=base == "mobile")
    fx = twin["fixtures"]
    sinks = [f for f in fx if f["type"] == "sink"]
    world = []
    for w in twin["walls"]:
        (x0, y0), (x1, y1) = w["start"], w["end"]
        L = math.hypot(x1 - x0, y1 - y0)
        yaw = math.atan2(y1 - y0, x1 - x0)
        world.append(f'<geom name="{w["id"]}" type="box" pos="{(x0 + x1) / 2} {(y0 + y1) / 2} {w["height"] / 2}" '
                     f'size="{L / 2} {max(w["thickness"], WALL_MIN_T) / 2} {w["height"] / 2}" euler="0 0 {yaw}" '
                     f'rgba="{WALL}"/>')
    world += portals_xml(twin)
    for f in fx:
        t = f["type"]
        if is_counter_host(f):
            world.append(table_xml(f) if is_table(f) else counter_xml(f, sinks, fx))
        elif t == "sink":
            if _host_of(f, fx) is None:  # set into a counter or cabinet, the basin is cut by counter_xml
                world.append(sink_unit_xml(f))
        elif t.startswith("fridge"):
            world.append(fridge_xml(f))
        elif t == "stove":
            world.append(stove_xml(f))
        elif t == "dishwasher":
            world.append(dishwasher_xml(f))
        elif t == "microwave":
            world.append(microwave_xml(f))
        else:
            world.append(furniture_xml(f))
    for o in twin["objects"]:
        world.append(object_xml(o, object_pos.get(o["id"], o["position"])))
    world += [people.person_xml(i, people_seed) for i in range(n_people)]
    world.append(a["body"])
    xs = [p for w in twin["walls"] for p in (w["start"][0], w["end"][0])]
    ys = [p for w in twin["walls"] for p in (w["start"][1], w["end"][1])]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    lights, li = lights_xml(lighting, cx, cy, max(ys) - 0.3, seed)
    body = "".join(world)
    for rgba, mat in MATERIALS.items():
        body = body.replace(f'rgba="{rgba}"', f'rgba="{rgba}" material="{mat}"')
    meshdir = f' meshdir="{a["meshdir"]}"' if a["meshdir"] else ""
    assets = assets_xml().replace("</asset>", a["assets"] + "</asset>")
    return f"""<mujoco model="{twin['room_id']}_sim">
<compiler angle="radian" autolimits="true"{meshdir}/>
<option timestep="0.002" cone="elliptic" impratio="10" integrator="implicitfast"/>
<visual><global offwidth="640" offheight="480"/><headlight ambient="{0.15 * li:.3f} {0.15 * li:.3f} {0.15 * li:.3f}" diffuse="{0.1 * li:.3f} {0.1 * li:.3f} {0.1 * li:.3f}" specular="0 0 0"/><quality shadowsize="2048"/></visual>
{assets}
<default><geom friction="1.0 0.01 0.001"/>{a['defaults']}</default>
<worldbody>
{lights}
<geom name="floor" type="plane" pos="{cx} {cy} 0" size="{max(xs)} {max(ys)} 0.1" material="floor"/>
{overview_camera_xml(xs, ys, base_pos, base_yaw_deg, twin["fixtures"])}
{body}
</worldbody>{a['contact']}
<actuator>{a['actuators']}</actuator>
</mujoco>"""
