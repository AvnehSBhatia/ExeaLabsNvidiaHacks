"""Procedural Panda 7-DoF arm (Franka kinematic chain) + parallel gripper.

Collision uses the original capsule links and hand box (hidden, group 3); the visible arm is built from
primitive Panda-shaped visual geoms (housings, joint rings, Franka hand) that never collide.
"""

from __future__ import annotations

import math

HOME = [0.0, -0.3, 0.0, -2.2, 0.0, 1.9, 0.785]
RANGES = [(-2.9, 2.9), (-1.76, 1.76), (-2.9, 2.9), (-3.07, -0.07), (-2.9, 2.9), (-0.02, 3.75), (-2.9, 2.9)]
KP = [3000, 3000, 2500, 2500, 1200, 1200, 600]
FINGER_OPEN = 0.05
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
COL_ATTRS = {"contype": "2", "conaffinity": "1"}  # hits world + moving parts, no self-collision
COL = " ".join(f'{k}="{v}"' for k, v in COL_ATTRS.items())
WRIST_CAMERA = {"name": "wrist", "pos": "0.08 0 0.02", "xyaxes": "0 -1 0 0.5 0 0.86", "fovy": "75"}
HID = 'group="3"'
WHITE, DARK, GREY = "0.94 0.94 0.95 1", "0.13 0.13 0.14 1", "0.55 0.57 0.6 1"


def _vis(kind: str, rgba: str, **kw: str) -> str:
    attrs = " ".join(f'{k}="{v}"' for k, v in kw.items())
    return f'<geom type="{kind}" {attrs} rgba="{rgba}" contype="0" conaffinity="0" group="2"/>'


def _cap(rgba: str, r: float, a: str, b: str) -> str:
    return _vis("capsule", rgba, fromto=f"{a} {b}", size=r)


def _cyl(rgba: str, r: float, a: str, b: str) -> str:
    return _vis("cylinder", rgba, fromto=f"{a} {b}", size=r)


def _link(i: int, pos: str, quat: str, geom: str, inner: str) -> str:
    lo, hi = RANGES[i - 1]
    return (f'<body name="link{i}" pos="{pos}" quat="{quat}" gravcomp="1">'
            f'<inertial pos="0 0 0" mass="1.5" diaginertia="0.01 0.01 0.01"/>'
            f'<joint name="joint{i}" axis="0 0 1" range="{lo} {hi}" damping="8" armature="0.1"/>'
            f'{geom}{inner}</body>')


def arm_xml(base_pos: tuple[float, float, float], base_yaw_deg: float) -> str:
    """Body tree for the arm on a pedestal; Franka Panda link offsets (d1=.333, d3=.316, a=.0825, d5=.384, a7=.088)."""
    cap = f'type="capsule" rgba="0.92 0.92 0.95 1" {HID} ' + COL
    z0 = "0 0 0"
    hand_vis = (_cyl(DARK, 0.031, z0, "0 0 0.012")
                + _vis("box", WHITE, size="0.028 0.075 0.021", pos="0 0 0.033")
                + "".join(_cyl(WHITE, 0.021, f"-0.028 {y} 0.033", f"0.028 {y} 0.033") for y in (-0.075, 0.075)))
    hand = (f'<body name="hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834" gravcomp="1">'
            f'<inertial pos="0 0 0.03" mass="0.6" diaginertia="0.002 0.002 0.002"/>'
            f'<geom type="box" size="0.03 0.09 0.025" pos="0 0 0.03" rgba="0.2 0.2 0.22 1" {HID} {COL}/>'
            + hand_vis
            + "<camera " + " ".join(f'{k}="{v}"' for k, v in WRIST_CAMERA.items()) + "/>" +
            f'<site name="eef" pos="0 0 0.115" size="0.01" rgba="1 0 0 0"/>'
            + "".join(
                f'<body name="finger_{s}" pos="0 0 0.0584" gravcomp="1">'
                f'<inertial pos="0 0 0.03" mass="0.05" diaginertia="1e-5 1e-5 1e-5"/>'
                f'<joint name="finger_{s}" type="slide" axis="0 {sg} 0" range="0 {FINGER_OPEN}" damping="5"/>'
                f'<geom type="box" size="0.01 0.006 0.03" pos="0 {sg * 0.012} 0.05" rgba="0.1 0.1 0.1 1" '
                f'friction="2.0 0.05 0.001" condim="6" {COL}/>'
                + _vis("box", GREY, size="0.0105 0.007 0.026", pos=f"0 {sg * 0.021} 0.0236")
                + '</body>'
                for s, sg in (("l", 1), ("r", -1)))
            + '</body>')
    l7 = _link(7, "0.088 0 0", "1 1 0 0", f'<geom {cap} size="0.04" fromto="0 0 -0.02 0 0 0.09"/>'
               + _cyl(WHITE, 0.043, "0 0 -0.04", "0 0 0.07") + _cyl(DARK, 0.038, "0 0 0.07", "0 0 0.107"), hand)
    l6 = _link(6, "0 0 0", "1 1 0 0", f'<geom {cap} size="0.045" fromto="0 0 -0.03 0.088 0 0"/>'
               + _cyl(WHITE, 0.052, "0 0 -0.07", "0 0 0.03") + _cyl(DARK, 0.042, "0 0 -0.078", "0 0 -0.07")
               + _cap(WHITE, 0.045, z0, "0.088 0 0"), l7)
    l5 = _link(5, "-0.0825 0.384 0", "1 -1 0 0", f'<geom {cap} size="0.05" fromto="0 0 -0.26 0 0 -0.02"/>'
               + _cyl(WHITE, 0.048, "0 0 -0.23", "0 0 -0.07") + _cyl(DARK, 0.057, "0 0 -0.235", "0 0 -0.225"), l6)
    l4 = _link(4, "0.0825 0 0", "1 1 0 0", f'<geom {cap} size="0.05" fromto="0 0 0 -0.0825 0.384 0"/>'
               + _cyl(WHITE, 0.062, "0 0 -0.062", "0 0 0.062") + _cyl(DARK, 0.048, "0 0 0.062", "0 0 0.07")
               + _cap(WHITE, 0.055, z0, "-0.0825 0.05 0")
               + _cyl(WHITE, 0.056, "-0.0825 0.05 0", "-0.0825 0.16 0"), l5)
    l3 = _link(3, "0 -0.316 0", "1 1 0 0", f'<geom {cap} size="0.05" fromto="0 0 -0.1 0.0825 0 0"/>'
               + _cyl(WHITE, 0.058, "0 0 -0.17", "0 0 -0.03") + _cyl(DARK, 0.059, "0 0 -0.17", "0 0 -0.16")
               + _cap(WHITE, 0.055, "0 0 -0.03", "0.0825 0 0"), l4)
    l2 = _link(2, "0 0 0", "1 -1 0 0", f'<geom {cap} size="0.05" fromto="0 0 0 0 -0.316 0"/>'
               + _cyl(WHITE, 0.065, "0 0 -0.065", "0 0 0.065") + _cyl(DARK, 0.05, "0 0 0.065", "0 0 0.073")
               + _cyl(WHITE, 0.058, z0, "0 -0.18 0"), l3)
    l1 = _link(1, "0 0 0.333", "1 0 0 0", f'<geom {cap} size="0.05" fromto="0 0 -0.2 0 0 0"/>'
               + _cyl(WHITE, 0.06, "0 0 -0.205", "0 0 -0.03"), l2)
    x, y, z = base_pos
    return (f'<body name="robot_base" pos="{x} {y} {z}" euler="0 0 {math.radians(base_yaw_deg)}">'
            + _column_xml(z) +
            f'<geom type="cylinder" size="0.08 0.05" pos="0 0 0.05" rgba="0.92 0.92 0.95 1" {HID}/>'
            + _vis("cylinder", GREY, size="0.13 0.004", pos="0 0 0")
            + _plate_xml(z)
            + _vis("cylinder", WHITE, size="0.09 0.06", pos="0 0 0.06")
            + _vis("box", WHITE, size="0.06 0.075 0.045", pos="-0.075 0 0.045")
            + _vis("cylinder", DARK, size="0.062 0.006", pos="0 0 0.126")
            + EGO_CAMERA +
            f'{l1}</body>')


EGO_CAMERA = '<camera name="ego_view" pos="-0.35 0.12 1.0" xyaxes="0 -1 0 0.62 0 0.78" fovy="70"/>'


def _column_xml(z: float) -> str:
    return f'<geom type="cylinder" size="0.12 {z / 2}" pos="0 0 {-z / 2}" rgba="0.25 0.27 0.3 1"/>'


def _plate_xml(z: float) -> str:
    return _vis("cylinder", DARK, size="0.2 0.01", pos=f"0 0 {-z + 0.01}")


def pedestal_xml(z: float) -> str:
    """Column to the floor, floor plate and the ego camera for a mount that brings its own arm (inside robot_base)."""
    return _column_xml(z) + _plate_xml(z) + EGO_CAMERA


def actuators_xml(finger_open: float = FINGER_OPEN) -> str:
    arm = "".join(f'<position name="a{i + 1}" joint="joint{i + 1}" kp="{KP[i]}" kv="{KP[i] / 15}" '
                  f'ctrlrange="{RANGES[i][0]} {RANGES[i][1]}"/>' for i in range(7))
    fing = "".join(f'<position name="g_{s}" joint="finger_{s}" kp="800" kv="30" ctrlrange="0 {finger_open}" '
                   f'forcerange="-40 40"/>' for s in ("l", "r"))
    return arm + fing
