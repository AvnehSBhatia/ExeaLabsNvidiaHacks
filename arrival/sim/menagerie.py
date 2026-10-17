"""MuJoCo Menagerie Franka Panda (Apache-2.0) rewritten to the names the sim relies on.

The meshes are not committed (34 MB): `python scripts/fetch_menagerie.py` puts them in `arrival/assets/menagerie/`.
`arm_fragments` parses the Menagerie `panda.xml` and returns MJCF fragments that drop into `build_mjcf` in place
of the procedural arm: same body, joint, site and camera names, the repo's collision classes, and the repo's
position actuators, so `collect`, `ik` and the policies do not know which arm is loaded.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

from arrival.sim import robot

PANDA_DIR = Path(__file__).resolve().parents[1] / "assets" / "menagerie" / "franka_emika_panda"
FINGER_OPEN = 0.04  # per finger: the real hand opens 8 cm, the procedural one 10 cm
EEF_Z = 0.103  # hand-frame height of the fingertip pad centres (procedural arm: 0.115)
RENAMES = {"link0": "robot_base", "left_finger": "finger_l", "right_finger": "finger_r",
           "finger_joint1": "finger_l", "finger_joint2": "finger_r"}
CLASSES = {"panda": "panda", "finger": "panda_finger", "visual": "panda_visual", "collision": "panda_collision"}
PAD_FRICTION = {"friction": "2.0 0.05 0.001", "condim": "6"}  # same pads as robot.py's fingers
FETCH_HINT = "Menagerie Panda not found; run: python scripts/fetch_menagerie.py"


def available() -> bool:
    return (PANDA_DIR / "panda.xml").is_file()


def _cls(name: str) -> str:
    return CLASSES.get(name, "panda_" + name)


def _s(el: ET.Element) -> str:
    return ET.tostring(el, encoding="unicode")


def arm_fragments(base_pos: tuple[float, float, float], base_yaw_deg: float) -> dict[str, str]:
    """{"meshdir", "defaults", "assets", "body", "actuators"} for the Menagerie arm mounted like robot.arm_xml."""
    if not available():
        raise FileNotFoundError(FETCH_HINT)
    root = ET.parse(PANDA_DIR / "panda.xml").getroot()
    for el in root.iter():
        for attr in ("class", "childclass"):
            if attr in el.attrib:
                el.set(attr, _cls(el.get(attr)))
        if el.tag in ("body", "joint") and el.get("name") in RENAMES:
            el.set("name", RENAMES[el.get("name")])
        if el.tag == "material" and "name" in el.attrib:
            el.set("name", "panda_" + el.get("name"))
        if el.tag == "mesh":
            el.set("name", "panda_" + el.get("name", Path(el.get("file")).stem))
        if el.tag == "geom":
            if "material" in el.attrib:
                el.set("material", "panda_" + el.get("material"))
            if "mesh" in el.attrib:
                el.set("mesh", "panda_" + el.get("mesh"))
    default = root.find("default/default")
    for d in default.iter("default"):
        if d.get("class") == "panda_collision":
            d.find("geom").attrib.update(robot.COL_ATTRS)
        for g in d.findall("general"):
            d.remove(g)
    base = root.find("worldbody/body")
    x, y, z = base_pos
    base.set("pos", f"{x} {y} {z}")
    base.set("euler", f"0 0 {math.radians(base_yaw_deg)}")
    for b in base.iter("body"):
        if b is not base:
            b.set("gravcomp", "1")
    hand = next(b for b in base.iter("body") if b.get("name") == "hand")
    ET.SubElement(hand, "camera", robot.WRIST_CAMERA)
    ET.SubElement(hand, "site", name="eef", pos=f"0 0 {EEF_Z}", size="0.01", rgba="1 0 0 0")
    for b in base.iter("body"):
        if b.get("name") in ("finger_l", "finger_r"):
            for g in b.findall("geom"):
                if g.get("class") != "panda_visual":
                    g.attrib.update(PAD_FRICTION)
    base.extend(ET.fromstring(f"<x>{robot.pedestal_xml(z)}</x>"))
    contact = root.find("contact")
    for ex in (contact if contact is not None else []):
        for attr in ("body1", "body2"):
            ex.set(attr, RENAMES.get(ex.get(attr), ex.get(attr)))
    return {"meshdir": str(PANDA_DIR / "assets"), "defaults": _s(default),
            "assets": "".join(_s(el) for el in root.find("asset")), "body": _s(base),
            "actuators": robot.actuators_xml(FINGER_OPEN), "contact": _s(contact) if contact is not None else ""}
