"""Object label -> catalog id, and the physical spec (shape, nominal size, mass, friction) behind each id."""

from __future__ import annotations

import random

# shape, nominal size [x, y, z] (m), mass (kg) at the nominal size, sliding friction, rgba
OBJECT_SPECS = {
    "mug": ("cylinder", [0.08, 0.08, 0.10], 0.30, 1.0, "0.85 0.3 0.2 1"),
    "cup": ("cylinder", [0.07, 0.07, 0.09], 0.20, 1.0, "0.3 0.55 0.85 1"),
    "bowl": ("cylinder", [0.14, 0.14, 0.06], 0.25, 1.0, "0.9 0.9 0.8 1"),
    "plate": ("cylinder", [0.22, 0.22, 0.03], 0.40, 0.8, "0.92 0.92 0.9 1"),  # a deep plate: the rim is what the gripper holds
    "sponge": ("box", [0.10, 0.06, 0.03], 0.03, 1.2, "0.95 0.85 0.2 1"),
    "apple": ("sphere", [0.076, 0.076, 0.076], 0.18, 0.9, "0.75 0.15 0.12 1"),
    "bottle": ("bottle", [0.066, 0.066, 0.22], 0.55, 0.6, "0.55 0.75 0.9 0.7"),
    "pan": ("pan", [0.24, 0.24, 0.045], 0.90, 0.6, "0.2 0.2 0.22 1"),
    "spatula": ("box", [0.035, 0.22, 0.010], 0.05, 0.7, "0.3 0.3 0.32 1"),
}
OBJECT_MAP = {label: label for label in OBJECT_SPECS}
PICKABLE = ("mug", "cup", "sponge", "apple", "bottle")  # fit the 10 cm gripper opening and are held by the body;
# the 1 cm spatula sits under the grasp height the expert uses, so it is scenery until the grasp rule changes
RIM_HELD = ("bowl", "plate")  # hollow, pinched by the rim nearest the robot (MAN-4)
HANDLE_HELD = ("pan",)  # pinched across the handle, close to the body
GRASPABLE = PICKABLE + RIM_HELD + HANDLE_HELD
SIZE_VARIATION = 0.08  # +-8% on the footprint and on the height, drawn separately


def catalog_id(label: str) -> str:
    return OBJECT_MAP.get(label, "box")


def object_spec(cid: str, size: list[float]) -> dict:
    """Shape, mass scaled from the nominal mass by volume, friction and colour for an object of `size`."""
    shape, nominal, mass, friction, rgba = OBJECT_SPECS.get(cid, ("box", list(size), 0.2, 1.0, "0.5 0.5 0.9 1"))
    ratio = (size[0] * size[1] * size[2]) / (nominal[0] * nominal[1] * nominal[2])
    return {"shape": shape, "mass": round(mass * ratio, 4), "friction": friction, "rgba": rgba}


def sample_size(cid: str, rng: random.Random) -> list[float]:
    """Nominal size with seeded variation; round objects stay round."""
    nominal = OBJECT_SPECS[cid][1]
    kxy, kz = (1 + rng.uniform(-SIZE_VARIATION, SIZE_VARIATION) for _ in range(2))
    return [round(nominal[0] * kxy, 4), round(nominal[1] * kxy, 4), round(nominal[2] * kz, 4)]
