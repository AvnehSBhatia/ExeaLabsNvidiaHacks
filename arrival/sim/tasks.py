"""Sim task families: pick-and-place (counter_to_sink, counter_to_counter, put_in_drawer, load/unload_dishwasher),
drawer and door tasks (open/close), the oven and dishwasher doors, the faucet lever and wiping."""

from __future__ import annotations

import random

from arrival.catalog.objects import PICKABLE
from arrival.tasks.propose import LIGHTING, POSE_OFFSETS, WORDINGS, top_z

SIM_FAMILIES = ("counter_to_sink", "counter_to_counter", "open_drawer", "open_door", "close_drawer", "close_door",
                "put_in_drawer", "wipe_counter", "open_oven", "open_dishwasher", "turn_on_faucet",
                "load_dishwasher", "unload_dishwasher")
REACH = 0.78  # max horizontal eef reach from the arm base used when placing the base
HOLDOUT_WORDING = 3  # docs/07-eval.md: template 4 and the (0, +0.08) offset are for evaluation only
HOLDOUT_OFFSET = [0.0, 0.08]
DISHWASHER_SIDE = 0.55  # the base parks this far beside the dishwasher centre, clear of its open door
SIM_WORDINGS = {
    **WORDINGS,
    "open_drawer": ["Open the top drawer.", "Pull the top drawer of the {name} open.",
                    "Slide open the drawer under the counter."],
    "open_door": ["Open the cabinet door.", "Swing the {name} cabinet door open.", "Pull the cabinet door open."],
    "close_drawer": ["Close the drawer.", "Push the open drawer of the {name} shut.", "Slide the drawer closed."],
    "close_door": ["Close the cabinet door.", "Push the {name} cabinet door shut.", "Swing the open door closed."],
    "put_in_drawer": ["Put the {name} in the open drawer.", "Place the {name} into the drawer.",
                      "Move the {name} from the counter into the drawer."],
    "wipe_counter": ["Wipe the counter with the {name}.", "Clean the counter top with the {name}.",
                     "Scrub the counter using the {name}."],
    "open_oven": ["Open the oven.", "Pull the oven door down.", "Open the oven door of the {name}."],
    "open_dishwasher": ["Open the dishwasher.", "Pull the dishwasher door open.", "Open the {name} door."],
    "turn_on_faucet": ["Turn on the faucet.", "Lift the faucet lever to start the water.", "Turn the tap on."],
    "load_dishwasher": ["Put the {name} in the dishwasher.", "Load the {name} into the dishwasher rack.",
                        "Move the {name} from the counter into the dishwasher."],
    "unload_dishwasher": ["Take the {name} out of the dishwasher.", "Unload the {name} onto the counter.",
                          "Move the {name} from the dishwasher rack to the counter."],
}


def _fx(twin: dict, fid: str) -> dict:
    return next(f for f in twin["fixtures"] if f["id"] == fid)


def make_tasks(twin: dict, families: list[str], n: int, seed: int = 0, people: int = 0, holdout: bool = False,
               split: str = "train", use_generated_instructions: bool = False,
               log=print) -> list[dict]:
    """n task variations cycling through the requested families; unreachable families are skipped and logged.
    holdout=False draws the training set: wording HOLDOUT_WORDING (families with 4 templates) and HOLDOUT_OFFSET
    are excluded. holdout=True draws only those; families with 3 templates keep all 3 either way.
    use_generated_instructions draws from the Nemotron cache (or the template fallback) for `split` instead."""
    rng = random.Random(seed)
    counters = [f for f in twin["fixtures"] if f["type"] == "counter"]
    sinks = [f for f in twin["fixtures"] if f["type"] == "sink"]
    pools: dict[str, list[dict]] = {}
    for fam in families:
        pool = []
        if fam in ("counter_to_sink", "counter_to_counter"):
            for o in twin["objects"]:
                # bowl is held by its rim (MAN-4); plate, pan and spatula stay out of the pick pool
                if o["catalog_id"] not in PICKABLE + ("bowl",):
                    continue
                src = next((c for c in counters if abs(o["position"][0] - c["position"][0]) < c["size"][0] / 2), None)
                if src is None:
                    continue
                if fam == "counter_to_sink":
                    tgts = [s for s in sinks if abs(s["position"][0] - o["position"][0]) < 2 * REACH - 0.3]
                    tgt_xy = [(s, s["position"][0]) for s in tgts]
                else:
                    tgt_xy = []
                    for c in counters:
                        if c is src:
                            continue
                        edge = c["position"][0] - (c["size"][0] / 2 - 0.12) * (1 if c["position"][0] > o["position"][0] else -1)
                        if abs(edge - o["position"][0]) < 1.0:  # base sits between source and target
                            tgt_xy.append((c, edge))
                for tgt, tx in tgt_xy:
                    pool.append({"object_id": o["id"], "object_label": o["label"], "source": src["id"],
                                 "target": tgt["id"], "target_x": tx, "start": list(o["position"])})
        elif fam == "put_in_drawer":
            from arrival.sim.scene import drawer_x
            for o in twin["objects"]:
                if o["catalog_id"] != "mug":
                    continue
                src = next((c for c in counters if abs(o["position"][0] - c["position"][0]) < c["size"][0] / 2), None)
                if src is None or abs(o["position"][0] - (src["position"][0] + drawer_x(src, twin["fixtures"]))) > 0.6:
                    continue  # the base parks beside the open drawer, so the mug must be within reach of that spot
                pool.append({"object_id": o["id"], "object_label": o["label"], "source": src["id"], "fixture": src["id"],
                             "start": list(o["position"])})
        elif fam == "wipe_counter":
            for o in twin["objects"]:
                if o["catalog_id"] != "sponge":
                    continue
                src = next((c for c in counters if abs(o["position"][0] - c["position"][0]) < c["size"][0] / 2), None)
                if src is not None:
                    pool.append({"object_id": o["id"], "object_label": o["label"], "source": src["id"],
                                 "fixture": src["id"], "start": list(o["position"])})
        elif fam in ("open_drawer", "open_door", "close_drawer", "close_door"):
            pool = [{"fixture": c["id"], "name": c["id"].replace("_", " ")} for c in counters]
        elif fam == "open_oven":
            pool = [{"fixture": f["id"], "name": f["id"].replace("_", " ")} for f in twin["fixtures"] if f["type"] == "stove"]
        elif fam == "open_dishwasher":
            pool = [{"fixture": f["id"], "name": f["id"].replace("_", " ")} for f in twin["fixtures"]
                    if f["type"] == "dishwasher"]
        elif fam == "turn_on_faucet":
            pool = [{"fixture": s["id"], "name": "faucet"} for s in sinks
                    if any(abs(s["position"][0] - c["position"][0]) < c["size"][0] / 2 for c in counters)]
        elif fam in ("load_dishwasher", "unload_dishwasher"):
            # the base parks beside the dishwasher in front of the counter, so the object (or its spot on the
            # counter for unloading) has to be within reach of that parking spot
            for d in (f for f in twin["fixtures"] if f["type"] == "dishwasher"):
                for o in twin["objects"]:
                    if o["catalog_id"] not in PICKABLE:
                        continue
                    src = next((c for c in counters if abs(o["position"][0] - c["position"][0]) < c["size"][0] / 2), None)
                    if src is None:
                        continue
                    side = 1 if o["position"][0] > d["position"][0] else -1
                    park = d["position"][0] + side * DISHWASHER_SIDE
                    if abs(o["position"][0] - park) > 0.45 or abs(o["position"][0] - d["position"][0]) < d["size"][0] / 2 + 0.1:
                        continue
                    pool.append({"object_id": o["id"], "object_label": o["label"], "source": src["id"],
                                 "fixture": d["id"], "start": list(o["position"])})
        if not pool:
            log(f"[sim] family {fam}: no reachable variation in this room for a fixed-base arm; skipped")
            continue
        pools[fam] = pool
    tasks = []
    fams = list(pools)
    for i in range(n if fams else 0):
        fam = fams[i % len(fams)]
        v = dict(rng.choice(pools[fam]))
        light = rng.choice(LIGHTING)
        name = v.get("object_label") or v.get("name")
        if use_generated_instructions:
            from arrival.tasks.instructions import generate_instructions
            gen = generate_instructions(fam, object_name=v.get("object_label"), fixture_name=v.get("name"))
            options = gen.get(split, gen.get("train", SIM_WORDINGS[fam]))
            w = rng.randrange(len(options))
            instr = options[w]
        else:
            words = SIM_WORDINGS[fam]
            w = rng.choice([k for k in range(len(words)) if len(words) <= HOLDOUT_WORDING or (k == HOLDOUT_WORDING) == holdout])
            instr = words[w].format(name=name)
        t = {"family": fam, "lighting": light, "wording_index": w, "people": people, "seed": seed * 1000 + i,
             "holdout": holdout, "instruction": instr, "split": split, **v}
        if "start" in v:
            off = rng.choice([o for o in POSE_OFFSETS if (list(o) == HOLDOUT_OFFSET) == holdout])
            t["start"] = [v["start"][0] + off[0], v["start"][1] + off[1] * 0.5,
                          top_z(_fx(twin, v["source"])) + 0.06]
        t["task_id"] = f"sim_{fam}_{i:04d}"
        tasks.append(t)
    return tasks
