"""Dynamic instruction generation with Nemotron Lightning (Token Factory).

Generates diverse natural language robot instructions per task family and object/fixture,
maintaining a strict train vs held-out evaluation split (POL-2).
Results are cached locally so tests and offline runs work without API keys.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from arrival.tasks.propose import WORDINGS

DEFAULT_CACHE = Path(__file__).resolve().parents[1] / "models" / "instruction_cache.json"

FALLBACK_TEMPLATES = {
    "counter_to_sink": [
        "Pick up the {name} from the counter and place it in the sink.",
        "Put the {name} in the sink.",
        "Move the {name} off the counter into the sink.",
        "Take the {name} and drop it in the sink.",
        "Carefully grab the {name} and transfer it to the sink basin.",
        "Clear the counter by moving the {name} into the kitchen sink.",
        "Set the {name} down inside the sink.",
        "Retrieve the {name} from the countertop and put it into the sink.",
    ],
    "counter_to_counter": [
        "Pick up the {name} from the counter and place it onto the other counter.",
        "Put the {name} onto the other counter.",
        "Move the {name} off the counter onto the other counter.",
        "Take the {name} and drop it onto the other counter.",
        "Transfer the {name} from this counter across to the opposite counter.",
        "Shift the {name} over to the second countertop.",
        "Relocate the {name} onto the other counter space.",
        "Lift the {name} and set it down on the adjoining counter.",
    ],
    "open_drawer": [
        "Open the top drawer.",
        "Pull the top drawer of the {name} open.",
        "Slide open the drawer under the counter.",
        "Pull out the top drawer on the {name}.",
        "Gently slide the {name} drawer open.",
        "Open up the kitchen drawer.",
        "Pull the upper drawer handle to open it.",
        "Access the counter drawer by sliding it outward.",
    ],
    "open_door": [
        "Open the cabinet door.",
        "Swing the {name} cabinet door open.",
        "Pull the cabinet door open.",
        "Swing open the lower cabinet door on the {name}.",
        "Grip the cabinet door handle and pull it outward.",
        "Open up the cupboard door beneath the counter.",
        "Unlatch and swing the cabinet door open.",
        "Pull open the door to reveal the cabinet interior.",
    ],
    "open_oven": [
        "Open the oven.",
        "Pull the oven door down.",
        "Open the oven door of the {name}.",
        "Lower the oven door until it is flat.",
    ],
    "open_dishwasher": [
        "Open the dishwasher.",
        "Pull the dishwasher door open.",
        "Open the {name} door.",
        "Fold the dishwasher door down.",
    ],
    "turn_on_faucet": [
        "Turn on the faucet.",
        "Lift the faucet lever to start the water.",
        "Turn the tap on.",
        "Push the faucet handle up to run the water.",
    ],
    "load_dishwasher": [
        "Put the {name} in the dishwasher.",
        "Load the {name} into the dishwasher rack.",
        "Move the {name} from the counter into the dishwasher.",
        "Set the {name} on the pulled-out dishwasher rack.",
    ],
    "unload_dishwasher": [
        "Take the {name} out of the dishwasher.",
        "Unload the {name} onto the counter.",
        "Move the {name} from the dishwasher rack to the counter.",
        "Lift the {name} off the dishwasher rack and put it on the counter.",
    ],
}


def _cache_key(family: str, object_name: str | None, fixture_name: str | None) -> str:
    return f"{family}::obj={object_name or 'none'}::fx={fixture_name or 'none'}"


def load_cache(cache_path: Path | str | None = None) -> dict:
    p = Path(cache_path) if cache_path else DEFAULT_CACHE
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cache(cache: dict, cache_path: Path | str | None = None) -> None:
    p = Path(cache_path) if cache_path else DEFAULT_CACHE
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cache, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass


def generate_instructions(family: str, object_name: str | None = None,
                          fixture_name: str | None = None, n: int = 8,
                          role: str = "lightning", eval_ratio: float = 0.25,
                          cache_path: Path | str | None = None,
                          log=print) -> dict[str, list[str]]:
    """Generate diverse instructions split into 'train' and 'eval' sets.

    Uses Nemotron Lightning via Nebius Token Factory if available, with offline
    caching and deterministic template fallback.
    """
    key = _cache_key(family, object_name, fixture_name)
    cache = load_cache(cache_path)
    if key in cache:
        return cache[key]

    instructions: list[str] = []
    source = "token_factory"

    from arrival.models.token_factory import TokenFactoryNotConfigured, complete

    name = object_name or fixture_name or "object"
    prompt = (
        f"Generate {n} distinct, natural imperative instructions commanding a kitchen robot to perform this task:\n"
        f"Task family: {family}\n"
        f"Target/object: {name}\n\n"
        f"Output ONLY a valid JSON array of {n} strings, e.g. [\"Do X\", \"Please do Y\"]."
    )

    try:
        raw = complete(
            messages=[
                {"role": "system", "content": "You are an AI generating diverse robot instructions. Return only a JSON list of strings."},
                {"role": "user", "content": prompt},
            ],
            role=role,
        )
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                instructions = [x.strip() for x in parsed if x.strip()]
    except (TokenFactoryNotConfigured, Exception) as e:
        source = "template_fallback"
        templates = FALLBACK_TEMPLATES.get(family, WORDINGS.get(family, ["Do task {name}"]))
        instructions = [t.format(name=name) for t in templates]

    if not instructions:
        templates = FALLBACK_TEMPLATES.get(family, ["Do {name}"])
        instructions = [t.format(name=name) for t in templates]

    # Split into train and held-out evaluation sets
    eval_count = max(1, int(len(instructions) * eval_ratio))
    split = {
        "source": source,
        "train": instructions[:-eval_count],
        "eval": instructions[-eval_count:],
    }

    cache[key] = split
    save_cache(cache, cache_path)
    return split
