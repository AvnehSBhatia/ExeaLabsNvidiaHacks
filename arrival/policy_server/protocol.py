"""Wire format shared by VLAClient, the mock server and the GR00T shim.

POST /act   {"video.ego_view": b64 PNG, "video.wrist": b64 PNG, "state.single_arm": [7], "state.gripper": [1],
             "annotation.human.task_description": str}
        ->  {"action.single_arm": [[7] * T], "action.gripper": [[1] * T]}   (absolute joint targets, T >= 1)
GET /health -> {"ok": true, ...}

Two schemas. "arm" (default, fixed base): state and action steps are 8 floats, 7 arm joints + gripper. "mobile": 11
floats, the same 8 plus the base x, y, yaw in the room frame, sent as "state.base" [3] and returned as "action.base"
[[3] * T]. A sim state is always 11 wide; the arm schema drops the base columns on purpose (the server then returns 8
wide chunks and the sim holds the base). Any other width is a ProtocolError.

The keys match configs/gr00t/arrival_panda_config.py and the meta/modality.json of datasets written by `arrival sim`.
"""

from __future__ import annotations

import base64
import io

import numpy as np

VIDEO_KEYS = ("ego_view", "wrist")
TASK_KEY = "annotation.human.task_description"
ARM_DOF = 7
ACTION_KEYS = ("action.single_arm", "action.gripper")
BASE_DOF = 3
SCHEMA_DIM = {"arm": ARM_DOF + 1, "mobile": ARM_DOF + 1 + BASE_DOF}  # 8 and 11 floats per state and per action step


def _schema(schema: str) -> int:
    if schema not in SCHEMA_DIM:
        raise ProtocolError(f"unknown schema {schema!r}; choose from {sorted(SCHEMA_DIM)}")
    return SCHEMA_DIM[schema]


class ProtocolError(ValueError):
    """A request or response that does not follow the contract."""


def encode_image(img: np.ndarray) -> str:
    from PIL import Image

    if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
        raise ProtocolError("camera frame must be an HxWx3 uint8 array")
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def decode_image(s: str) -> np.ndarray:
    from PIL import Image

    if not isinstance(s, str):
        raise ProtocolError("encoded image must be a base64 string")
    try:
        return np.asarray(Image.open(io.BytesIO(base64.b64decode(s, validate=True))).convert("RGB"))
    except Exception as e:
        raise ProtocolError(f"undecodable image: {e}") from e


def encode_request(obs: dict, instruction: str, schema: str = "arm") -> dict:
    """obs has 'state' (8 or 11 floats) and one frame per camera; the wrist frame is sent whenever the env has it.
    schema "arm" sends the first 8 state floats (an 11-float sim state loses its base columns); "mobile" needs 11."""
    if not isinstance(instruction, str) or not instruction.strip():
        raise ProtocolError("instruction must be a non-empty string")
    state = np.asarray(obs["state"], dtype=float)
    dim = _schema(schema)
    if state.ndim != 1 or state.shape[0] < dim or state.shape[0] not in SCHEMA_DIM.values():
        raise ProtocolError(f"state must be 8 or 11 floats and at least the {dim} the {schema!r} schema needs "
                            f"(7 joints + gripper{' + base x, y, yaw' if schema == 'mobile' else ''}), "
                            f"got shape {list(state.shape)}")
    state = state[:dim]
    if not np.isfinite(state).all():
        raise ProtocolError("state has non-finite values")
    for k in VIDEO_KEYS:
        if k in obs and (np.ndim(obs[k]) != 3 or np.shape(obs[k])[-1] != 3):
            raise ProtocolError(f"{k} frame must be HxWx3, got shape {list(np.shape(obs[k]))}")
    body = {f"video.{k}": encode_image(obs[k]) for k in VIDEO_KEYS if k in obs}
    if "video.ego_view" not in body:
        raise ProtocolError("obs has no ego_view frame")
    body["state.single_arm"] = state[:ARM_DOF].tolist()
    body["state.gripper"] = state[ARM_DOF:ARM_DOF + 1].tolist()
    if schema == "mobile":
        body["state.base"] = state[ARM_DOF + 1:].tolist()
    body[TASK_KEY] = instruction
    return body


def decode_request(body: dict) -> dict:
    """Server side: {'video': {name: HxWx3}, 'state': [8] or [11] with state.base, 'instruction': str}; raises ProtocolError when malformed."""
    if not isinstance(body, dict):
        raise ProtocolError("request is not a JSON object")
    video = {}
    for k in VIDEO_KEYS:
        if f"video.{k}" not in body:
            raise ProtocolError(f"missing video.{k}")
        video[k] = decode_image(body[f"video.{k}"])
    arm, grip = _vec(body, "state.single_arm", ARM_DOF), _vec(body, "state.gripper", 1)
    parts = [arm, grip] + ([_vec(body, "state.base", BASE_DOF)] if "state.base" in body else [])
    if not isinstance(body.get(TASK_KEY), str) or not body[TASK_KEY].strip():
        raise ProtocolError(f"{TASK_KEY} must be a non-empty string")
    return {"video": video, "state": np.concatenate(parts), "instruction": body[TASK_KEY]}


def _vec(body: dict, key: str, n: int) -> np.ndarray:
    if key not in body:
        raise ProtocolError(f"missing {key}")
    v = _numeric_array(body[key], key)
    if v.shape != (n,) or not np.isfinite(v).all():
        raise ProtocolError(f"{key} must be {n} finite floats, got shape {list(v.shape)}")
    return v


def _numeric_array(value, key: str) -> np.ndarray:
    try:
        raw = np.asarray(value, dtype=object)
        if any(isinstance(item, (bool, np.bool_)) for item in raw.flat):
            raise ProtocolError(f"{key} must contain numbers, not booleans")
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as e:
        raise ProtocolError(f"{key} is not numeric: {e}") from e


def encode_response(chunk: np.ndarray) -> dict:
    """chunk [T,8] (7 arm targets + gripper) or [T,11] (plus base x, y, yaw) -> response body."""
    try:
        chunk = np.asarray(chunk, dtype=float)
    except (TypeError, ValueError, OverflowError) as e:
        raise ProtocolError(f"chunk is not numeric: {e}") from e
    if chunk.ndim != 2 or chunk.shape[1] not in SCHEMA_DIM.values() or chunk.shape[0] < 1:
        raise ProtocolError(f"chunk must be [T,8] (arm schema) or [T,11] (mobile schema), got shape {list(chunk.shape)}")
    if not np.isfinite(chunk).all():
        raise ProtocolError("chunk has non-finite values")
    out = {"action.single_arm": chunk[:, :ARM_DOF].tolist(), "action.gripper": chunk[:, ARM_DOF:ARM_DOF + 1].tolist()}
    if chunk.shape[1] == SCHEMA_DIM["mobile"]:
        out["action.base"] = chunk[:, ARM_DOF + 1:].tolist()
    return out


def decode_response(out: dict, schema: str | None = None) -> np.ndarray:
    """Response body -> chunk [T,8], or [T,11] when it carries action.base; schema (if given) says which is required.
    Raises ProtocolError on missing keys, wrong shapes, non-finite values or T < 1."""
    if not isinstance(out, dict):
        raise ProtocolError("response is not a JSON object")
    parts = []
    keys = ACTION_KEYS + (("action.base",) if "action.base" in out else ())
    if schema is not None and len(keys) + 0 != (3 if schema == "mobile" else 2):
        raise ProtocolError(f"the {schema!r} schema expects {_schema(schema)} floats per step but the server returned "
                            f"{'11 (with action.base)' if len(keys) == 3 else '8 (no action.base)'}")
    for key, n in zip(keys, (ARM_DOF, 1, BASE_DOF)):
        if key not in out:
            raise ProtocolError(f"response is missing {key}")
        a = _numeric_array(out[key], key)
        if a.ndim != 2 or a.shape[1] != n:
            raise ProtocolError(f"{key} must be a chunk of shape [T,{n}], got {list(a.shape)}")
        if a.shape[0] < 1:
            raise ProtocolError(f"{key} is an empty chunk")
        if not np.isfinite(a).all():
            raise ProtocolError(f"{key} has non-finite values")
        parts.append(a)
    if any(part.shape[0] != parts[0].shape[0] for part in parts[1:]):
        lengths = ", ".join(f"{key}={part.shape[0]}" for key, part in zip(keys, parts))
        raise ProtocolError(f"chunk lengths differ: {lengths}")
    return np.concatenate(parts, axis=1)
