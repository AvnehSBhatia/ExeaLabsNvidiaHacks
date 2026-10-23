"""Golden wire format for the policy server. The golden file was written from arrival.policy_server.protocol itself:
it pins the format against accidental change, it is not a capture from a real Isaac-GR00T server."""

import json
from pathlib import Path

import numpy as np
import pytest

from arrival.policy_server import gr00t_shim
from arrival.policy_server.protocol import (ProtocolError, decode_image, decode_request, decode_response,
                                            encode_request, encode_response)

GOLDEN = json.loads((Path(__file__).parent / "golden" / "policy_wire.json").read_text())
STATE11 = np.array([0.1, -0.2, 0.3, -1.5, 0.0, 1.2, 0.4, 0.04, 1.5, 0.8, 0.25])
REQ_KEYS = {"video.ego_view", "video.wrist", "state.single_arm", "state.gripper",
            "annotation.human.task_description"}


def _obs(state=STATE11):
    ego = np.zeros((2, 2, 3), np.uint8)
    ego[0, 0], ego[1, 1] = [255, 0, 0], [0, 0, 255]
    return {"state": state, "ego_view": ego, "wrist": np.full((2, 2, 3), 200, np.uint8)}


@pytest.mark.parametrize("schema,keys,dim", [("arm", REQ_KEYS, 8), ("mobile", REQ_KEYS | {"state.base"}, 11)])
def test_golden_request_matches_encoder(schema, keys, dim):
    body = encode_request(_obs(), "open the drawer", schema)
    g = GOLDEN[schema]["request"]
    assert set(body) == set(g) == keys
    for k in keys - {"video.ego_view", "video.wrist"}:
        assert body[k] == g[k]
    assert len(g["state.single_arm"]) == 7 and len(g["state.gripper"]) == 1
    assert len(g.get("state.base", [])) == dim - 8
    for k in ("video.ego_view", "video.wrist"):  # pixels, not PNG bytes (zlib output may differ by version)
        assert np.array_equal(decode_image(body[k]), decode_image(g[k]))
    assert decode_request(g)["state"].shape == (dim,)


@pytest.mark.parametrize("schema,dim", [("arm", 8), ("mobile", 11)])
def test_golden_response_chunk_shape(schema, dim):
    r = GOLDEN[schema]["response"]
    assert np.array(r["action.single_arm"]).shape == (3, 7) and np.array(r["action.gripper"]).shape == (3, 1)
    assert ("action.base" in r) == (schema == "mobile")
    chunk = decode_response(r, schema)
    assert chunk.shape == (3, dim)
    assert np.allclose(chunk[:, 7], STATE11[7]) and np.allclose(chunk[:, 0] - STATE11[0], [0, 0.01, 0.02])
    assert encode_response(chunk) == r


def test_arm_schema_drops_base_and_mobile_keeps_it():
    assert "state.base" not in encode_request(_obs(), "x", "arm")
    assert np.allclose(decode_request(encode_request(_obs(), "x", "mobile"))["state"], STATE11)


def test_wrong_widths_refused_clearly():
    with pytest.raises(ProtocolError, match="state must be 8 or 11 floats"):
        encode_request(_obs(np.zeros(10)), "x")
    with pytest.raises(ProtocolError, match="at least the 11"):
        encode_request(_obs(np.zeros(8)), "x", "mobile")
    with pytest.raises(ProtocolError, match="unknown schema"):
        encode_request(_obs(), "x", "legged")
    with pytest.raises(ProtocolError, match=r"\[T,8\].*\[T,11\]"):
        encode_response(np.zeros((4, 9)))
    with pytest.raises(ProtocolError, match=r"\[T,8\].*\[T,11\]"):
        encode_response(np.zeros(8))
    with pytest.raises(ProtocolError, match="shape \\[T,7\\]"):
        decode_response({"action.single_arm": [[0.0] * 10], "action.gripper": [[0.0]]})
    with pytest.raises(ProtocolError, match="shape \\[T,3\\]"):
        decode_response({**GOLDEN["arm"]["response"], "action.base": [[0.0] * 2] * 3})
    with pytest.raises(ProtocolError, match="expects 11 floats.*returned 8"):
        decode_response(GOLDEN["arm"]["response"], "mobile")
    with pytest.raises(ProtocolError, match="expects 8 floats.*returned 11"):
        decode_response(GOLDEN["mobile"]["response"], "arm")
    bad_mobile = dict(GOLDEN["mobile"]["response"])
    bad_mobile["action.base"] = [[0.0] * 3]
    with pytest.raises(ProtocolError, match="chunk lengths differ.*action.base=1"):
        decode_response(bad_mobile, "mobile")
    body = dict(GOLDEN["arm"]["request"])
    body["state.single_arm"] = [0.0] * 10
    with pytest.raises(ProtocolError, match="7 finite floats, got shape \\[10\\]"):
        decode_request(body)


def test_bad_frame_shape_refused():
    with pytest.raises(ProtocolError, match="HxWx3"):
        encode_request({"state": STATE11, "ego_view": np.zeros((4, 4), np.uint8)}, "x")


@pytest.mark.parametrize("chunk", [np.zeros((0, 8)), np.full((1, 8), np.nan),
                                   np.full((1, 11), np.inf), [["bad"] * 8]])
def test_encoder_refuses_chunks_the_decoder_cannot_accept(chunk):
    with pytest.raises(ProtocolError):
        encode_response(chunk)

@pytest.mark.parametrize("instruction", [None, "", "  ", 3])
def test_policy_requests_require_a_named_task(instruction):
    with pytest.raises(ProtocolError, match="non-empty string"):
        encode_request(_obs(), instruction)
    body = GOLDEN["arm"]["request"].copy()
    body["annotation.human.task_description"] = instruction
    with pytest.raises(ProtocolError, match="non-empty string"):
        decode_request(body)

@pytest.mark.parametrize("frame", [np.full((2, 2, 3), 300, np.int16),
                                   np.full((2, 2, 3), np.nan, np.float32),
                                   np.zeros((2, 2, 3), bool)])
def test_camera_encoder_does_not_coerce_bad_pixel_types(frame):
    obs = _obs()
    obs["ego_view"] = frame
    with pytest.raises(ProtocolError, match="uint8"):
        encode_request(obs, "x")

def test_image_decoder_rejects_trailing_non_base64_data():
    valid = GOLDEN["arm"]["request"]["video.ego_view"]
    with pytest.raises(ProtocolError, match="undecodable image"):
        decode_image(valid + "!not-an-image")
    with pytest.raises(ProtocolError, match="base64 string"):
        decode_image(123)

def test_inbound_policy_vectors_reject_json_booleans():
    body = GOLDEN["arm"]["request"].copy()
    body["state.single_arm"] = [True] + [0.0] * 6
    with pytest.raises(ProtocolError, match="booleans"):
        decode_request(body)
    response = GOLDEN["arm"]["response"].copy()
    response["action.gripper"] = [[False], [0.0], [0.0]]
    with pytest.raises(ProtocolError, match="booleans"):
        decode_response(response)


def test_integers_too_large_for_a_float_are_protocol_errors():
    huge = 10 ** 400
    body = GOLDEN["arm"]["request"].copy()
    body["state.gripper"] = [huge]
    with pytest.raises(ProtocolError, match="not numeric"):
        decode_request(body)
    response = GOLDEN["arm"]["response"].copy()
    response["action.gripper"] = [[huge], [0.0], [0.0]]
    with pytest.raises(ProtocolError, match="not numeric"):
        decode_response(response)
    with pytest.raises(ProtocolError, match="not numeric"):
        encode_response([[huge] * 8])


def test_shim_maps_base_key():
    req = decode_request(GOLDEN["mobile"]["request"])
    obs = gr00t_shim.to_observation(req)
    assert obs["state"]["base"].shape == (1, 1, 3) and obs["state"]["gripper"].shape == (1, 1, 1)
    chunk = gr00t_shim.to_chunk({"single_arm": np.zeros((1, 4, 7)), "gripper": np.zeros((1, 4, 1)),
                                 "base": np.ones((1, 4, 3))})
    assert chunk.shape == (4, 11) and chunk[0, 10] == 1.0
