"""L22: one settings loader, set/unset status without values, redaction, and no other module reading env vars."""

import logging
import re

import pytest

from arrival import config

from conftest import REPO

FAKE_KEY = "nvapi-FAKEFAKEFAKEFAKE1234"


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """No known setting set, cwd in an empty dir, .env lookup on (conftest turns it off for every other test)."""
    for name in config.SETTINGS:  # setenv first so teardown removes whatever a .env load adds during the test
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.setenv("ARRIVAL_DOTENV", "1")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path / "no_repo_root")
    return tmp_path


def test_env_wins_over_dotenv_and_dotenv_fills_unset(clean_env, monkeypatch):
    (clean_env / ".env").write_text(
        "# comment\nexport TAVILY_API_KEY='from-file'\nGR00T_POLICY_URL=\"http://box:8000\"\nNOT_A_LINE\n")
    monkeypatch.setenv("TAVILY_API_KEY", "from-env")
    assert config.get("TAVILY_API_KEY") == "from-env"
    assert config.get("GR00T_POLICY_URL") == "http://box:8000"


def test_defaults_and_empty_values(clean_env, monkeypatch):
    assert config.get("TAVILY_BASE_URL") == "https://api.tavily.com/search"
    assert config.get("COSMOS_URL") is None
    assert config.get("COSMOS_URL", "http://fallback") == "http://fallback"
    monkeypatch.setenv("TOKEN_FACTORY_API_KEY", "")
    assert config.get("TOKEN_FACTORY_API_KEY") is None


def test_dotenv_can_be_turned_off(clean_env, monkeypatch):
    (clean_env / ".env").write_text("COSMOS_URL=http://cosmos\n")
    monkeypatch.setenv("ARRIVAL_DOTENV", "0")
    assert config.get("COSMOS_URL") is None
    config.load_dotenv([clean_env / ".env"])  # explicit paths are still read
    assert config.get("COSMOS_URL") == "http://cosmos"


def test_status_reports_set_without_values(clean_env, monkeypatch):
    monkeypatch.setenv("TOKEN_FACTORY_API_KEY", FAKE_KEY)
    rows = {r["name"]: r for r in config.status()}
    assert rows.keys() == config.SETTINGS.keys()
    assert rows["TOKEN_FACTORY_API_KEY"]["set"] and rows["TOKEN_FACTORY_API_KEY"]["secret"]
    assert not rows["TAVILY_API_KEY"]["set"]
    assert FAKE_KEY not in repr(rows)


@pytest.mark.parametrize("text", [
    f"Authorization: Bearer {FAKE_KEY}",
    "key tvly-abcdefghijklmnop leaked",
    '{"api_key": "abcd1234efgh"}',
    "GET http://user:hunter2pass@gpu-box:8000/act",
    "POST https://cosmos.example/v1?token=abc123def456&x=1",
])
def test_redact_scrubs_token_shapes(clean_env, text):
    out = config.redact(text)
    assert config.REDACTED in out
    for secret in (FAKE_KEY, "tvly-abcdefghijklmnop", "abcd1234efgh", "hunter2pass", "abc123def456"):
        assert secret not in out


def test_redact_scrubs_set_secret_values_and_keeps_plain_text(clean_env, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "plainsecretvalue")
    assert config.redact("upstream said: plainsecretvalue is invalid") == f"upstream said: {config.REDACTED} is invalid"
    plain = "policy vla: no answer from http://127.0.0.1:8000: [Errno 61] Connection refused"
    assert config.redact(plain) == plain
    assert config.redact("max_tokens: 1000, 16 episodes") == "max_tokens: 1000, 16 episodes"


def test_log_filter_redacts_formatted_records(clean_env):
    records = []

    class Keep(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    log = logging.getLogger("arrival.test_config")
    log.propagate = False
    h = Keep()
    log.addHandler(h)
    try:
        config.install_log_redaction(log)
        log.warning("calling with %s", f"Bearer {FAKE_KEY}")
    finally:
        log.removeHandler(h)
    assert records == [f"calling with Bearer {config.REDACTED}"]


def test_clients_read_dotenv_through_config(clean_env):
    (clean_env / ".env").write_text("GR00T_POLICY_URL=http://box:8000\nCOSMOS_URL=http://cosmos:9000\n"
                                    "TOKEN_FACTORY_API_KEY=tf-key\nTAVILY_API_KEY=tv-key\n")
    from arrival.models import tavily, token_factory
    from arrival.render.variation import CosmosClient
    from arrival.sim.policy import VLAClient

    assert VLAClient().url == "http://box:8000"
    assert CosmosClient().url == "http://cosmos:9000"
    assert token_factory.config()["api_key"] == "tf-key"
    assert tavily.config()["api_key"] == "tv-key"


def test_only_config_reads_the_environment():
    """The single-loader rule: every known setting is read through arrival.config, nowhere else."""
    names = "|".join(config.SETTINGS)
    pattern = re.compile(rf"os\.(environ|getenv)\b[^\n]*({names})")
    offenders = [str(p.relative_to(REPO)) for p in (REPO / "arrival").rglob("*.py")
                 if p.name != "config.py" and pattern.search(p.read_text(encoding="utf-8"))]
    assert offenders == []


def test_env_example_lists_every_setting():
    text = (REPO / ".env.example").read_text()
    for name in config.SETTINGS:
        assert re.search(rf"^#? ?{name}=", text, re.M), name
    assert not re.search(r"^(\w+_API_KEY)=\S", text, re.M), "no real keys in the example"


def test_env_example_lists_every_job_limit():
    from arrival.jobs import limits

    text = (REPO / ".env.example").read_text()
    for var, _ in limits.ENV.values():
        assert re.search(rf"^#? ?{var}=", text, re.M), var
