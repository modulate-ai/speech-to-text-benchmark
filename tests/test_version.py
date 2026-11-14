import os
import subprocess
import sys
from pathlib import Path

import pytest

import benchmark
from engines import ENGINES

ROOT = Path(benchmark.__file__).resolve().parent


def test_version_flag_prints_exact_output_without_loading_env_or_constructing_engines(
    monkeypatch, capsys
):
    def fail(*args, **kwargs):
        pytest.fail("--version must not load .env or construct an engine")

    monkeypatch.setattr(sys, "argv", ["benchmark.py", "--version"])
    monkeypatch.setattr(benchmark, "load_dotenv", fail)
    for engine_cls in ENGINES.values():
        monkeypatch.setattr(engine_cls, "__init__", fail)

    with pytest.raises(SystemExit) as exc:
        benchmark.main()

    assert exc.value.code == 0
    assert capsys.readouterr().out.endswith(" 0.1.0\n")


def test_version_command_succeeds_without_api_keys():
    env = os.environ.copy()
    for key in (
        "VELMA_API_KEY",
        "DEEPGRAM_API_KEY",
        "ASSEMBLYAI_API_KEY",
        "HF_TOKEN",
        "OPENROUTER_API_KEY",
        "WHISPER_CPP_DIR",
    ):
        env.pop(key, None)

    result = subprocess.run(
        [sys.executable, str(ROOT / "benchmark.py"), "--version"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "benchmark.py 0.1.0\n"
