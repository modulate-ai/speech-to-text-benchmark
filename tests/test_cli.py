import os
import subprocess
import sys
from pathlib import Path

import pytest

import benchmark
from engines import ENGINES

ROOT = Path(benchmark.__file__).resolve().parent
EXPECTED_ENGINES = [
    "velma: -",
    "deepgram: -",
    "assemblyai: -",
    "whisper_hf: USD 0.027 per audio hour",
    "moonshine_tiny (local): USD 0 per audio hour",
    "moonshine_base (local): USD 0 per audio hour",
    "chirp_3: USD 0.961 per audio hour",
    "mai_transcribe_2: USD 0.1 per audio hour",
    "whisper_cpp_tiny (local): USD 0 per audio hour",
    "whisper_cpp_base (local): USD 0 per audio hour",
]


def test_list_engines_uses_class_metadata_without_loading_env_or_constructing_engines(monkeypatch, capsys):
    def fail(*args, **kwargs):
        pytest.fail("listing engines must not load .env or construct an engine")

    monkeypatch.setattr(sys, "argv", ["benchmark.py", "--list-engines"])
    monkeypatch.setattr(benchmark, "load_dotenv", fail)
    for engine_cls in ENGINES.values():
        monkeypatch.setattr(engine_cls, "__init__", fail)

    benchmark.main()

    assert capsys.readouterr().out.splitlines() == EXPECTED_ENGINES


def test_list_engines_command_exits_successfully_without_api_keys():
    env = os.environ.copy()
    for key in ("VELMA_API_KEY", "DEEPGRAM_API_KEY", "ASSEMBLYAI_API_KEY", "HF_TOKEN",
                "OPENROUTER_API_KEY", "WHISPER_CPP_DIR"):
        env.pop(key, None)

    result = subprocess.run(
        [sys.executable, str(ROOT / "benchmark.py"), "--list-engines"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == EXPECTED_ENGINES
