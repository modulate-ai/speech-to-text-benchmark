"""Offline unit tests for Oruk Fourier 2B local and hosted engines."""
import sys
import types
from contextlib import contextmanager

import numpy as np
import pytest
import soundfile as sf

from engines import ENGINES
from engines import oruk_fourier_2b as fourier
from engines.base import EngineNotConfigured, FatalEngineError


def test_registered_engine_metadata():
    assert ENGINES["oruk_fourier_2b"] is fourier.OrukFourier2B
    assert ENGINES["oruk_fourier_2b_local"] is fourier.OrukFourier2BLocal
    assert fourier.OrukFourier2B.local is False
    assert fourier.OrukFourier2B.price_per_hour == pytest.approx(0.48)
    assert fourier.OrukFourier2BLocal.local is True
    assert fourier.OrukFourier2BLocal.price_per_hour == 0.0
    assert fourier.HF_REPO == "oruk/oruk-fourier-2b"
    assert len(fourier.HF_REVISION) == 40


def test_hosted_engine_requires_api_key(monkeypatch):
    monkeypatch.delenv("ORUK_API_KEY", raising=False)
    with pytest.raises(EngineNotConfigured, match="ORUK_API_KEY"):
        fourier.OrukFourier2B()


def test_hosted_engine_transcribe_success_and_retry(tmp_path, monkeypatch):
    wav = tmp_path / "clip.wav"
    sf.write(str(wav), np.zeros(1600, dtype=np.float32), 16000)

    monkeypatch.setenv("ORUK_API_KEY", "oruk_test_dummy")
    monkeypatch.delenv("ORUK_SPEECH_API_URL", raising=False)
    monkeypatch.setattr(fourier.time, "sleep", lambda _: None)

    calls = []

    class FakeResponse:
        def __init__(self, status_code, payload=None, text=""):
            self.status_code = status_code
            self._payload = payload or {}
            self.text = text

        def json(self):
            return self._payload

    responses = iter([
        FakeResponse(429, text="rate limited"),
        FakeResponse(
            200,
            payload={
                "text": "  hello fourier  ",
                "audio_seconds": 0.1,
                "usage": {"estimated_cost_usd": 0.000013},
            },
        ),
    ])

    def fake_post(url, headers=None, data=None, files=None, timeout=None):
        calls.append({"url": url, "headers": headers, "data": data, "filename": files["file"][0]})
        return next(responses)

    monkeypatch.setattr(fourier.requests, "post", fake_post)

    eng = fourier.OrukFourier2B(max_retries=2)
    out = eng.transcribe(str(wav))
    assert out.text == "hello fourier"
    assert out.duration_s == pytest.approx(0.1)
    assert out.cost_usd == pytest.approx(0.000013)
    assert len(calls) == 2
    assert calls[0]["url"] == "https://speech-api.oruk.ai/v1/audio/transcriptions"
    assert calls[0]["headers"] == {"Authorization": "Bearer oruk_test_dummy"}
    assert calls[0]["data"] == {"model": "oruk-fourier-2b", "language": "en", "response_format": "json"}


@pytest.mark.parametrize("status_code", [401, 402, 403])
def test_hosted_engine_fatal_status_stops_without_retry(tmp_path, monkeypatch, status_code):
    wav = tmp_path / "clip.wav"
    sf.write(str(wav), np.zeros(1600, dtype=np.float32), 16000)
    monkeypatch.setenv("ORUK_API_KEY", "oruk_test_dummy")

    calls = []

    def fake_post(*args, **kwargs):
        calls.append(1)
        return types.SimpleNamespace(status_code=status_code, text="denied")

    monkeypatch.setattr(fourier.requests, "post", fake_post)
    eng = fourier.OrukFourier2B(max_retries=3)
    with pytest.raises(FatalEngineError, match=f"HTTP {status_code}"):
        eng.transcribe(str(wav))
    assert len(calls) == 1


def test_local_engine_requires_transformers_5_21(monkeypatch):
    for bad in (None, "4.57.6", "5.20.9"):
        monkeypatch.setattr(fourier, "_transformers_version", lambda v=bad: v)
        with pytest.raises(EngineNotConfigured, match=r"transformers >= 5\.21"):
            fourier.OrukFourier2BLocal()


def test_local_engine_pins_revision_resamples_and_decodes(tmp_path, monkeypatch):
    # Write a stereo 48 kHz clip to exercise mono mixdown + resampling.
    wav = tmp_path / "stereo48k.wav"
    sf.write(str(wav), np.zeros((4800, 2), dtype=np.float32), 48000)

    monkeypatch.setattr(fourier, "_transformers_version", lambda: "5.24.0")
    events = {"proc_loads": [], "model_loads": [], "requests": [], "generate": []}

    class FakeTensor:
        def __init__(self, shape):
            self.shape = shape

        def __getitem__(self, item):
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], slice):
                return FakeTensor((self.shape[0], self.shape[1] - (item[1].start or 0)))
            return self

    class FakeBatch(dict):
        def to(self, device, dtype=None):
            events["to"] = (device, dtype)
            return self

    class FakeProcessor:
        feature_extractor = types.SimpleNamespace(sampling_rate=16000)

        def apply_transcription_request(self, language=None, audio=None, return_tensors=None):
            events["requests"].append({
                "language": language,
                "frames": len(audio),
                "return_tensors": return_tensors,
            })
            return FakeBatch({"input_ids": FakeTensor((1, 5))})

        def batch_decode(self, ids, skip_special_tokens=False):
            events["decoded_shape"] = ids.shape
            events["skip_special_tokens"] = skip_special_tokens
            return ["language English<asr_text> Concord returned to its place. "]

    class FakeModel:
        def to(self, device):
            events["model_device"] = device
            return self

        def eval(self):
            events["eval"] = True
            return self

        def generate(self, **kwargs):
            events["generate"].append(kwargs["max_new_tokens"])
            return FakeTensor((1, 14))

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = types.SimpleNamespace(
        from_pretrained=lambda src, **kw: (events["proc_loads"].append((src, kw)), FakeProcessor())[1]
    )
    fake_transformers.AutoModelForMultimodalLM = types.SimpleNamespace(
        from_pretrained=lambda src, **kw: (events["model_loads"].append((src, kw)), FakeModel())[1]
    )

    @contextmanager
    def inference_mode():
        yield

    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = "float32"
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.inference_mode = inference_mode
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    eng = fourier.OrukFourier2BLocal()
    eng.warmup(str(wav))
    out = eng.transcribe(str(wav))

    assert events["proc_loads"] == [("oruk/oruk-fourier-2b", {"revision": fourier.HF_REVISION})]
    assert events["model_loads"] == [
        ("oruk/oruk-fourier-2b", {"dtype": "float32", "revision": fourier.HF_REVISION})
    ]
    assert events["model_device"] == "cpu" and events["eval"] is True
    assert events["requests"][-1] == {"language": "en", "frames": 1600, "return_tensors": "pt"}
    assert events["decoded_shape"] == (1, 9) and events["skip_special_tokens"] is True
    assert out.text == "Concord returned to its place."
    assert out.latency_s >= 0.0
