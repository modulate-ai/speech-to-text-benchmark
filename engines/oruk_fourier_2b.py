"""Oruk Fourier 2B English speech-to-text engines (local Hugging Face + hosted Oruk API).

Model card: https://huggingface.co/oruk/oruk-fourier-2b (CC BY-NC 4.0 noncommercial open weights;
2.04B-parameter English Qwen3-ASR fine-tune, native HF Transformers `qwen3_asr` architecture).
Pinned revision: 8c2a8871f1304aa2f634da4fb95e24438c4945b9.

Two engines are provided:
- `oruk_fourier_2b_local` (local=True, $0/hr): loads `oruk/oruk-fourier-2b` directly from
  Hugging Face with `AutoProcessor` + `AutoModelForMultimodalLM` (requires `transformers >= 5.21`).
  Uses the model's published `generation_config.json` (3-beam search, repetition guard). Latency
  measures local audio decode + feature extraction + generation + token decoding (excluding
  one-time model loading in `__init__` and the untimed warm-up call).
- `oruk_fourier_2b` (local=False, $0.48/hr): calls Oruk's hosted OpenAI-compatible endpoint
  `POST https://speech-api.oruk.ai/v1/audio/transcriptions` ($0.0080/min = $0.48/hour,
  https://oruk.ai/pricing) using `ORUK_API_KEY`.
"""
import importlib.metadata
import os
import random
import re
import time
from math import gcd
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

from .base import Engine, EngineNotConfigured, FatalEngineError, Transcription

HF_REPO = "oruk/oruk-fourier-2b"
HF_REVISION = "8c2a8871f1304aa2f634da4fb95e24438c4945b9"
MIN_TRANSFORMERS_VERSION = (5, 21)

API_URL = "https://speech-api.oruk.ai/v1/audio/transcriptions"
API_MODEL = "oruk-fourier-2b"
API_PRICE_PER_HOUR = 0.48  # $0.0080/min x 60 (https://oruk.ai/pricing)
RETRY_STATUS = {408, 429, 500, 502, 503, 504}
FATAL_STATUS = {401, 402, 403}  # bad key, insufficient credits, or forbidden


def _transformers_version() -> str | None:
    try:
        return importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        return None


def _version_at_least(version_text: str | None, minimum: tuple[int, ...]) -> bool:
    if not version_text:
        return False
    parts = [int(p) for p in re.findall(r"\d+", str(version_text))[:len(minimum)]]
    while len(parts) < len(minimum):
        parts.append(0)
    return tuple(parts) >= minimum


def _extract_transcript(text: str) -> str:
    text = (text or "").strip()
    if "<asr_text>" in text:
        text = text.split("<asr_text>", 1)[-1].strip()
    return text


class OrukFourier2BLocal(Engine):
    """Oruk Fourier 2B run locally with Hugging Face Transformers (>= 5.21, CC BY-NC 4.0)."""

    name = "oruk_fourier_2b_local"
    local = True
    price_per_hour = 0.0
    model_id = HF_REPO

    def __init__(
        self,
        model: str = None,
        revision: str = HF_REVISION,
        device: str = None,
        max_new_tokens: int = 512,
    ):
        installed = _transformers_version()
        if not _version_at_least(installed, MIN_TRANSFORMERS_VERSION):
            min_ver = ".".join(str(p) for p in MIN_TRANSFORMERS_VERSION)
            found = installed or "not installed"
            raise EngineNotConfigured(
                f"oruk_fourier_2b_local requires transformers >= {min_ver} (found {found}). "
                f"Upgrade with: pip install 'transformers>={min_ver}'"
            )

        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self._torch = torch
        self._model_id = model or self.model_id
        self.model = self._model_id
        self._revision = revision if self._model_id == HF_REPO else None
        self._max_new_tokens = max_new_tokens

        chosen = (device or os.environ.get("ORUK_FOURIER_DEVICE") or "").strip().lower()
        if not chosen or chosen == "auto":
            if torch.cuda.is_available():
                chosen = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                chosen = "mps"
            else:
                chosen = "cpu"
        self._device = chosen
        self._dtype = torch.bfloat16 if chosen in ("cuda", "mps") else torch.float32

        load_kwargs = {"revision": self._revision} if self._revision else {}
        self._proc = AutoProcessor.from_pretrained(self._model_id, **load_kwargs)
        self._model = (
            AutoModelForMultimodalLM.from_pretrained(
                self._model_id, dtype=self._dtype, **load_kwargs
            )
            .to(self._device)
            .eval()
        )
        self._sr = int(getattr(self._proc.feature_extractor, "sampling_rate", 16000))

    def _load(self, path: str) -> np.ndarray:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != self._sr:
            from scipy.signal import resample_poly

            g = gcd(int(sr), int(self._sr))
            audio = resample_poly(audio, int(self._sr) // g, int(sr) // g).astype(np.float32)
        return np.ascontiguousarray(audio, dtype=np.float32)

    def _sync(self) -> None:
        torch = self._torch
        if self._device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif self._device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()

    def _run(self, audio: np.ndarray) -> str:
        torch = self._torch
        inputs = self._proc.apply_transcription_request(
            language="en",
            audio=audio,
            return_tensors="pt",
        ).to(self._device, dtype=self._dtype)
        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, max_new_tokens=self._max_new_tokens)
        self._sync()
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        decoded = self._proc.batch_decode(generated, skip_special_tokens=True)
        text = decoded[0] if isinstance(decoded, list) and decoded else str(decoded or "")
        return _extract_transcript(text)

    def warmup(self, audio_path: str) -> None:
        self._run(self._load(audio_path))

    def transcribe(self, audio_path: str) -> Transcription:
        start = time.perf_counter()
        text = self._run(self._load(audio_path))
        return Transcription(text=text, latency_s=time.perf_counter() - start)


class OrukFourier2B(Engine):
    """Oruk Fourier 2B hosted OpenAI-compatible batch transcription API."""

    name = "oruk_fourier_2b"
    price_per_hour = API_PRICE_PER_HOUR

    def __init__(self, model: str = API_MODEL, max_retries: int = 5, timeout: float = 300):
        key = os.environ.get("ORUK_API_KEY")
        if not key:
            raise EngineNotConfigured(
                "ORUK_API_KEY is not set (put it in .env; sign up at https://oruk.ai/auth/signup)"
            )
        self._key = key
        self.model = model
        self.endpoint = (os.environ.get("ORUK_SPEECH_API_URL") or API_URL).rstrip("/")
        self.price_per_hour = API_PRICE_PER_HOUR if model == API_MODEL else None
        self.max_retries = max_retries
        self.timeout = timeout

    def transcribe(self, audio_path: str) -> Transcription:
        path = Path(audio_path)
        audio_bytes = path.read_bytes()
        last = None
        for attempt in range(self.max_retries + 1):
            start = time.perf_counter()
            try:
                resp = requests.post(
                    self.endpoint,
                    headers={"Authorization": f"Bearer {self._key}"},
                    data={"model": self.model, "language": "en", "response_format": "json"},
                    files={"file": (path.name, audio_bytes)},
                    timeout=self.timeout,
                )
            except requests.RequestException as e:
                last = f"network error: {type(e).__name__}"
            else:
                latency = time.perf_counter() - start
                if resp.status_code == 200:
                    data = resp.json()
                    usage = data.get("usage") or {}
                    dur = data.get("audio_seconds")
                    if dur is None:
                        dur = data.get("duration")
                    cost = usage.get("estimated_cost_usd")
                    return Transcription(
                        text=(data.get("text") or "").strip(),
                        latency_s=latency,
                        duration_s=float(dur) if dur is not None else None,
                        raw=data,
                        cost_usd=float(cost) if cost is not None else None,
                    )
                if resp.status_code in FATAL_STATUS:
                    raise FatalEngineError(
                        f"Oruk API returned HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                last = f"HTTP {resp.status_code}: {resp.text[:200]}"
                if resp.status_code not in RETRY_STATUS:
                    break
            if attempt < self.max_retries:
                time.sleep(min(30, 2 ** attempt) + random.random())
        raise RuntimeError(f"Oruk API request failed for {audio_path}: {last}")
