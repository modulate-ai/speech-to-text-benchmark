from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from engines import audio_utils


def test_to_wav16k_converts_and_reuses_cached_output(tmp_path, monkeypatch):
    sample_rate = 44_100
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    tone = 0.5 * np.sin(2 * np.pi * 440 * time)
    stereo = np.column_stack((tone, tone))
    source = tmp_path / "tone.wav"
    sf.write(source, stereo, sample_rate)
    monkeypatch.setattr(audio_utils, "AUDIO_ROOT", tmp_path)

    output = Path(audio_utils.to_wav16k(str(source)))

    info = sf.info(output)
    assert info.samplerate == 16_000
    assert info.channels == 1
    assert info.subtype == "PCM_16"
    assert info.duration == pytest.approx(1.0, abs=0.01)

    modified_at = output.stat().st_mtime_ns

    def fail_if_rewritten(*args, **kwargs):
        raise AssertionError("cached output should not be rewritten")

    monkeypatch.setattr(audio_utils.sf, "write", fail_if_rewritten)
    assert Path(audio_utils.to_wav16k(str(source))) == output
    assert output.stat().st_mtime_ns == modified_at
