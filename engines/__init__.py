from .assemblyai import AssemblyAI
from .deepgram import Deepgram
from .moonshine_local import MoonshineBase, MoonshineTiny
from .openrouter_stt import Chirp3, MaiTranscribe2
from .oruk_fourier_2b import OrukFourier2B, OrukFourier2BLocal
from .velma import Velma
from .whisper_cpp_local import WhisperCppBase, WhisperCppTiny
from .whisper_hf import WhisperHF

ENGINES = {
    "velma": Velma,
    "deepgram": Deepgram,
    "assemblyai": AssemblyAI,
    "whisper_hf": WhisperHF,
    "moonshine_tiny": MoonshineTiny,
    "moonshine_base": MoonshineBase,
    "chirp_3": Chirp3,
    "mai_transcribe_2": MaiTranscribe2,
    "whisper_cpp_tiny": WhisperCppTiny,
    "whisper_cpp_base": WhisperCppBase,
    "oruk_fourier_2b": OrukFourier2B,
    "oruk_fourier_2b_local": OrukFourier2BLocal,
}
