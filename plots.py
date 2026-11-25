"""Charts and banner for the benchmark results, drawn in a light navy, purple and coral theme.

Every chart is drawn on one pixel-coordinate canvas (1 data unit = 1 px at 100 dpi) so the same
panel functions build every chart.
Each engine has its own color; Velma uses the theme's main coral red.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch  # noqa: E402
from matplotlib.textpath import TextPath  # noqa: E402
from matplotlib.transforms import Affine2D  # noqa: E402

FONTS = ["Helvetica Neue", "Avenir Next", "Arial", "DejaVu Sans"]
plt.rcParams["font.family"] = FONTS

# Theme: white page, light cards, deep navy text, purple to coral gradient accents
BG, BG2 = "#FFFFFF", "#F2F1F8"
CARD, EDGE, TRACK = "#FFFFFF", "#E3E2EE", "#EEEDF5"
NAVY, MUTED, FAINT = "#12163F", "#686C84", "#B9BACB"
PURPLE, CORAL = "#5B1E78", "#D6455D"
WAVE = "#CFCBE3"

PALETTE = {
    "velma": CORAL, "deepgram": "#1E96E8", "assemblyai": "#2F3D9A", "chirp_3": "#6FAE1F",
    "mai_transcribe_2": "#12A594", "moonshine_tiny": "#F0A020", "moonshine_base": "#F26B1D",
    "whisper_cpp_tiny": "#64748B", "whisper_cpp_base": "#8E4EC6", "whisper_hf": "#2E9E5B",
    "oruk_fourier_2b": "#0F766E", "oruk_fourier_2b_local": "#14B8A6",
}
NAMES = {
    "velma": "Velma Fast", "deepgram": "Deepgram nova-3", "assemblyai": "AssemblyAI 3.5 Pro",
    "chirp_3": "Google Chirp 3", "mai_transcribe_2": "MAI-Transcribe 2",
    "moonshine_tiny": "Moonshine Tiny", "moonshine_base": "Moonshine Base",
    "whisper_cpp_tiny": "whisper.cpp tiny.en", "whisper_cpp_base": "whisper.cpp base.en",
    "whisper_hf": "Whisper large-v3 (HF)",
    "oruk_fourier_2b": "Oruk Fourier 2B (API)", "oruk_fourier_2b_local": "Oruk Fourier 2B",
}
# Model files on disk for the local engines: Moonshine and Oruk Fourier 2B from Hugging Face, whisper.cpp from its README
MODEL_SIZE = {"moonshine_tiny": "105 MiB", "moonshine_base": "237 MiB",
              "whisper_cpp_tiny": "75 MiB", "whisper_cpp_base": "142 MiB",
              "oruk_fourier_2b_local": "3.81 GiB"}
DATASET_NAMES = {"librispeech": "LibriSpeech", "voxpopuli": "VoxPopuli", "commonvoice": "Common Voice"}


def _fs(px):  # pixel size to points at 100 dpi
    return px * 0.72


def _mix(color, other, t):
    a, b = np.array(to_rgb(color)), np.array(to_rgb(other))
    return tuple(a * (1 - t) + b * t)


def _t(ax, x, y, s, size, color=NAVY, weight="normal", ha="left", va="center", **kw):
    return ax.text(x, y, s, fontsize=_fs(size), color=color, fontweight=weight, ha=ha, va=va, **kw)


def _gtext(ax, x, y, s, size, ha="left", weight="bold", z=8, c0=PURPLE, c1=CORAL):
    """Text filled with the purple to coral gradient. y is the baseline (pixel coordinates, y down)."""
    tp = TextPath((0, 0), s, size=size, prop=FontProperties(family=FONTS, weight=weight))
    bb = tp.get_extents()
    xoff = x - bb.x0 - (bb.width / 2 if ha == "center" else bb.width if ha == "right" else 0)
    patch = PathPatch(tp, transform=Affine2D().scale(1, -1).translate(xoff, y) + ax.transData,
                      fc="none", ec="none", zorder=z)
    ax.add_patch(patch)
    im = ax.imshow(np.linspace(0, 1, 256)[None, :], extent=(xoff + bb.x0, xoff + bb.x1, y - bb.y0, y - bb.y1),
                   cmap=LinearSegmentedColormap.from_list("t", [c0, c1]), aspect="auto", zorder=z)
    im.set_clip_path(patch)
    return bb.width


def _waves(ax, w, h, y0, y1, alpha=0.5, n=34, amp=0.5, seed=0):
    """Faint flowing waveform lines, like the ribbon behind the Modulate hero."""
    xs = np.linspace(0, w, 500)
    env = 0.30 + 0.70 * np.exp(-(((xs - 0.55 * w) / (0.40 * w)) ** 2))
    mid, span = (y0 + y1) / 2, (y1 - y0) / 2 * amp
    cm = LinearSegmentedColormap.from_list("w", [PURPLE, CORAL])
    for i in range(n):
        f = i / (n - 1)
        ph = f * 2.4 + seed
        y = mid + span * env * np.sin(2 * np.pi * xs / (0.62 * w) + ph) * (0.35 + 0.65 * (1 - abs(f - 0.5) * 1.4))
        y = y + (f - 0.5) * span * 0.55 * np.cos(2 * np.pi * xs / (0.9 * w) - ph)
        ax.plot(xs, y, color=_mix(WAVE, cm(f)[:3], 0.35), alpha=alpha * (0.45 + 0.55 * np.sin(np.pi * f)),
                lw=0.9, zorder=1)


def _canvas(w, h, waves=True):
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    ax.imshow(np.linspace(0, 1, 256)[:, None], extent=(0, w, h, 0),
              cmap=LinearSegmentedColormap.from_list("bg", [BG, BG2]), aspect="auto", zorder=0)
    if waves:
        _waves(ax, w, h, 0, h, alpha=0.45, amp=0.55)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    return fig, ax


def _card(ax, x, y, w, h):
    for k, a in [(10, 0.025), (6, 0.035), (3, 0.05)]:  # soft shadow
        ax.add_patch(FancyBboxPatch((x - k + 1, y - k + 7), w + 2 * k - 2, h + 2 * k - 2,
                                    boxstyle=f"round,pad=0,rounding_size={22 + k}", fc=NAVY, ec="none",
                                    alpha=a, zorder=1.5))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=22", fc=CARD, ec=EDGE,
                                lw=1.3, zorder=2))


def _grad_bar(ax, x, y, w, h, color, z=5):
    """Rounded bar with a light-to-full horizontal gradient and a soft shadow."""
    r = min(h / 2, w / 2)
    ax.add_patch(FancyBboxPatch((x + 1, y + 4), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=color,
                                ec="none", alpha=0.18, zorder=z - 1))
    clip = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc="none", ec="none", zorder=z)
    ax.add_patch(clip)
    cmap = LinearSegmentedColormap.from_list("g", [_mix(color, "#FFFFFF", 0.55), color])
    im = ax.imshow(np.linspace(0, 1, 256)[None, :], extent=(x, x + w, y + h, y), cmap=cmap, aspect="auto",
                   zorder=z, interpolation="bicubic")
    im.set_clip_path(clip)
    ax.plot([x + r, x + w - r], [y + 3, y + 3], color="#FFFFFF", alpha=0.45, lw=1.2, zorder=z + 1,
            solid_capstyle="round")


def _tag_dot(ax, x, y, local, r=4.2, z=6):
    """Filled dot for hosted engines, ring for local engines (same convention as the scatter plot)."""
    if local:
        ax.add_patch(Circle((x, y), r - 0.6, fc="#FFFFFF", ec=NAVY, lw=1.5, zorder=z))
    else:
        ax.add_patch(Circle((x, y), r, fc=NAVY, ec="none", zorder=z))


def panel_bars(ax, rect, title, subtitle, items, fmt, local, log=False, compact=False):
    """Ranked horizontal bars. items: list of (engine, value); best (lowest) first."""
    x, y, w, h = rect
    _card(ax, x, y, w, h)
    _gtext(ax, x + 30, y + 52, title, 29 if not compact else 24)
    _t(ax, x + 30, y + 80, subtitle, 15 if not compact else 13, color=MUTED)
    items = sorted(items, key=lambda it: it[1])
    n = len(items)
    top, bottom = y + (112 if not compact else 104), y + h - 22
    rh = (bottom - top) / n
    name_w = 262 if not compact else 196
    bx0, bx1 = x + 40 + name_w, x + w - 30 - (92 if not compact else 78)
    vals = [v for _, v in items]
    vmax = max(vals) or 1
    pos = [v for v in vals if v > 0]
    lo = (min(pos) * 0.45) if (log and pos) else 0
    for i, (e, v) in enumerate(items):
        cy = top + rh * (i + 0.5)
        col = PALETTE.get(e, "#8A8DA3")
        ax.add_patch(Circle((x + 48, cy), 14, fc="#FFFFFF", ec=col, lw=1.8, zorder=4))
        _t(ax, x + 48, cy + 0.5, str(i + 1), 14, color=col, weight="bold", ha="center", zorder=5)
        _t(ax, x + 74, cy - 9, NAMES.get(e, e), 17.5 if not compact else 15, weight="bold", zorder=5)
        is_local = e in local
        _tag_dot(ax, x + 79, cy + 13, is_local)
        tag = ("LOCAL  ·  " + MODEL_SIZE[e]) if (is_local and e in MODEL_SIZE) else ("LOCAL" if is_local else "HOSTED API")
        _t(ax, x + 90, cy + 13, tag, 12 if not compact else 11, color=MUTED, weight="600", zorder=5)
        ax.add_patch(FancyBboxPatch((bx0, cy - 10), bx1 - bx0, 20, boxstyle="round,pad=0,rounding_size=10",
                                    fc=TRACK, ec="none", zorder=3))
        if log and v > 0:
            frac = (np.log10(v) - np.log10(lo)) / (np.log10(vmax * 1.05) - np.log10(lo))
        else:
            frac = v / vmax
        length = max(frac * (bx1 - bx0), 14)
        _grad_bar(ax, bx0, cy - 10, length, 20, col)
        _t(ax, bx0 + length + 12, cy, fmt(v), 18 if not compact else 15.5, weight="bold", zorder=6)


def panel_scatter(ax, rect, items, local):
    """WER against latency. items: list of (engine, wer, latency)."""
    x, y, w, h = rect
    _card(ax, x, y, w, h)
    _gtext(ax, x + 30, y + 52, "Accuracy against speed", 29)
    _t(ax, x + 30, y + 80, "Top right is more accurate and faster · rings run locally, no network", 15, color=MUTED)
    px0, px1, py0, py1 = x + 88, x + w - 40, y + 120, y + h - 66
    wers = [i[1] for i in items]
    lats = [i[2] for i in items]
    wmin, wmax = 0, max(wers) * 1.12
    lmin, lmax = min(lats) * 0.6, max(lats) * 1.9

    def X(wv):  # reversed: lower WER on the right
        return px0 + (wmax - wv) / (wmax - wmin) * (px1 - px0)

    def Y(lv):  # log, faster at the top
        return py0 + (np.log10(lv) - np.log10(lmin)) / (np.log10(lmax) - np.log10(lmin)) * (py1 - py0)

    yy, xx = np.mgrid[int(py0):int(py1):4, int(px0):int(px1):4]
    a = np.clip(1 - np.hypot(xx - px1, yy - py0) / ((px1 - px0) * 0.62), 0, 1) ** 2 * 0.40
    rgba = np.zeros(a.shape + (4,))
    rgba[..., :3] = to_rgb("#8EDCC0")
    rgba[..., 3] = a
    ax.imshow(rgba, extent=(px0, px1, py1, py0), aspect="auto", zorder=3)
    for gv in [0.1, 1, 10]:
        if lmin <= gv <= lmax:
            ax.plot([px0, px1], [Y(gv)] * 2, color=EDGE, lw=1, zorder=3)
            _t(ax, px0 - 12, Y(gv), f"{gv:g}s", 13.5, color=MUTED, ha="right")
    step = 2 if wmax > 8 else 1
    for wv in range(0, int(wmax) + 1, step):
        ax.plot([X(wv)] * 2, [py0, py1], color=EDGE, lw=1, zorder=3)
        _t(ax, X(wv), py1 + 20, f"{wv}%", 13.5, color=MUTED, ha="center")
    _t(ax, (px0 + px1) / 2, y + h - 22, "Word error rate (lower is better, axis reversed)", 14, color=MUTED, ha="center")
    ax.text(x + 30, (py0 + py1) / 2, "Mean latency per file (log scale)", fontsize=_fs(14), color=MUTED,
            rotation=90, ha="center", va="center")
    off = {"velma": (0, -24, "center"), "deepgram": (-18, -22, "right"), "assemblyai": (20, 4, "left"),
           "chirp_3": (-18, 24, "right"), "mai_transcribe_2": (0, 26, "center"),
           "moonshine_tiny": (0, -24, "center"), "moonshine_base": (0, 26, "center"),
           "whisper_cpp_tiny": (22, 4, "left"), "whisper_cpp_base": (0, 26, "center")}
    for e, wv, lv in items:
        col = PALETTE.get(e, "#8A8DA3")
        cx, cy = X(wv), Y(lv)
        ax.add_patch(Circle((cx + 1, cy + 4), 13, fc=col, ec="none", alpha=0.20, zorder=4))
        if e in local:
            ax.add_patch(Circle((cx, cy), 10.5, fc="#FFFFFF", ec=col, lw=3.2, zorder=6))
        else:
            ax.add_patch(Circle((cx, cy), 10.5, fc=col, ec="#FFFFFF", lw=1.8, zorder=6))
        dx, dy, ha = off.get(e, (0, -24, "center"))
        _t(ax, cx + dx, cy + dy, NAMES.get(e, e), 14.5, weight="bold", ha=ha, zorder=7)


def _overall(summaries):
    return {r["engine"]: r for r in summaries if r["dataset"] == "overall"}


def make_banner(path):
    """Repository banner, 1920 x 1080 px: gradient title over flowing waveform lines."""
    w, h = 1920, 1080
    fig, ax = _canvas(w, h, waves=False)
    _waves(ax, w, h, 70, h - 70, alpha=0.8, n=52, amp=0.9, seed=0.6)
    _gtext(ax, w / 2, 520, "Speech-to-Text Benchmark", 112, ha="center")
    _t(ax, w / 2, 598, "Nine engines, three datasets, one shared scoring pipeline", 36, color=MUTED, ha="center")
    chips = ["WORD ERROR RATE", "LATENCY", "COST PER HOUR"]
    widths = [64 + len(c) * 16 for c in chips]
    x = w / 2 - (sum(widths) + 22 * (len(chips) - 1)) / 2
    for i, (c, cw) in enumerate(zip(chips, widths)):
        filled = i != 1
        ax.add_patch(FancyBboxPatch((x, 660), cw, 60, boxstyle="round,pad=0,rounding_size=30",
                                    fc=NAVY if filled else "#FFFFFF", ec=NAVY if filled else EDGE, lw=1.6, zorder=4))
        _t(ax, x + cw / 2, 690, c, 20, color="#FFFFFF" if filled else NAVY, weight="bold", ha="center", zorder=5)
        x += cw + 22
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, facecolor=BG)
    plt.close(fig)


def make_all(summaries, out_dir, engines):
    local = {e for e, c in engines.items() if c.local}
    ov = _overall(summaries)
    pw, ph = 1260, 720
    pct = lambda v: f"{v:g}%"
    secs = lambda v: f"{v:.2f}s"
    usd = lambda v: "$0" if v == 0 else f"${v:.3f}"
    wer = [(e, r["wer_percent"]) for e, r in ov.items()]
    lat = [(e, r["mean_latency_s"]) for e, r in ov.items()]
    cost = [(e, r["price_per_hour_usd"]) for e, r in ov.items() if r.get("price_per_hour_usd") is not None]
    sc = [(e, r["wer_percent"], r["mean_latency_s"]) for e, r in ov.items()]
    note_wer = "Lower means fewer word errors · all 60 files"
    note_lat = "Mean seconds per file · log scale · local engines have no network, so are not directly comparable"
    note_cost = "USD per hour of audio · local engines: no API charge (compute not counted)"

    def single(name, draw):
        fig, ax = _canvas(pw + 80, ph + 80)
        draw(ax, (40, 40, pw, ph))
        fig.savefig(out_dir / name, dpi=100, facecolor=BG)
        plt.close(fig)

    single("wer.png", lambda ax, r: panel_bars(ax, r, "Word error rate", note_wer, wer, pct, local))
    single("latency.png", lambda ax, r: panel_bars(ax, r, "Latency per file", note_lat, lat, secs, local, log=True))
    single("cost_per_hour.png", lambda ax, r: panel_bars(ax, r, "Cost per audio hour", note_cost, cost, usd, local))
    single("wer_vs_latency.png", lambda ax, r: panel_scatter(ax, r, sc, local))

    for key, fname, title, fmt, log in [("wer_percent", "wer_by_dataset.png", "Word error rate", pct, False),
                                         ("mean_latency_s", "latency_by_dataset.png", "Latency per file", secs, True)]:
        cw, chh, gap = 880, 720, 30
        fig, ax = _canvas(3 * cw + 4 * gap, chh + 2 * gap)
        for i, ds in enumerate(["librispeech", "voxpopuli", "commonvoice"]):
            items = [(r["engine"], r[key]) for r in summaries if r["dataset"] == ds]
            panel_bars(ax, (gap + i * (cw + gap), gap, cw, chh), f"{title} · {DATASET_NAMES[ds]}",
                       "20 files" + (" · log scale" if log else ""), items, fmt, local, log=log, compact=True)
        fig.savefig(out_dir / fname, dpi=100, facecolor=BG)
        plt.close(fig)
