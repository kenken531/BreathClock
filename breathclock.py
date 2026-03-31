"""
BreathClock  —  Windows Edition
================================
Captures mic input, detects your breathing rhythm from the audio amplitude,
visualizes each breath as a live waveform, and computes breaths-per-minute
with a rolling average.

Usage:
    Breathe normally into your mic and watch the live waveform.
    Close the chart window to stop.
"""

# ── Force a real interactive window — must happen before any other matplotlib import
import matplotlib
matplotlib.use("TkAgg")   # opens a real OS window, works outside Spyder plots pane

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

import numpy as np
import threading
import time
import sys
from collections import deque

import pyaudio
import scipy.signal as signal

# ── Config ───────────────────────────────────────────────────────────────────

CFG = {
    # Audio
    "sample_rate":      44100,
    "chunk_size":       1024,
    "channels":         1,

    # Butterworth low-pass on the envelope (breathing = 0.1–0.5 Hz)
    "filter_cutoff":    0.5,
    "filter_order":     4,

    # How many raw samples per envelope point
    "envelope_window":  512,

    # Display
    "waveform_seconds": 30,

    # Breath detection
    "min_breath_gap":   1.5,        # seconds between counted breaths
    "peak_threshold":   0.0005,     # raise if too sensitive, lower if missing breaths
    "bpm_window":       10,         # recent breaths used for rolling BPM
}

ENVELOPE_RATE  = CFG["sample_rate"] // CFG["envelope_window"]   # ~86 Hz
DISPLAY_POINTS = CFG["waveform_seconds"] * ENVELOPE_RATE

# ── Shared state ──────────────────────────────────────────────────────────────

_lock           = threading.Lock()
_envelope_buf   = deque(maxlen=DISPLAY_POINTS)
_breath_times   = deque(maxlen=50)
_last_peak_time = 0.0
_running        = True
_current_bpm    = 0.0
_total_breaths  = 0
_session_start  = time.time()

# ── Butterworth filter ────────────────────────────────────────────────────────

def make_butter_filter():
    nyq    = ENVELOPE_RATE / 2
    cutoff = CFG["filter_cutoff"] / nyq
    b, a   = signal.butter(CFG["filter_order"], cutoff, btype="low", analog=False)
    return b, a

BUTTER_B, BUTTER_A = make_butter_filter()
_filter_zi = signal.lfilter_zi(BUTTER_B, BUTTER_A) * 0.0

# ── Audio thread ──────────────────────────────────────────────────────────────

def audio_thread():
    global _last_peak_time, _current_bpm, _total_breaths, _filter_zi

    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(
            format=pyaudio.paFloat32,
            channels=CFG["channels"],
            rate=CFG["sample_rate"],
            input=True,
            frames_per_buffer=CFG["chunk_size"],
        )
    except Exception as e:
        print(f"\n  ERROR: Could not open microphone: {e}\n")
        pa.terminate()
        return

    peak_history = deque(maxlen=int(ENVELOPE_RATE * 3))
    print("  Microphone opened. Breathe normally...\n")

    while _running:
        try:
            data  = stream.read(CFG["chunk_size"], exception_on_overflow=False)
            chunk = np.frombuffer(data, dtype=np.float32)
        except Exception:
            continue

        i = 0
        while i + CFG["envelope_window"] <= len(chunk):
            block = chunk[i : i + CFG["envelope_window"]]
            rms   = float(np.sqrt(np.mean(block ** 2)))

            filtered, _filter_zi = signal.lfilter(
                BUTTER_B, BUTTER_A, [rms], zi=_filter_zi
            )
            env_val = float(filtered[0])

            with _lock:
                _envelope_buf.append(env_val)
            peak_history.append(env_val)

            # Peak detection: local max above threshold with cooldown
            if len(peak_history) >= 3:
                prev2 = peak_history[-3]
                prev1 = peak_history[-2]
                curr  = peak_history[-1]

                is_peak      = (prev1 > prev2) and (prev1 > curr)
                above_thresh = prev1 > CFG["peak_threshold"]
                gap_ok       = (time.time() - _last_peak_time) > CFG["min_breath_gap"]

                if is_peak and above_thresh and gap_ok:
                    now = time.time()
                    _last_peak_time = now
                    with _lock:
                        _breath_times.append(now)
                        _total_breaths += 1
                        if len(_breath_times) >= 2:
                            intervals    = [_breath_times[j] - _breath_times[j-1]
                                            for j in range(1, len(_breath_times))]
                            recent       = intervals[-CFG["bpm_window"]:]
                            avg_interval = np.mean(recent)
                            _current_bpm = round(60.0 / avg_interval, 1) if avg_interval > 0 else 0.0

            i += CFG["envelope_window"]

    stream.stop_stream()
    stream.close()
    pa.terminate()

# ── Plot setup ────────────────────────────────────────────────────────────────

DARK_BG    = "#0d0d0f"
PANEL_BG   = "#13131a"
ACCENT     = "#00e5ff"
ACCENT2    = "#ff4081"
GREEN_OK   = "#69ff47"
TEXT_COLOR = "#c8c8d4"
GRID_COLOR = "#1e1e2e"

def setup_plot():
    fig = plt.figure(figsize=(13, 7), facecolor=DARK_BG)
    fig.canvas.manager.set_window_title("BreathClock — Day 06 | BUILDCORED ORCAS")

    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        left=0.07, right=0.97,
        top=0.88,  bottom=0.10,
        wspace=0.35, hspace=0.55,
    )

    # Waveform
    ax_wave = fig.add_subplot(gs[0, :])
    ax_wave.set_facecolor(PANEL_BG)
    ax_wave.set_title("Breathing Envelope  (Butterworth low-pass filtered)",
                       color=TEXT_COLOR, fontsize=10, pad=8, loc="left")
    ax_wave.set_xlim(0, CFG["waveform_seconds"])
    ax_wave.set_ylim(-0.005, 0.12)
    ax_wave.set_xlabel("seconds ago  (right = now)", color=TEXT_COLOR, fontsize=8)
    ax_wave.set_ylabel("Amplitude (RMS)", color=TEXT_COLOR, fontsize=8)
    ax_wave.tick_params(colors=TEXT_COLOR, labelsize=7)
    ax_wave.spines[:].set_color(GRID_COLOR)
    ax_wave.grid(True, color=GRID_COLOR, linewidth=0.5, linestyle="--")
    ax_wave.axhline(CFG["peak_threshold"], color=ACCENT2, linewidth=0.8,
                    linestyle=":", alpha=0.7, label=f"threshold ({CFG['peak_threshold']})")
    ax_wave.legend(loc="upper left", fontsize=7, facecolor=PANEL_BG,
                   labelcolor=TEXT_COLOR, framealpha=0.6)

    line_wave, = ax_wave.plot([], [], color=ACCENT, linewidth=1.4, alpha=0.9)

    # BPM
    ax_bpm = fig.add_subplot(gs[1, 0])
    ax_bpm.set_facecolor(PANEL_BG)
    ax_bpm.axis("off")
    bpm_text = ax_bpm.text(0.5, 0.55, "—", transform=ax_bpm.transAxes,
                            ha="center", va="center", fontsize=42,
                            color=GREEN_OK, fontweight="bold", fontfamily="monospace")
    ax_bpm.text(0.5, 0.15, "breaths / min", transform=ax_bpm.transAxes,
                ha="center", va="center", fontsize=9, color=TEXT_COLOR)
    ax_bpm.set_title("Rolling BPM", color=TEXT_COLOR, fontsize=9, pad=4)

    # Count
    ax_count = fig.add_subplot(gs[1, 1])
    ax_count.set_facecolor(PANEL_BG)
    ax_count.axis("off")
    count_text = ax_count.text(0.5, 0.55, "0", transform=ax_count.transAxes,
                                ha="center", va="center", fontsize=42,
                                color=ACCENT, fontweight="bold", fontfamily="monospace")
    ax_count.text(0.5, 0.15, "total breaths", transform=ax_count.transAxes,
                  ha="center", va="center", fontsize=9, color=TEXT_COLOR)
    ax_count.set_title("Session Count", color=TEXT_COLOR, fontsize=9, pad=4)

    # Timer
    ax_time = fig.add_subplot(gs[1, 2])
    ax_time.set_facecolor(PANEL_BG)
    ax_time.axis("off")
    time_text = ax_time.text(0.5, 0.55, "0:00", transform=ax_time.transAxes,
                              ha="center", va="center", fontsize=42,
                              color=ACCENT2, fontweight="bold", fontfamily="monospace")
    ax_time.text(0.5, 0.15, "session time", transform=ax_time.transAxes,
                 ha="center", va="center", fontsize=9, color=TEXT_COLOR)
    ax_time.set_title("Elapsed", color=TEXT_COLOR, fontsize=9, pad=4)

    # Header
    fig.text(0.5, 0.96, "BreathClock", ha="center", va="top",
             fontsize=18, fontweight="bold", color=ACCENT, fontfamily="monospace")
    fig.text(0.5, 0.925, "Day 06  —  BUILDCORED ORCAS",
             ha="center", va="top", fontsize=8, color=TEXT_COLOR, alpha=0.6)

    return dict(fig=fig, ax_wave=ax_wave, line_wave=line_wave,
                bpm_text=bpm_text, count_text=count_text, time_text=time_text)

# ── Animation update ──────────────────────────────────────────────────────────

_fill_ref = [None]

def make_update(handles):
    ax_wave   = handles["ax_wave"]
    line_wave = handles["line_wave"]

    def update(frame):
        with _lock:
            env    = list(_envelope_buf)
            bpm    = _current_bpm
            count  = _total_breaths
            btimes = list(_breath_times)

        if not env:
            return

        xs = np.linspace(CFG["waveform_seconds"], 0, len(env))
        ys = np.array(env)
        line_wave.set_data(xs, ys)

        # Replace fill
        if _fill_ref[0] is not None:
            try:
                _fill_ref[0].remove()
            except Exception:
                pass
        _fill_ref[0] = ax_wave.fill_between(xs, ys, color=ACCENT, alpha=0.07)

        # Remove old breath marker lines (keep only threshold hline + new fill)
        for coll in ax_wave.lines[2:]:
            try:
                coll.remove()
            except Exception:
                pass

        # Draw breath markers
        now = time.time()
        for bt in btimes:
            age = now - bt
            if 0 <= age <= CFG["waveform_seconds"]:
                ax_wave.axvline(age, color=ACCENT2, linewidth=0.8, alpha=0.5)

        # BPM
        if bpm > 0:
            handles["bpm_text"].set_text(f"{bpm:.1f}")
            handles["bpm_text"].set_color(
                GREEN_OK if 10 <= bpm <= 20 else (ACCENT if bpm < 10 else ACCENT2)
            )
        else:
            handles["bpm_text"].set_text("—")

        handles["count_text"].set_text(str(count))

        elapsed = int(time.time() - _session_start)
        m, s = divmod(elapsed, 60)
        handles["time_text"].set_text(f"{m}:{s:02d}")

    return update

# ── Console UI ────────────────────────────────────────────────────────────────

CYAN  = "\033[96m"
GREEN = "\033[92m"
YELLOW= "\033[93m"
RED   = "\033[91m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
RESET = "\033[0m"

def clear():
    print("\033[2J\033[H", end="")

def banner():
    print(f"{BOLD}{CYAN}")
    print("  ██████╗ ██████╗ ███████╗ █████╗ ████████╗██╗  ██╗")
    print("  ██╔══██╗██╔══██╗██╔════╝██╔══██╗╚══██╔══╝██║  ██║")
    print("  ██████╔╝██████╔╝█████╗  ███████║   ██║   ███████║")
    print("  ██╔══██╗██╔══██╗██╔══╝  ██╔══██║   ██║   ██╔══██║")
    print("  ██████╔╝██║  ██║███████╗██║  ██║   ██║   ██║  ██║")
    print("  ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝")
    print(f"  {DIM}           C L O C K{RESET}{BOLD}{CYAN}         Day 06 — BUILDCORED ORCAS{RESET}")
    print()

def instructions_panel():
    print(f"  {BOLD}── HOW TO USE ──────────────────────────────────────────{RESET}")
    print(f"  {GREEN}🎙  Breathe normally{RESET}     →  Mic captures your rhythm")
    print(f"  {CYAN}📈  Live waveform{RESET}        →  Butterworth-filtered envelope")
    print(f"  {YELLOW}💨  Breath peaks{RESET}         →  Pink vertical lines on chart")
    print(f"  {BOLD}  Close the chart window  →  Quit{RESET}")
    print(f"  {BOLD}────────────────────────────────────────────────────────{RESET}")
    print()
    print(f"  {DIM}Tip: if breaths aren't detected, breathe a little louder")
    print(f"  or lower 'peak_threshold' in CFG at the top of the file.{RESET}")
    print()
    print(f"  {RED}{BOLD}⚠  Run from terminal, not Spyder's Run button:{RESET}")
    print(f"  {YELLOW}     python breathclock.py{RESET}")
    print()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global _running

    clear()
    banner()
    instructions_panel()

    input(f"  {YELLOW}  Press ENTER to start BreathClock...{RESET} ")
    print()

    # Start audio thread
    t = threading.Thread(target=audio_thread, daemon=True)
    t.start()
    time.sleep(0.3)

    # Setup plot
    handles = setup_plot()
    update_fn = make_update(handles)

    # Keep animation reference alive — critical, or it gets garbage collected
    anim = animation.FuncAnimation(
        handles["fig"],
        update_fn,
        interval=120,
        blit=False,
        cache_frame_data=False,
    )

    print(f"  {GREEN}Chart open — breathe into your mic!{RESET}")
    print(f"  {DIM}Close the chart window to stop.{RESET}\n")

    try:
        plt.show(block=True)   # block=True keeps the window alive until closed
    finally:
        _running = False
        print(f"\n  {BOLD}{RED}■ BreathClock stopped.{RESET}")
        if _total_breaths > 0:
            elapsed = time.time() - _session_start
            print(f"  Total breaths : {_total_breaths}")
            print(f"  Session time  : {int(elapsed // 60)}m {int(elapsed % 60)}s")
            print(f"  Last BPM      : {_current_bpm:.1f}")
        print()


if __name__ == "__main__":
    main()