# BreathClock

BreathClock is a real-time breathing tracker that uses your microphone to detect airflow patterns, visualize your breathing waveform, and calculate breaths per minute (BPM) with a rolling average. Built for BUILDCORED ORCAS — Day 06.

## How it works

- Uses PyAudio to capture live microphone input in real time.
- Splits incoming audio into small chunks and computes the RMS (root mean square) amplitude to estimate airflow intensity.
- Applies a Butterworth low-pass filter to isolate breathing frequencies (approximately 0.1–0.5 Hz).
- Constructs a smoothed breathing envelope from the filtered signal.
- Detects breaths using peak detection:
  - Identifies local maxima in the waveform
  - Applies a threshold to ignore noise
  - Enforces a minimum time gap between breaths
- Stores recent breath timestamps and computes a rolling BPM using the average interval between breaths.

## Features

- Real-time breathing waveform visualization
- Automatic breath detection using signal processing
- Rolling breaths-per-minute (BPM) calculation
- Total breath counter for the session
- Live session timer
- Clean, multi-panel UI:
  - Waveform (top)
  - BPM display
  - Breath count
  - Elapsed time
- Visual markers for detected breaths
- Threshold line for tuning sensitivity

## Requirements

- Python 3.10+
- A working microphone

Install dependencies:
```
pip install numpy matplotlib pyaudio scipy
```

## Setup

1. Clone the repository or download the script.
2. Ensure your microphone is connected and enabled.
3. Install the required Python packages (see Requirements section or run:
```
pip install -r requirements.txt
```
after downloading requirements.txt)

## Usage

From the project folder:
```
python breathclock.py
```

- A window opens showing the live breathing waveform.
- Breathe normally into your microphone.
- Close the window to stop the program.

## Interface

The visualization includes:

- **Waveform panel**
  - Shows the filtered breathing envelope over the last 30 seconds
  - Right side represents the current moment
  - Threshold line indicates detection sensitivity
  - Vertical markers indicate detected breaths

- **BPM panel**
  - Displays rolling breaths per minute
  - Color-coded:
    - Green: normal range (10–20 BPM)
    - Cyan: slow breathing
    - Pink: fast breathing

- **Session count**
  - Total number of breaths detected

- **Elapsed time**
  - Duration since session start

## Tuning

You can adjust detection sensitivity and behavior via the `CFG` dictionary:

| Parameter | Default | Description |
|----------|--------|-------------|
| `sample_rate` | 44100 | Audio sampling rate |
| `chunk_size` | 1024 | Audio buffer size |
| `envelope_window` | 512 | Samples per envelope point |
| `filter_cutoff` | 0.5 | Low-pass cutoff frequency (Hz) |
| `filter_order` | 4 | Butterworth filter order |
| `waveform_seconds` | 30 | Length of displayed waveform |
| `min_breath_gap` | 1.5 | Minimum time between breaths |
| `peak_threshold` | 0.0005 | Detection sensitivity |
| `bpm_window` | 10 | Number of recent breaths for BPM |

### Tuning tips

- If breaths are not detected → lower `peak_threshold`
- If noise triggers false breaths → increase `peak_threshold`
- If BPM feels unstable → increase `bpm_window`
- If detection is too fast → increase `min_breath_gap`

## Technical Details

- RMS amplitude is used as a proxy for airflow intensity.
- A Butterworth filter removes high-frequency noise, isolating slow breathing patterns.
- Peak detection uses:
  - Local maxima comparison (prev > neighbors)
  - Threshold gating
  - Time-based cooldown
- BPM is computed as:

  breaths per minute = 60 / average interval between recent breaths

- Threading is used to separate:
  - Audio processing (background thread)
  - Visualization (main thread)

## Limitations

- Sensitive to background noise and microphone quality
- Requires consistent breathing direction toward the mic
- Not a medical-grade measurement tool
- Latency depends on buffer size and filtering

## Concept

BreathClock treats breathing as a low-frequency biological signal:

- Airflow → audio amplitude
- Amplitude → filtered envelope
- Envelope peaks → breath events
- Breath intervals → BPM

It effectively converts your breathing into a measurable time-series signal.

## Credits

- PyAudio (audio capture)
- NumPy (signal processing)
- SciPy (Butterworth filtering)
- Matplotlib (real-time visualization)

Built as part of the BUILDCORED ORCAS — Day 06: BreathClock challenge.
