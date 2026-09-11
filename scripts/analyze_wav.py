import wave
import struct
import math
import hashlib

path = r"C:\Users\GAME\Desktop\tts_wavs\wave1a_mixed_en1.wav"

with open(path, "rb") as f:
    data = f.read()
    sha256 = hashlib.sha256(data).hexdigest()

with wave.open(path, "rb") as wf:
    nchannels = wf.getnchannels()
    sampwidth = wf.getsampwidth()
    framerate = wf.getframerate()
    nframes = wf.getnframes()
    duration = nframes / float(framerate)
    raw_frames = wf.readframes(nframes)

fmt = f"<{nframes * nchannels}h" if sampwidth == 2 else f"<{nframes * nchannels}i"
samples = struct.unpack(fmt, raw_frames)

max_amp = max(abs(s) for s in samples)
sum_sq = sum(float(s)**2 for s in samples)
rms = math.sqrt(sum_sq / len(samples))
dc_offset = sum(samples) / len(samples)

full_scale = 32767.0 if sampwidth == 2 else 2147483647.0
max_db = 20 * math.log10(max_amp / full_scale) if max_amp > 0 else -999
rms_db = 20 * math.log10(rms / full_scale) if rms > 0 else -999

print("=== WAV FILE ANALYSIS ===")
print(f"File: {path}")
print(f"File Size: {len(data)} bytes")
print(f"SHA-256: {sha256}")
print(f"Channels: {nchannels}")
print(f"Sample Width: {sampwidth * 8} bits")
print(f"Sample Rate: {framerate} Hz")
print(f"Frames: {nframes}")
print(f"Duration: {duration:.4f} seconds")
print(f"Max Amplitude: {max_amp} ({max_db:.2f} dBFS)")
print(f"RMS Amplitude: {rms:.2f} ({rms_db:.2f} dBFS)")
print(f"DC Offset: {dc_offset:.4f}")
print(f"Non-silent (RMS > 0.001): {rms > 0.001}")

# Check if faster-whisper or whisper is available
try:
    from faster_whisper import WhisperModel
    print("\nAttempting transcription with faster-whisper...")
    model = WhisperModel("base.en", device="cpu", compute_type="int8")
    segments, info = model.transcribe(path, beam_size=5)
    text = " ".join([s.text for s in segments])
    print(f"Detected language: {info.language} ({info.language_probability:.2f})")
    print(f"Transcription: '{text.strip()}'")
except Exception as e:
    print(f"\nfaster-whisper transcription not run: {e}")
    try:
        import whisper
        print("Attempting transcription with openai-whisper...")
        model = whisper.load_model("base.en")
        res = model.transcribe(path)
        print(f"Transcription: '{res['text'].strip()}'")
    except Exception as e2:
        print(f"whisper transcription not run: {e2}")
