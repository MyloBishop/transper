import os
import sys
import time
import wave
import tempfile
import threading

import torch
import pyaudiowpatch as pyaudio
from faster_whisper import WhisperModel as whisper

# A bigger audio buffer gives better accuracy
# but also increases latency in response.
AUDIO_BUFFER = 5


def record_audio(p, device):
    """Record audio from output device and save to temporary WAV file."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        filename = f.name
        wave_file = wave.open(filename, "wb")
        wave_file.setnchannels(device["maxInputChannels"])
        wave_file.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
        wave_file.setframerate(int(device["defaultSampleRate"]))

        def callback(in_data, frame_count, time_info, status):
            """Write frames and return PA flag"""
            wave_file.writeframes(in_data)
            return (in_data, pyaudio.paContinue)

        stream = p.open(
            format=pyaudio.paInt16,
            channels=device["maxInputChannels"],
            rate=int(device["defaultSampleRate"]),
            frames_per_buffer=pyaudio.get_sample_size(pyaudio.paInt16),
            input=True,
            input_device_index=device["index"],
            stream_callback=callback,
        )

        try:
            time.sleep(AUDIO_BUFFER)  # Blocking execution while playing
        finally:
            stream.stop_stream()
            stream.close()
            wave_file.close()
            # print(f"{filename} saved.")
    return filename


def whisper_audio(filename, model):
    """Transcribe audio buffer and display."""
    segments, info = model.transcribe(filename, beam_size=5, task="translate")
    os.remove(filename)
    # print(f"{filename} removed.")
    for segment in segments:
        print(f"[{segment.start:.2f} -> {segment.end:.2f}] {segment.text.strip()}")


def main():
    """Load model record audio and transcribe from default output device."""
    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper("large-v2", device=device, compute_type="float16")
    print("Model loaded.")

    with pyaudio.PyAudio() as pya:
        # Create PyAudio instance via context manager.
        try:
            # Get default WASAPI info
            wasapi_info = pya.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError:
            print("Looks like WASAPI is not available on the system. Exiting...")
            sys.exit()

        # Get default WASAPI speakers
        default_speakers = pya.get_device_info_by_index(
            wasapi_info["defaultOutputDevice"]
        )

        if not default_speakers["isLoopbackDevice"]:
            for loopback in pya.get_loopback_device_info_generator():
                # Try to find loopback device with same name(and [Loopback suffix]).
                # Unfortunately, this is the most adequate way at the moment.
                if default_speakers["name"] in loopback["name"]:
                    default_speakers = loopback
                    break
            else:
                print(
                    """
                    Default loopback output device not found.
                    Run `python -m pyaudiowpatch` to check available devices.
                    Exiting...
                    """
                )
                sys.exit()

        print(
            f"Recording from: {default_speakers['name']} ({default_speakers['index']})\n"
        )

        while True:
            filename = record_audio(pya, default_speakers)
            thread = threading.Thread(target=whisper_audio, args=(filename, model))
            thread.start()


if __name__ == "__main__":
    main()
