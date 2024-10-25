import whisper
import sounddevice as sd
import numpy as np

# Load the Whisper model
model_name = "base"
model = whisper.load_model(model_name)


def record_audio(duration=5, fs=16000):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    print("Recording complete.")
    return np.squeeze(audio)


def transcribe_audio_whisper(audio, fs=16000):
    # Save the audio to a temporary file
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        temp_audio_file_name = temp_audio_file.name
        # Convert the NumPy array to a WAV file
        from scipy.io.wavfile import write
        write(temp_audio_file_name, fs, audio)
    # Transcribe the audio file using Whisper
    result = model.transcribe(temp_audio_file_name)
    os.remove(temp_audio_file_name)
    transcribed_text = result["text"]
    print(f"Transcribed Text: {transcribed_text}")
    return transcribed_text
