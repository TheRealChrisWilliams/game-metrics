import os
from openai import OpenAI
import sounddevice as sd
import numpy as np
from dotenv import load_dotenv
from scipy.io.wavfile import write

load_dotenv()

client = OpenAI()


def record_audio(duration=7, fs=16000):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    print("Recording complete.")
    return np.squeeze(audio)


def transcribe_audio_openai(audio, fs=16000):
    # Define a normal file path for temporary storage
    temp_audio_file_path = "temp_audio.wav"

    # Save the audio data to this file
    write(temp_audio_file_path, fs, audio)

    # Open the saved audio file and send it to OpenAI for transcription
    with open(temp_audio_file_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(model="whisper-1", file=audio_file)

    # Delete the temporary file after transcription
    os.remove(temp_audio_file_path)
    print(response)
    # Get transcription text
    transcribed_text = response.text
    print(f"Transcribed Text: {transcribed_text}")
    return transcribed_text
