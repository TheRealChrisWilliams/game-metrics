import speech_recognition as sr
import torch

# Initialize the recognizer
recognizer = sr.Recognizer()


def capture_voice_input():
    # Use the microphone as the source
    with sr.Microphone() as source:
        print("Please speak into the microphone...")
        # Adjust for ambient noise
        recognizer.adjust_for_ambient_noise(source)
        # Capture the audio
        audio = recognizer.listen(source)
    return audio


def transcribe_audio(audio):
    try:
        # Use Google's free Web Speech API for transcription
        text = recognizer.recognize_google(audio)
        print(f"Transcribed Text: {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, could not understand the audio.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return None
