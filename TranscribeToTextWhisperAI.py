import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display
import soundfile as sf
import speech_recognition as sr

from jiwer import wer, cer
from IPython.display import Audio

import whisper

import csv
import os
import tempfile
import wave

from gtts import gTTS

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

audio_signal, sample_rate = librosa.load('speech_01.wav', sr=None)

plt.figure(figsize=(12, 4))
librosa.display.waveshow(audio_signal, sr=sample_rate)
plt.title('Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# Play the audio in the notebook
Audio('speech_01.wav')

# Transciribing Audio with OpenAI's Whisper
recognizer = sr.Recognizer()
file_path = 'speech_01.wav'
ground_truth = """My name is Ivan and I am excited to have you as part of our learning community! 
Before we get started, I’d like to tell you a little bit about myself. I’m a sound engineer turned data scientist,
curious about machine learning and Artificial Intelligence. My professional background is primarily in media production,
with a focus on audio, IT, and communications"""
def transcribe_audio(file_path):
    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        print(text)
        return text
transcribed_text = transcribe_audio(file_path)

calculated_wer = wer(ground_truth, transcribed_text)
calculated_cer = cer(ground_truth, transcribed_text)
print(f"Word Error Rate (WER): {calculated_wer:.4f}")
print(f"Character Error Rate (CER): {calculated_cer:.4f}")

model = whisper.load_model("base")
result = model.transcribe(file_path)

transcribed_text_whisper = result["text"]

calculated_wer = wer(ground_truth, transcribed_text_whisper)
calculated_cer = cer(ground_truth, transcribed_text_whisper)
print(f"Word Error Rate (WER): {calculated_wer:.4f}")
print(f"Character Error Rate (CER): {calculated_cer:.4f}")

# Transcribing Multiple Audio Files from a Directory
directory_path = "C:/Users/PC/Downloads/Speech Recognition/Recordings"

def transcribe_directory_whisper(directory_path):
    transcriptions = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".wav"):
            files_path = os.path.join(directory_path, file_name)
            # Transcribe the audio file
            result = model.transcribe(files_path)
            transcription = result["text"]
            transcriptions.append({"file_name": file_name, "transcription": transcription})
    return transcriptions

transcriptions = transcribe_directory_whisper(directory_path)

# Saving Audio Transcriptions to CSV for Easy Analysis
output_file = "transcriptions.csv"

with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Track Number", "File Name", "Transcription"])  # Write the header
    for number, transcription in enumerate(transcriptions, start=1):
        writer.writerow([number, transcription['file_name'], transcription['transcription']])

text = """Thank you for taking the time to watch our course on speech recognition!
This concludes the final lesson of this section. See you soon!"""

tts = gTTS(text=text, lang='en')
tts.save("output.mp3")

os.system("start output.mp3")