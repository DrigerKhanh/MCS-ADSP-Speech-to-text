import matplotlib.pyplot as plt

import librosa.display
import speech_recognition as sr

from jiwer import wer, cer
from IPython.display import Audio

import csv
import os


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

def transcribe_directory(directory_path, output_file):
    transcriptions = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".wav"):
            file_path = os.path.join(directory_path, file_name)
            transcription = transcribe_audio(file_path)
            transcriptions.append({"file_name": file_name, "transcription": transcription})

    # Ghi kết quả vào file CSV
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["File Name", "Transcription"])  # Ghi tiêu đề cột
        for entry in transcriptions:
            writer.writerow([entry['file_name'], entry['transcription']])

    print(f"Transcriptions saved to {output_file}")

# Đường dẫn thư mục chứa file âm thanh và file kết quả đầu ra
directory_path = "./dataset/Recordings"
output_file = "./output/transcriptions.csv"

transcribe_directory(directory_path, output_file)
