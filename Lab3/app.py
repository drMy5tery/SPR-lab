import pyaudio
import wave
import numpy as np
import streamlit as st
import os
from google.cloud import speech
import whisper
from vosk import Model, KaldiRecognizer
import wave, json
import soundfile as sf
import numpy as np
import speech_recognition as sr
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import librosa


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "ardent-sun-466808-r2-c5f890327883.json"



p = pyaudio.PyAudio()
device_list = []
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        device_list.append((i, info['name']))

device_index = st.selectbox("Select your microphone device:", [f"{i} - {name}" for i, name in device_list])
device_index = int(device_index.split(" - ")[0])  # extract index

# -----------------------------
# 2. Mic recording function
# -----------------------------
def record_from_mic(output_file="mic_record.wav", record_seconds=5, device_index=device_index):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=CHUNK)

    st.info(f"Recording for {record_seconds} seconds... Speak now!")
    frames = []

    for _ in range(0, int(RATE / CHUNK * record_seconds)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(output_file, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    st.success(f"Recording saved to {output_file}")
    return output_file

st.title("Speech-to-Text Application with Microphone Input")

# Option: upload file or record from mic
audio_option = st.radio("Choose input method:", ["Upload Audio File", "Record from Microphone"])

audio_file = None
if audio_option == "Upload Audio File":
    audio_file = st.file_uploader("Upload a WAV file", type=["wav", "flac"])
elif audio_option == "Record from Microphone":
    record_seconds = st.slider("Recording Duration (seconds)", min_value=3, max_value=15, value=5)
    if st.button("Start Recording"):
        audio_file = record_from_mic(record_seconds=record_seconds)
def convert_to_mono(input_file, output_file="temp_mono.wav"):
    data, samplerate = sf.read(input_file)
    if len(data.shape) > 1 and data.shape[1] == 2:
        data = np.mean(data, axis=1)
    sf.write(output_file, data, samplerate)
    return output_file

def transcribe_wav2vec(file_path):
    file_path_16k = resample_to_16k(file_path)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    speech, rate = sf.read(file_path_16k)
    input_values = processor(speech, sampling_rate=rate, return_tensors="pt").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription
# Whisper
def transcribe_whisper(file_path):
    st.info("Whisper: Recognizing...")
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    st.success("Speech successfully converted to text!")
    return result["text"]

# Vosk
def transcribe_vosk(file_path):
    st.info("Vosk: Recognizing...")
    wf = wave.open(file_path, "rb")
    model = Model("vosk-model-small-en-us-0.15")
    rec = KaldiRecognizer(model, wf.getframerate())
    result_text = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result_text += json.loads(rec.Result())['text'] + " "
    result_text += json.loads(rec.FinalResult())['text']
    st.success("Speech successfully converted to text!")
    return result_text.strip()



def transcribe_sphinx(file_path):
    r = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = r.record(source)
    try:
        text = r.recognize_sphinx(audio)
        return text
    except sr.UnknownValueError:
        return "Sphinx could not understand audio"
    except sr.RequestError as e:
        return f"Sphinx error; {e}"
def resample_to_16k(file_path, output_file="temp_16k.wav"):
    # Load audio (any sampling rate)
    audio, sr = librosa.load(file_path, sr=None)
    # Resample to 16 kHz
    audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    # Save as WAV
    sf.write(output_file, audio_16k, 16000)
    return output_file


# Google Cloud
def transcribe_google(file_path):
    st.info("Google Cloud: Recognizing...")
    client = speech.SpeechClient()
    # Convert to mono if needed
    mono_file = convert_to_mono(file_path)
    data, samplerate = sf.read(mono_file)

    with open(mono_file, "rb") as f:
        content = f.read()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=samplerate,
        language_code="en-US"
    )
    try:
        response = client.recognize(config=config, audio=audio)
        text = " ".join([result.alternatives[0].transcript for result in response.results])
        st.success("Speech successfully converted to text!")
        return text
    except Exception as e:
        return f"Error: {str(e)}"
if audio_file:
    if isinstance(audio_file, str):  # recorded from mic
        file_path = audio_file
    else:  # uploaded
        with open("temp.wav", "wb") as f:
            f.write(audio_file.getbuffer())
        file_path = "temp.wav"

    # Convert to mono
    file_path = convert_to_mono(file_path)

    st.subheader("Transcription Outputs")

    whisper_text = transcribe_whisper(file_path)
    st.text_area("Whisper Output", value=whisper_text, height=100)

    vosk_text = transcribe_vosk(file_path)
    st.text_area("Vosk Output", value=vosk_text, height=100)

    google_text = transcribe_google(file_path)
    st.text_area("Google Cloud Output", value=google_text, height=100)
    st.subheader("Additional Transcription Methods")

    wav2vec_text = transcribe_wav2vec(file_path)
    st.text_area("Coqui STT Output", value=wav2vec_text, height=100)

    sphinx_text = transcribe_sphinx(file_path)
    st.text_area("CMU Sphinx Output", value=sphinx_text, height=100)

