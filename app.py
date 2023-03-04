
import pyaudio
import wave
from pydub import AudioSegment
import openai
from gtts import gTTS
import os
import wave
import pyaudio
from transformers import (Wav2Vec2Processor, Wav2Vec2ForCTC)
import torchaudio
import torch
from mtranslate import translate

# Define constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 6
MP3_OUTPUT_FILENAME = "output.mp3"

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Start recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
print("Recording started...")
frames = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
print("Recording stopped.")

# Stop recording and save file
stream.stop_stream()
stream.close()
audio.terminate()
waveFile = wave.open("output.wav", 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()

#get the arabic text from sound
def speech_file_to_array_fn(voice_path, resampling_to=16000):
    speech_array, sampling_rate = torchaudio.load(voice_path)
    resampler = torchaudio.transforms.Resample(sampling_rate, resampling_to)
    
    return resampler(speech_array)[0].numpy(), sampling_rate

# load the model
cp = "bakrianoo/sinai-voice-ar-stt"
processor = Wav2Vec2Processor.from_pretrained(cp)
model = Wav2Vec2ForCTC.from_pretrained(cp)

# recognize the text in a sample sound file
sound_path = 'output.wav'

sample, sr = speech_file_to_array_fn(sound_path)
inputs = processor([sample], sampling_rate=16_000, return_tensors="pt", padding=True)

with torch.no_grad():
    logits = model(inputs.input_values,).logits

predicted_ids = torch.argmax(logits, dim=-1)

arabic_text = str(processor.batch_decode(predicted_ids)[0])
english_text = translate(arabic_text, "en")


openai.api_key =  "*******" # Modify this
model2 = "text-davinci-002"
temperature = 0.5
max_tokens = 60
response = openai.Completion.create(
                engine=model2,
                prompt=english_text,
                temperature=temperature,
                max_tokens=max_tokens,
           )
with open("output.txt", "w") as f:
    f.write(str(response.choices[0].text.strip()))
       
arab_text = translate(response.choices[0].text.strip(), "ar")
# The text that you want to convert to audio
# Language in which you want to convert
language = 'ar'
# Passing the text and language to the engine, 
# here we have marked slow=False. Which tells 
# the module that the converted audio should 
# have a high speed
myobj = gTTS(text=arab_text, lang=language, slow=False)
# Saving the converted audio in a mp3 file named
myobj.save("output.mp3")
# Playing the converted file
os.system("output.mp3")



