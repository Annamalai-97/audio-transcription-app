from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torchaudio
import torch
from pydub import AudioSegment
import io

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# Load pre-trained Whisper model and processor
model_name = "openai/whisper-base"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)
model.to('cuda' if torch.cuda.is_available() else 'cpu')

def load_audio(file):
    audio = AudioSegment.from_file(file)
    audio = audio.set_frame_rate(16000)  # Whisper models use 16kHz sampling rate
    audio = audio.set_channels(1)        # Convert to mono
    audio_wav = io.BytesIO()
    audio.export(audio_wav, format="wav")
    waveform, sample_rate = torchaudio.load(audio_wav)
    return waveform

def transcribe_audio_chunks(waveform):
    sample_rate = 16000
    chunk_duration = 30  # seconds
    chunk_size = chunk_duration * sample_rate
    num_chunks = (waveform.size(1) + chunk_size - 1) // chunk_size
    full_transcription = []

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, waveform.size(1))
        chunk_waveform = waveform[:, start:end]
        
        inputs = processor(chunk_waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=sample_rate, language='en')
        inputs = {key: value.to('cuda' if torch.cuda.is_available() else 'cpu') for key, value in inputs.items()}
        
        with torch.amp.autocast('cuda'):
            with torch.no_grad():
                logits = model.generate(inputs["input_features"])

        transcription = processor.batch_decode(logits, skip_special_tokens=True)
        full_transcription.append(transcription[0])
        print(f"Processing chunk {i + 1}/{num_chunks}...")

    return " ".join(full_transcription)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    file = request.files['file']
    waveform = load_audio(file)
    transcription = transcribe_audio_chunks(waveform)
    return jsonify({'transcription': transcription})

if __name__ == '__main__':
    app.run(debug=True)
