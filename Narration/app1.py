from flask import Flask, render_template, request, send_file, jsonify, session
from flask import send_from_directory
import os
import uuid
import numpy as np
from bark import generate_audio, SAMPLE_RATE
import re
import logging
import tempfile
from scipy.io.wavfile import write
from langdetect import detect, LangDetectException
import subprocess
from subprocess import Popen
from pydub import AudioSegment
from indicnlp.tokenize import sentence_tokenize
from threading import Event
from collections import defaultdict
from werkzeug.exceptions import ClientDisconnected


app = Flask(__name__)
# Session secret key (replace with a secure value in production)
app.secret_key = os.getenv('SECRET_KEY', 'replace-me-with-real-secret')


# Registry: session_id -> { 'cancel_event': Event, 'procs': [Popen, ...] }
running_tasks = defaultdict(lambda: {'cancel_event': None, 'procs': []})

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Predefined speaker lists per language
SPEAKERS = {
    'english': [
        'YouTubeaudio.npz', 'en_british.npz',  'en_fiery.npz',
        'en_german_professor.npz'],
    'hindi': ['speaker_h1.npz', 'speaker_h2.npz'],
    'telugu': ['speaker_t1.npz', 'speaker_t2.npz']
}

HINDI_SPEAKER_MAP = {
    'speaker_h1.npz': 'female',
    'speaker_h2.npz': 'male'
}
TELUGU_SPEAKER_MAP = {
    'speaker_t1.npz': 'female',
    'speaker_t2.npz': 'male'
}
LANGUAGE_ISO_MAP = {
    'english': 'en',
    'hindi': 'hi',
    'telugu': 'te'
}
LANGUAGE_NAME_MAP = {
    'en': 'English',
    'hi': 'Hindi',
    'te': 'Telugu'
}

# Preload BARK models once
try:
    logger.info("Preloading BARK models...")
    from bark.generation import preload_models
    preload_models()
    logger.info("BARK models loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load BARK models: {e}")
    raise RuntimeError("BARK model loading failed") from e

@app.route('/get_speakers')
def get_speakers():
    language = request.args.get('language', 'english').lower().strip()
    speakers = SPEAKERS.get(language, [])
    logger.info(f"Returning speakers for {language}: {speakers}")
    return jsonify(speakers)

@app.route('/')
def index():
    return render_template('index.html', languages=list(SPEAKERS.keys()))

# Cancellation helper
def cancel_previous_task(sid):
    task = running_tasks.get(sid)
    if not task:
        return
    # Signal Python loops to stop
    if task['cancel_event'] is not None:
        task['cancel_event'].set()
    # Terminate any subprocesses
    for p in task['procs']:
        try:
            p.terminate()
        except Exception:
            pass
    running_tasks.pop(sid, None)

# Hindi TTS with subprocess tracking
def text_to_speech_long_hindi(text, output_file, speaker_idx, sid, max_chunk_length=500):
    sentences = sentence_tokenize.sentence_split(text, lang='hi')
    temp_files = []
    for i, chunk in enumerate(sentences):
        if not chunk:
            continue
        temp_file = f"temp_hindi_{i}.wav"
        command = [
            "tts", "--text", chunk,
            "--model_path", "./hi/fastpitch/best_model.pth",
            "--config_path", "./hi/fastpitch/config.json",
            "--speakers_file_path", "./hi/fastpitch/speakers.pth",
            "--vocoder_path", "./hi/hifigan/best_model.pth",
            "--vocoder_config_path", "./hi/hifigan/config.json",
            "--out_path", temp_file,
            "--speaker_idx", speaker_idx
        ]
        # launch and track
        p = Popen(command)
        running_tasks[sid]['procs'].append(p)
        p.wait()
        temp_files.append(temp_file)
    combined = AudioSegment.empty()
    for f in temp_files:
        combined += AudioSegment.from_wav(f)
        os.remove(f)
    combined.export(output_file, format="wav")
    logger.info(f"Hindi audio generated at {output_file}")

# Telugu TTS with subprocess tracking
def text_to_speech_long_telugu(text, output_file, speaker_idx, sid, max_chunk_length=500):
    sentences = sentence_tokenize.sentence_split(text, lang='te')
    def split_text(chunk, max_length):
        return [chunk[i:i+max_length] for i in range(0, len(chunk), max_length)]
    temp_files = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            chunks = [sentence] if len(sentence) <= max_chunk_length else split_text(sentence, max_chunk_length)
            for j, chunk in enumerate(chunks):
                temp_file = os.path.join(temp_dir, f"temp_telugu_{i}_{j}.wav")
                command = [
                    "tts", "--text", chunk,
                    "--model_path", "./te/fastpitch/best_model.pth",
                    "--config_path", "./te/fastpitch/config.json",
                    "--speakers_file_path", "./te/fastpitch/speakers.pth",
                    "--vocoder_path", "./te/hifigan/best_model.pth",
                    "--vocoder_config_path", "./te/hifigan/config.json",
                    "--out_path", temp_file,
                    "--speaker_idx", speaker_idx
                ]
                p = Popen(command)
                running_tasks[sid]['procs'].append(p)
                p.wait()
                temp_files.append(temp_file)
        combined = AudioSegment.empty()
        for f in sorted(temp_files):
            combined += AudioSegment.from_wav(f)
        combined.export(output_file, format="wav")
        logger.info(f"Telugu audio generated at {output_file}")

# Simple splitter for English
def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

@app.route('/generate_audio', methods=['POST'])
def generate_audio_route():
    # Get or assign session ID
    sid = session.get('sid')
    if sid is None:
        sid = str(uuid.uuid4())
        session['sid'] = sid
    # Cancel any previous task for this session
    cancel_previous_task(sid)
    # Prepare new task context
    running_tasks[sid]['cancel_event'] = Event()
    running_tasks[sid]['procs'] = []
    cancel_event = running_tasks[sid]['cancel_event']
    try:
        text = request.form.get('text', '').strip()
        language = request.form.get('language', '').lower().strip()
        speaker_file = request.form.get('speaker', '').strip()
        # Input validation
        if not text or len(text) < 3:
            return jsonify({"error": "Text must be at least 3 characters long"}), 400
        if language not in SPEAKERS:
            return jsonify({"error": "Invalid language selected"}), 400
        if speaker_file not in SPEAKERS[language]:
            return jsonify({"error": "Invalid speaker selected"}), 400
        # Detect language
        try:
            detected_lang = detect(text)
        except LangDetectException as e:
            logger.error(f"Language detection failed: {e}")
            return jsonify({"error": "Could not detect language."}), 400
        selected_iso = LANGUAGE_ISO_MAP.get(language)
        if detected_lang != selected_iso:
            detected_name = LANGUAGE_NAME_MAP.get(detected_lang, detected_lang.upper())
            return jsonify({"error": f"Text appears to be in {detected_name}"}), 400
        # Hindi
        if language == 'hindi':
            speaker_idx = HINDI_SPEAKER_MAP.get(speaker_file, 'female')
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                out_path = tmp.name
            text_to_speech_long_hindi(text, out_path, speaker_idx, sid)
            response = send_file(out_path, mimetype='audio/wav')
            running_tasks.pop(sid, None)
            return response
        # Telugu
        if language == 'telugu':
            speaker_idx = TELUGU_SPEAKER_MAP.get(speaker_file, 'female')
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                out_path = tmp.name
            text_to_speech_long_telugu(text, out_path, speaker_idx, sid)
            response = send_file(out_path, mimetype='audio/wav')
            running_tasks.pop(sid, None)
            return response
        # English (BARK)
        prompt_path = os.path.join('bark', 'assets', 'prompts', speaker_file)
        if not os.path.exists(prompt_path):
            return jsonify({"error": "Selected speaker not found"}), 404
        history_npz = np.load(prompt_path, allow_pickle=True)
        history_prompt = {k: history_npz[k] for k in history_npz.files}
        sentences = split_into_sentences(text)
        if not sentences:
            return jsonify({"error": "No valid sentences found"}), 400
        silence = np.zeros(int(0.25 * SAMPLE_RATE))
        pieces = []
        for sentence in sentences:
            if cancel_event.is_set():
                logger.info(f"Generation for session {sid} cancelled.")
                break
            try:
                audio_array = generate_audio(sentence, history_prompt=history_prompt)
                pieces.append(audio_array)
                pieces.append(silence.copy())
            except Exception as e:
                logger.error(f"Generation error: {e}")
                continue
        if not pieces:
            return jsonify({"error": "Audio generation failed for all sentences"}), 500
        final_audio = np.concatenate(pieces)
        normalized = final_audio / (np.max(np.abs(final_audio)) or 1.0)
        pcm16 = (normalized * 32767).astype(np.int16)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            write(tmp.name, SAMPLE_RATE, pcm16)
            out_path = tmp.name
        response = send_file(out_path, mimetype='audio/wav')
        running_tasks.pop(sid, None)
        return response
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
