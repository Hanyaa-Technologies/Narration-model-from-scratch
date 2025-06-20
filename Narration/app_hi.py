import subprocess
import os
from pydub import AudioSegment
from indicnlp.tokenize import sentence_tokenize  # हिंदी वाक्य विभाजन के लिए

def text_to_speech_long(text, output_file="output_hindi.wav", max_chunk_length=500):
    # टेक्स्ट को वाक्यों/भागों में विभाजित करें
    sentences = sentence_tokenize.sentence_split(text, lang='hi')
    
    # या फिर सरल विभाजन (यदि indicnlp उपलब्ध न हो)
    # sentences = [s.strip() for s in text.split('।') if s.strip()]
    
    temp_files = []
    
    # प्रत्येक भाग के लिए अलग-अलग ऑडियो बनाएं
    for i, chunk in enumerate(sentences):
        if not chunk:
            continue
            
        # अस्थायी फाइल नाम
        temp_file = f"temp_part_{i}.wav"
        
        # TTS कमांड
        command = [
            "tts",
            "--text", chunk,
            "--model_path", "./hi/fastpitch/best_model.pth",
            "--config_path", "./hi/fastpitch/config.json",
            "--speakers_file_path", "./hi/fastpitch/speakers.pth",
            "--vocoder_path", "./hi/hifigan/best_model.pth",
            "--vocoder_config_path", "./hi/hifigan/config.json",
            "--out_path", temp_file,
            "--speaker_idx", "female"
        ]
        
        # कमांड रन करें
        subprocess.run(command)
        temp_files.append(temp_file)
    
    # सभी ऑडियो भागों को जोड़ें
    combined = AudioSegment.empty()
    for f in temp_files:
        combined += AudioSegment.from_wav(f)
        os.remove(f)  # अस्थायी फाइल डिलीट करें
    
    # फाइल सेव करें
    combined.export(output_file, format="wav")
    print(f"ऑडियो फाइल सेव की गई: {output_file}")

# उदाहरण उपयोग
long_text = """गोतम बुद्ध से माना जाता हे, क्योंकि उस काल की बोद्ध-कथाओं में वर्णित व्यक्तियों का पुराणों की वंशावली में भी प्रसंग आता है। लोग वहीं से प्रामाणिक इतिहास मानते हैं।"""

text_to_speech_long(long_text)