from googletrans import Translator
from gtts import gTTS

def translate_text(text, target_lang="Hindi"):
    lang_map = {
        "Hindi": "hi",
        "English": "en",
        "Bengali": "bn",
        "Tamil": "ta",
        "Telugu": "te"
    }
    translator = Translator()
    dest_code = lang_map.get(target_lang, "hi")
    if dest_code == "en":
        return text
    result = translator.translate(text, src='en', dest=dest_code)
    return result.text

def synthesize_tts(text, lang="Hindi", out_path="output.mp3"):
    lang_map = {
        "Hindi": "hi",
        "English": "en",
        "Bengali": "bn",
        "Tamil": "ta",
        "Telugu": "te"
    }
    tts_code = lang_map.get(lang, "hi")
    tts = gTTS(text=text, lang=tts_code)
    tts.save(out_path)
