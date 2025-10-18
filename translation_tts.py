
from gtts import gTTS
from googletrans import Translator

def translate_text(text: str, target_lang: str = "hi") -> str:
    translator = Translator()
    # auto-detect source, translate to Hindi (hi)
    return translator.translate(text, dest=target_lang).text

def synthesize_tts(text: str, lang: str = "hi", out_path: str = "static/audio/hi.mp3") -> str:
    tts = gTTS(text=text or " ", lang=lang)
    tts.save(out_path)
    return out_path
