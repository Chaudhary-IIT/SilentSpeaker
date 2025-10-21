# history.py
import sqlite3, os, csv, time
from pathlib import Path
from datetime import datetime
import pandas as pd

DB_PATH = "static/history.db"
CSV_PATH = "static/history.csv"
os.makedirs("static", exist_ok=True)

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
      CREATE TABLE IF NOT EXISTS history(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts INTEGER,
        source TEXT,
        src_lang TEXT,
        text_en TEXT,
        text_trans TEXT,
        tgt_lang TEXT
      )
    """)
    con.commit()
    con.close()

def save_history_sqlite(source, src_lang, text_en, text_trans, tgt_lang):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO history(ts, source, src_lang, text_en, text_trans, tgt_lang) VALUES (?, ?, ?, ?, ?, ?)",
        (int(time.time()), source, src_lang, text_en, text_trans, tgt_lang),
    )
    con.commit()
    con.close()


def save_history_csv(source_type, source_lang, original_text, translated_text, target_lang, audio_path=""):
    Path("static").mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {
        'source_type': [source_type],
        'source_lang': [source_lang],
        'original_text': [original_text],
        'translated_text': [translated_text],
        'target_lang': [target_lang],
        'audio_path': [audio_path],
        'timestamp': [timestamp]
    }
    df = pd.DataFrame(data)
    csv_path = 'static/history.csv'
    if not Path(csv_path).exists():
        df.to_csv(csv_path, mode='w', index=False, header=True)
    else:
        df.to_csv(csv_path, mode='a', index=False, header=False)