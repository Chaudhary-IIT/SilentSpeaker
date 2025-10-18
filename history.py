# history.py
import sqlite3, os, csv, time

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

def save_history_csv(source, src_lang, text_en, text_trans, tgt_lang):
    new_file = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["ts","source","src_lang","text_en","text_trans","tgt_lang"])
        w.writerow([int(time.time()), source, src_lang, text_en, text_trans, tgt_lang])
