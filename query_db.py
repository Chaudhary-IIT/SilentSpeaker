import sqlite3

conn = sqlite3.connect("static/history.db")
cur = conn.cursor()

cur.execute("SELECT * FROM history")
rows = cur.fetchall()
if not rows:
    print("No rows found in the history table.")
for row in rows:
    print(row)

conn.close()
