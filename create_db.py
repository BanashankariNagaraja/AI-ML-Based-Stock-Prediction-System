import sqlite3

conn = sqlite3.connect('stock_market.db')

cur = conn.cursor()

# USERS TABLE

cur.execute('''
CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    email TEXT,
    password TEXT
)
''')

# CONTACT MESSAGES TABLE

cur.execute('''
CREATE TABLE IF NOT EXISTS messages(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT,
    subject TEXT,
    message TEXT
)
''')

conn.commit()
conn.close()

print("Database Created Successfully")