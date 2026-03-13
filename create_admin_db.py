import sqlite3

# Connect to SQLite database (creates it if it doesn't exist)
conn = sqlite3.connect("admin.db")
c = conn.cursor()

# Create admin_users table
c.execute('''
CREATE TABLE IF NOT EXISTS admin_users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL
)
''')

# Insert default admin credentials
c.execute("INSERT OR IGNORE INTO admin_users (username, password) VALUES (?, ?)", ("admin", "admin123"))

conn.commit()
conn.close()

print("✅ Admin table created and default credentials inserted.")
