"""
utils/auth.py — Authentication and Database Helpers.
"""

import sqlite3
import bcrypt
import os
import json

DB_NAME = "instaeda.db"


def init_db():
    """Initializes the SQLite database with users and reports tables."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            api_key TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            dataset_name TEXT,
            report_md TEXT,
            raw_results TEXT,
            viz_configs TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (username) REFERENCES users (username)
        )
    ''')
    conn.commit()
    conn.close()


def update_username(old_username, new_username):
    """Updates the username in the database."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        # Update users table
        c.execute("UPDATE users SET username = ? WHERE username = ?", (new_username, old_username))
        # Update reports table (foreign key update)
        c.execute("UPDATE reports SET username = ? WHERE username = ?", (new_username, old_username))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def update_password(username, new_password):
    """Updates the password for a given user."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    hashed = hash_password(new_password)
    c.execute("UPDATE users SET password = ? WHERE username = ?", (hashed, username))
    conn.commit()
    conn.close()


def save_report_to_db(username, dataset_name, report_md, raw_results, viz_configs):
    """Saves an EDA report to the history."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        INSERT INTO reports (username, dataset_name, report_md, raw_results, viz_configs)
        VALUES (?, ?, ?, ?, ?)
    ''', (username, dataset_name, report_md, json.dumps(raw_results), json.dumps(viz_configs)))
    conn.commit()
    conn.close()


def get_user_reports(username):
    """Retrieves all reports for a given user."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        SELECT id, dataset_name, report_md, raw_results, viz_configs, timestamp 
        FROM reports WHERE username = ? ORDER BY timestamp DESC
    ''', (username,))
    rows = c.fetchall()
    conn.close()
    return rows


def delete_report(report_id):
    """Deletes a specific report."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM reports WHERE id = ?", (report_id,))
    conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    """Returns a hashed password."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def check_password(password: str, hashed: str) -> bool:
    """Checks if a password matches its hashed version."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))


def register_user(username, password):
    """Adds a new user to the database."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        hashed = hash_password(password)
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def authenticate_user(username, password):
    """Verifies a user's credentials."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()
    if row and check_password(password, row[0]):
        return True
    return False


def save_api_key(username, api_key):
    """Saves the API key for a given user."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("UPDATE users SET api_key = ? WHERE username = ?", (api_key, username))
    conn.commit()
    conn.close()


def get_api_key(username):
    """Retrieves the API key for a given user."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT api_key FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None
