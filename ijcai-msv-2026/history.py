from dataclasses import asdict
from datetime import datetime, timezone
import json
import subprocess

import sqlite3
from sqlite3 import Error
from typing import Any

from metacognitive import MetacognitiveVector

def get_git_revision_hash() -> dict:
    try:
        git_revision_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        git_revision_short_hash = git_revision_hash[:7]

    except Exception as e:
        print(f"Exception retrieving git commit hash {e}")
        git_revision_hash = "unavailable"
        git_revision_short_hash = "unavailable"
    
    hashes = {"git_revision_hash": git_revision_hash, "git_revision_short_hash": git_revision_short_hash}
    return hashes

def create_database_and_table(db_file: str, configuration: dict[str, dict[str, Any]]) -> bool:
    """Create a SQLite database and a table called Interactions if it doesn't exist."""
    try:
        # Connect to the SQLite database (or create it if it doesn't exist)
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # Create the Interactions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                datetime TEXT NOT NULL,
                user_prompt TEXT NOT NULL,
                system_one_response TEXT NOT NULL,
                system_one_msv TEXT NOT NULL,
                system_two_response TEXT,
                system_two_msv TEXT
            )
        """)
        # Create the parameters table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS parameters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                datetime TEXT NOT NULL,
                parameters TEXT NOT NULL
            )
        """)
        # Commit the changes and close the connection
        conn.commit()
        
        configuration |= get_git_revision_hash()
        cursor.execute("""INSERT INTO parameters(datetime, parameters) VALUES (?, ?)""", (datetime.now(timezone.utc).isoformat(), json.dumps(configuration)))
        conn.commit()
        print("Database and table created successfully.")
        return True
    except Error as e:
        print(f"An error occurred: {e}")
        return False
    finally:
        if conn:
            conn.close()


def record_interaction(db_file: str, user_prompt: str, system_one_response: str, system_one_msv: MetacognitiveVector, system_two_response: str | None, system_two_msv: MetacognitiveVector | None) -> None:
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # Prepare the SQL insert statement
        cursor.execute('''
            INSERT INTO interactions (datetime, user_prompt, system_one_response, 
                                      system_two_response, system_one_msv, system_two_msv)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now(timezone.utc).isoformat(),
            user_prompt,
            system_one_response,
            system_two_response,
            json.dumps(asdict(system_one_msv)),
            json.dumps(asdict(system_two_msv)) if system_two_msv is not None else None
        ))

        # Commit the changes
        conn.commit()
    except Error as e:
        print(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()