import psycopg2
import os
# PostgreSQL DB config
DB_CONFIG = {
    "host": "localhost",
    "dbname": "postgres",
    "user": "postgres",
    "password": "1234",
    "port": 5432
}

def insert_audio_file(file_path):
    filename = os.path.basename(file_path)
    with open(file_path, 'rb') as f:
        audio_data = f.read()

    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO audio_input_table (filename, audio_file)
        VALUES (%s, %s);
    """, (filename, psycopg2.Binary(audio_data)))
    conn.commit()
    cursor.close()
    conn.close()
    print(f"Inserted {filename} into audio_input_table")

# Example usage
insert_audio_file("Recording (2).wav")
