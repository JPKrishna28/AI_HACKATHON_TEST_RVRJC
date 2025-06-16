from kafka import KafkaConsumer
import psycopg2
import random

# PostgreSQL DB config
DB_CONFIG = {
    "host": "localhost",
    "dbname": "postgres",
    "user": "postgres",
    "password": "1234",
    "port": 5432
}

KAFKA_BOOTSTRAP_SERVERS = ['localhost:9092']
TOPIC = 'audio-files'

def insert_audio_to_db(filename, audio_data):
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM audio_input_table WHERE filename = %s", (filename,))
        if cursor.fetchone():
            print(f"[SKIP] Duplicate file {filename}")
        else:
            cursor.execute("""
                INSERT INTO audio_input_table (filename, audio_file, processed)
                VALUES (%s, %s, %s);
            """, (filename, psycopg2.Binary(audio_data), False))
            conn.commit()
            print(f"[INSERTED] {filename}")
    except Exception as e:
        print(f"DB error: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

def consume_audio_files():
    group_id = f'audio-file-consumer-group-{random.randint(1,1000000)}'

    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id=group_id,
        session_timeout_ms=30000,
        heartbeat_interval_ms=10000,
        max_partition_fetch_bytes=1024 * 1024 * 10  # 10 MB
    )

    print(f"Listening on topic '{TOPIC}' with group '{group_id}'...")

    for message in consumer:
        try:
            filename = message.key.decode('utf-8') if message.key else "unknown"
            audio_data = message.value
            print(f"[RECEIVED] {filename} ({len(audio_data)} bytes)")

            insert_audio_to_db(filename, audio_data)

        except Exception as e:
            print(f"[ERROR] Exception processing message: {e}")

if __name__ == '__main__':
    consume_audio_files()
