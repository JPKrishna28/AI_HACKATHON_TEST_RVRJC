from kafka import KafkaProducer
import os
import glob
from pathlib import Path

KAFKA_BOOTSTRAP_SERVERS = ['localhost:9092']
TOPIC = 'audio-files'

class AudioKafkaProducer:
    def __init__(self, bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS, topic=TOPIC):
        self.producer = KafkaProducer(bootstrap_servers=bootstrap_servers)
        self.topic = topic
    
    def send_audio_file_in_chunks(self, file_path, chunk_size=1024 * 1024):
        """Send a single audio file in chunks to Kafka"""
        filename = os.path.basename(file_path)
        try:
            file_size = os.path.getsize(file_path)
            print(f"Processing {filename} ({file_size / (1024*1024):.2f} MB)...")
            
            with open(file_path, 'rb') as f:
                chunk_count = 0
                chunk = f.read(chunk_size)
                while chunk:
                    self.producer.send(self.topic, key=filename.encode('utf-8'), value=chunk)
                    chunk_count += 1
                    chunk = f.read(chunk_size)
            
            self.producer.flush()
            print(f"✓ Sent {filename} in {chunk_count} chunks to Kafka topic '{self.topic}'")
            return True
            
        except FileNotFoundError:
            print(f"✗ Error: File {file_path} not found.")
            return False
        except Exception as e:
            print(f"✗ Error processing {filename}: {e}")
            return False
    
    def send_multiple_files(self, file_paths, chunk_size=1024 * 1024):
        """Send multiple audio files to Kafka"""
        successful = 0
        failed = 0
        
        print(f"Starting to process {len(file_paths)} files...")
        print("-" * 50)
        
        for i, file_path in enumerate(file_paths, 1):
            print(f"[{i}/{len(file_paths)}] ", end="")
            if self.send_audio_file_in_chunks(file_path, chunk_size):
                successful += 1
            else:
                failed += 1
        
        print("-" * 50)
        print(f"Processing complete: {successful} successful, {failed} failed")
        return successful, failed
    
    def send_files_from_directory(self, directory_path, file_extensions=None, chunk_size=1024 * 1024):
        """Send all audio files from a directory"""
        if file_extensions is None:
            file_extensions = ['*.mp3', '*.wav', '*.m4a', '*.flac', '*.aac', '*.ogg']
        
        file_paths = []
        for extension in file_extensions:
            pattern = os.path.join(directory_path, extension)
            file_paths.extend(glob.glob(pattern))
        
        if not file_paths:
            print(f"No audio files found in {directory_path}")
            return 0, 0
        
        print(f"Found {len(file_paths)} audio files in {directory_path}")
        return self.send_multiple_files(file_paths, chunk_size)
    
    def close(self):
        """Close the Kafka producer"""
        self.producer.close()
        print("Kafka producer closed.")

def get_files_from_patterns(patterns):
    """Get file paths from glob patterns"""
    all_files = []
    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            all_files.extend(files)
        else:
            # If no glob pattern match, treat as direct file path
            if os.path.exists(pattern):
                all_files.append(pattern)
            else:
                print(f"Warning: No files found matching pattern '{pattern}'")
    return all_files

if __name__ == '__main__':
    kafka_producer = AudioKafkaProducer()
    
    try:
        # Method 1: Send specific files by name
        file_list = [
            'Recording (5).m4a',
            'Recording (6).m4a',
            'Recording (2).m4a'
        ]
        print("=== Method 1: Sending specific files ===")
        kafka_producer.send_multiple_files(file_list)
        
        print("\n" + "="*60 + "\n")
        
        # Method 2: Send all audio files from current directory
        print("=== Method 2: Sending all audio files from current directory ===")
        kafka_producer.send_files_from_directory('.')
        
        print("\n" + "="*60 + "\n")
        
        # Method 3: Send files using glob patterns
        print("=== Method 3: Sending files using glob patterns ===")
        patterns = [
            'Recording*.m4a',     # All recordings
            '*.mp3',              # All MP3 files
            'path/to/audio/*.wav' # All WAV files in specific directory
        ]
        files_from_patterns = get_files_from_patterns(patterns)
        if files_from_patterns:
            kafka_producer.send_multiple_files(files_from_patterns)
        
        print("\n" + "="*60 + "\n")
        
        # Method 4: Send from specific directory with custom extensions
        print("=== Method 4: Sending from specific directory ===")
        # kafka_producer.send_files_from_directory('path/to/audio/files', ['*.m4a', '*.mp3'])
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        kafka_producer.close()