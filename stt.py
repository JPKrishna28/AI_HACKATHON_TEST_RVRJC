# Enhanced Audio Processing Pipeline with Gemini AI Threat Detection

import requests
import io
import os
import psycopg2
from pydub import AudioSegment
import tempfile
import logging
from pathlib import Path
import google.generativeai as genai
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Configuration
SARVAM_AI_API = ""
GEMINI_API_KEY = ""  # Replace with your actual Gemini API key

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Sarvam AI configuration
api_url = "https://api.sarvam.ai/speech-to-text-translate"
headers = {"api-subscription-key": SARVAM_AI_API}
data = {
    "model": "saaras:v2",
    "with_diarization": False
}

# DB Configuration
DB_CONFIG = {
    "host": "localhost",
    "dbname": "postgres",
    "user": "postgres",
    "password": "1234",
    "port": 5432
}


def validate_audio_file(file_path):
    """Validate if audio file is readable and not corrupted"""
    try:
        # Check if file exists and has content
        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            return False

        if os.path.getsize(file_path) == 0:
            logger.error(f"File is empty: {file_path}")
            return False

        # Try to load with pydub to check if it's valid
        audio = AudioSegment.from_file(file_path)
        logger.info(f"Audio file validated: {file_path} - Duration: {len(audio)}ms")
        return True

    except Exception as e:
        logger.error(f"Audio validation failed for {file_path}: {e}")
        return False


def convert_to_wav(input_audio_path, output_dir=None):
    """Convert audio file to WAV format with error handling"""
    try:
        # Validate input file first
        if not validate_audio_file(input_audio_path):
            return None

        base = os.path.splitext(os.path.basename(input_audio_path))[0]

        if output_dir:
            wav_path = os.path.join(output_dir, f"{base}.wav")
        else:
            wav_path = f"{base}_converted.wav"

        # If already WAV and valid, return original path
        if input_audio_path.lower().endswith('.wav'):
            return input_audio_path

        # Convert to WAV
        audio = AudioSegment.from_file(input_audio_path)
        audio.export(wav_path, format="wav")
        logger.info(f"Converted {input_audio_path} to {wav_path}")
        return wav_path

    except Exception as e:
        logger.error(f"Audio conversion failed: {e}")
        return None


def split_audio(audio_path, chunk_duration_ms=5 * 60 * 1000):
    """Split audio into chunks with error handling"""
    try:
        if not validate_audio_file(audio_path):
            return []

        audio = AudioSegment.from_file(audio_path)

        # If audio is shorter than chunk duration, return as single chunk
        if len(audio) <= chunk_duration_ms:
            return [audio]

        chunks = []
        for i in range(0, len(audio), chunk_duration_ms):
            chunk = audio[i:i + chunk_duration_ms]
            chunks.append(chunk)

        logger.info(f"Split audio into {len(chunks)} chunks")
        return chunks

    except Exception as e:
        logger.error(f"Audio splitting failed: {e}")
        return []


def translate_audio(audio_file_path):
    """Transcribe audio with improved error handling"""
    try:
        chunks = split_audio(audio_file_path, chunk_duration_ms=5 * 60 * 1000)

        if not chunks:
            logger.error("No valid audio chunks to process")
            return ""

        responses = []

        for idx, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {idx + 1}/{len(chunks)}")

            # Create temporary buffer for chunk
            chunk_buffer = io.BytesIO()
            try:
                chunk.export(chunk_buffer, format="wav")
                chunk_buffer.seek(0)

                files = {"file": ("audiofile.wav", chunk_buffer, "audio/wav")}

                response = requests.post(api_url, headers=headers, files=files, data=data, timeout=30)

                if response.status_code in [200, 201]:
                    result = response.json()
                    transcript = result.get("transcript", "")
                    responses.append(transcript)
                    logger.info(f"Chunk {idx + 1} transcribed successfully")
                else:
                    logger.error(f"Chunk {idx + 1} failed: {response.status_code} - {response.text}")

            except requests.exceptions.RequestException as e:
                logger.error(f"Request error for chunk {idx + 1}: {e}")
            except Exception as e:
                logger.error(f"Chunk {idx + 1} processing error: {e}")
            finally:
                chunk_buffer.close()

        full_transcript = " ".join(responses)
        logger.info(f"Full transcript length: {len(full_transcript)} characters")
        return full_transcript

    except Exception as e:
        logger.error(f"Audio transcription failed: {e}")
        return ""


def classify_threat_with_gemini(transcript):
    """Classify threat type using Gemini AI with detailed analysis"""
    if not transcript or not transcript.strip():
        return {
            "threat_type": "unknown",
            "confidence": 0.0,
            "severity": "low",
            "analysis": "No transcript available for analysis",
            "keywords": []
        }

    try:
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-2.0-flash')

        # Create a detailed prompt for threat analysis
        prompt = f"""
        Analyze the following audio transcript for potential security threats and safety concerns. 
        Provide a detailed analysis in JSON format.

        Transcript: "{transcript}"

        Please analyze and classify this transcript according to the following categories:
        1. theft (stealing, robbery, snatching, burglary, pickpocketing)
        2. land_dispute (property disputes, boundary issues, ownership conflicts)
        3. domestic_violence (physical abuse, domestic fights, family violence)
        4. harassment (stalking, threats, intimidation, bullying)
        5. assault (physical attacks, violence against persons)
        6. fraud (scams, financial fraud, cheating)
        7. vandalism (property damage, destruction)
        8. drug_related (drug dealing, substance abuse issues)
        9. noise_complaint (loud music, disturbances)
        10. medical_emergency (health emergencies, accidents)
        11. fire_emergency (fire, smoke, burning)
        12. unknown (cannot be classified or no threat detected)

        Return your analysis in this exact JSON format:
        {{
            "threat_type": "category_name",
            "confidence": 0.0-1.0,
            "severity": "low|medium|high|critical",
            "analysis": "detailed explanation of your reasoning",
            "keywords": ["list", "of", "relevant", "keywords", "found"],
            "urgent": true/false,
            "recommended_action": "suggested response or action"
        }}

        Consider context, tone, and urgency when making your assessment.
        """

        # Generate response with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)

                # Extract JSON from response
                response_text = response.text.strip()

                # Try to find JSON in the response
                if response_text.startswith('```json'):
                    json_str = response_text.split('```json')[1].split('```')[0].strip()
                elif response_text.startswith('```'):
                    json_str = response_text.split('```')[1].strip()
                elif response_text.startswith('{'):
                    json_str = response_text
                else:
                    # Look for JSON pattern in the text
                    import re
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group()
                    else:
                        raise ValueError("No JSON found in response")

                # Parse JSON
                result = json.loads(json_str)

                # Validate required fields
                required_fields = ["threat_type", "confidence", "severity", "analysis"]
                for field in required_fields:
                    if field not in result:
                        result[field] = "unknown" if field == "threat_type" else "Not provided"

                # Ensure confidence is a float between 0 and 1
                try:
                    result["confidence"] = float(result["confidence"])
                    result["confidence"] = max(0.0, min(1.0, result["confidence"]))
                except:
                    result["confidence"] = 0.5

                logger.info(f"Gemini classification: {result['threat_type']} (confidence: {result['confidence']:.2f})")
                return result

            except Exception as e:
                logger.warning(f"Gemini API attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    # Fallback to basic classification
                    logger.error("All Gemini attempts failed, using fallback classification")
                    return classify_threat_fallback(transcript)
                time.sleep(2 ** attempt)  # Exponential backoff

    except Exception as e:
        logger.error(f"Gemini threat classification failed: {e}")
        return classify_threat_fallback(transcript)


def classify_threat_fallback(transcript):
    """Fallback threat classification using keyword matching"""
    if not transcript:
        return {
            "threat_type": "unknown",
            "confidence": 0.0,
            "severity": "low",
            "analysis": "Fallback analysis - no transcript available",
            "keywords": []
        }

    threats = {
        "theft": {
            "keywords": ["steal", "thief", "robbery", "snatching", "chain snatching", "purse", "wallet", "pickpocket"],
            "severity": "high"
        },
        "land_dispute": {
            "keywords": ["land", "property", "boundary", "property line", "dispute", "ownership", "encroachment"],
            "severity": "medium"
        },
        "domestic_violence": {
            "keywords": ["abuse", "hit", "violence", "scream", "fight", "domestic", "assault", "beat"],
            "severity": "critical"
        },
        "harassment": {
            "keywords": ["harass", "threat", "intimidate", "bully", "stalking", "menace"],
            "severity": "medium"
        },
        "assault": {
            "keywords": ["attack", "assault", "violence", "fight", "hurt", "injure"],
            "severity": "high"
        },
        "medical_emergency": {
            "keywords": ["emergency", "accident", "injured", "help", "ambulance", "medical"],
            "severity": "critical"
        }
    }

    transcript_lower = transcript.lower()
    found_keywords = []
    best_match = {"type": "unknown", "count": 0, "severity": "low"}

    for threat_type, threat_data in threats.items():
        keyword_count = 0
        threat_keywords = []

        for keyword in threat_data["keywords"]:
            if keyword in transcript_lower:
                keyword_count += 1
                threat_keywords.append(keyword)
                found_keywords.append(keyword)

        if keyword_count > best_match["count"]:
            best_match = {
                "type": threat_type,
                "count": keyword_count,
                "severity": threat_data["severity"]
            }

    confidence = min(0.8, best_match["count"] * 0.2) if best_match["count"] > 0 else 0.1

    return {
        "threat_type": best_match["type"],
        "confidence": confidence,
        "severity": best_match["severity"],
        "analysis": f"Fallback keyword-based analysis. Found {best_match['count']} relevant keywords.",
        "keywords": found_keywords,
        "urgent": best_match["severity"] in ["high", "critical"],
        "recommended_action": "Review and assess manually"
    }


def process_audio_from_db():
    """Main processing function with improved error handling and Gemini integration"""
    conn = None
    temp_dir = None

    try:
        # Create temporary directory for audio files
        temp_dir = tempfile.mkdtemp()
        logger.info(f"Created temporary directory: {temp_dir}")

        # Connect to database
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Fetch unprocessed audio records
        cursor.execute("SELECT id, audio_file, filename FROM audio_input_table WHERE processed = FALSE;")
        rows = cursor.fetchall()

        logger.info(f"Found {len(rows)} unprocessed audio records")

        for row in rows:
            record_id, audio_bytes, filename = row
            logger.info(f"Processing record ID: {record_id}, filename: {filename}")

            try:
                # Save audio file to temporary directory
                temp_audio_path = os.path.join(temp_dir, filename)
                with open(temp_audio_path, "wb") as f:
                    f.write(audio_bytes)

                # Validate and convert audio file
                wav_path = convert_to_wav(temp_audio_path, temp_dir)

                if not wav_path:
                    logger.error(f"Failed to convert audio for record {record_id}")
                    # Mark as processed but with error
                    cursor.execute("""
                        INSERT INTO threat_analysis_results (
                            audio_id, transcript, threat_type, confidence, severity, 
                            analysis, error_message
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s);
                    """, (record_id, "", "processing_error", 0.0, "low",
                          "Failed to convert audio file", "Failed to convert audio file"))

                    cursor.execute("UPDATE audio_input_table SET processed = TRUE WHERE id = %s;", (record_id,))
                    conn.commit()
                    continue

                # Transcribe audio
                transcript = translate_audio(wav_path)

                if not transcript:
                    logger.warning(f"No transcript generated for record {record_id}")
                    transcript = "No transcript available"

                # Classify threat using Gemini AI
                threat_analysis = classify_threat_with_gemini(transcript)

                # Read processed audio file
                with open(wav_path, 'rb') as f:
                    processed_audio_data = f.read()

                # Store results with enhanced data
                cursor.execute("""
                    INSERT INTO threat_analysis_results (
                        audio_id, transcript, threat_type, confidence, severity, 
                        analysis, keywords, urgent, recommended_action, audio_file
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                """, (
                    record_id,
                    transcript,
                    threat_analysis["threat_type"],
                    threat_analysis["confidence"],
                    threat_analysis["severity"],
                    threat_analysis["analysis"],
                    json.dumps(threat_analysis.get("keywords", [])),
                    threat_analysis.get("urgent", False),
                    threat_analysis.get("recommended_action", ""),
                    psycopg2.Binary(processed_audio_data)
                ))

                # Mark as processed
                cursor.execute("UPDATE audio_input_table SET processed = TRUE WHERE id = %s;", (record_id,))
                conn.commit()

                logger.info(f"Successfully processed record {record_id} - Threat: {threat_analysis['threat_type']} "
                            f"(confidence: {threat_analysis['confidence']:.2f}, severity: {threat_analysis['severity']})")

            except Exception as e:
                logger.error(f"Error processing record {record_id}: {e}")
                # Mark as processed with error
                try:
                    cursor.execute("""
                        INSERT INTO threat_analysis_results (
                            audio_id, transcript, threat_type, confidence, severity, 
                            analysis, error_message
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s);
                    """, (record_id, "", "processing_error", 0.0, "low",
                          f"Processing error: {str(e)}", str(e)))

                    cursor.execute("UPDATE audio_input_table SET processed = TRUE WHERE id = %s;", (record_id,))
                    conn.commit()
                except Exception as db_error:
                    logger.error(f"Database error while handling processing error: {db_error}")
                    conn.rollback()

        logger.info("Audio processing completed")

    except Exception as e:
        logger.error(f"Critical error in process_audio_from_db: {e}")
        if conn:
            conn.rollback()

    finally:
        # Cleanup
        if conn:
            conn.close()

        if temp_dir and os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")


def test_audio_file(file_path):
    """Test function to validate a specific audio file"""
    logger.info(f"Testing audio file: {file_path}")

    if not os.path.exists(file_path):
        logger.error(f"File does not exist: {file_path}")
        return False

    # Check file size
    file_size = os.path.getsize(file_path)
    logger.info(f"File size: {file_size} bytes")

    if file_size == 0:
        logger.error("File is empty")
        return False

    # Try to validate with pydub
    if validate_audio_file(file_path):
        logger.info("Audio file is valid")
        return True
    else:
        logger.error("Audio file validation failed")
        return False


def test_gemini_classification(sample_text):
    """Test Gemini classification with sample text"""
    logger.info("Testing Gemini classification...")
    result = classify_threat_with_gemini(sample_text)
    logger.info(f"Test result: {json.dumps(result, indent=2)}")
    return result


if __name__ == '__main__':
    # Test Gemini first


    # Test specific file first
    test_file = "Recording (2).m4a"  # Change this to your file

    if os.path.exists(test_file):
        logger.info(f"Testing file: {test_file}")
        if test_audio_file(test_file):
            logger.info("File test passed, proceeding with database processing")
            process_audio_from_db()
        else:
            logger.error("File test failed, please check your audio file")
    else:
        logger.info("Test file not found, proceeding with database processing")
        process_audio_from_db()