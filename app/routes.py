from flask import Blueprint, render_template, request, flash, redirect, url_for, jsonify, send_file
from app.models import db, AudioInput, ThreatAnalysisResult
from app.utils import (validate_audio_file, convert_to_wav, translate_audio,
                       classify_threat_with_gemini, allowed_file)
from werkzeug.utils import secure_filename
import os
from datetime import datetime
import io

main = Blueprint('main', __name__)


@main.route('/')
def index():
    # Get recent analyses
    recent_results = ThreatAnalysisResult.query.order_by(
        ThreatAnalysisResult.created_at.desc()
    ).limit(5).all()

    # Add severity class for bootstrap styling
    for result in recent_results:
        result.severity_class = {
            'low': 'info',
            'medium': 'warning',
            'high': 'danger',
            'critical': 'dark'
        }.get(result.severity.lower(), 'secondary')

    return render_template('index.html', recent_results=recent_results)


@main.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'audio_file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)

        file = request.files['audio_file']

        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            try:
                # Save the file temporarily
                filename = secure_filename(file.filename)
                temp_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
                file.save(temp_path)

                # Validate the audio file
                if not validate_audio_file(temp_path):
                    flash('Invalid audio file', 'error')
                    os.remove(temp_path)
                    return redirect(request.url)

                # Convert to WAV if needed
                wav_path = convert_to_wav(temp_path)
                if not wav_path:
                    flash('Error converting audio file', 'error')
                    os.remove(temp_path)
                    return redirect(request.url)

                # Read the WAV file
                with open(wav_path, 'rb') as f:
                    audio_data = f.read()

                # Create database entry
                audio_input = AudioInput(
                    filename=filename,
                    audio_file=audio_data,
                    processed=False
                )
                db.session.add(audio_input)
                db.session.commit()

                # Process the audio
                transcript = translate_audio(wav_path)
                threat_analysis = classify_threat_with_gemini(transcript)

                # Create result entry
                result = ThreatAnalysisResult(
                    audio_id=audio_input.id,
                    transcript=transcript,
                    threat_type=threat_analysis['threat_type'],
                    confidence=threat_analysis['confidence'],
                    severity=threat_analysis['severity'],
                    analysis=threat_analysis['analysis'],
                    keywords=threat_analysis.get('keywords', []),
                    urgent=threat_analysis.get('urgent', False),
                    recommended_action=threat_analysis.get('recommended_action', ''),
                    audio_file=audio_data
                )
                db.session.add(result)

                # Mark as processed
                audio_input.processed = True
                db.session.commit()

                # Cleanup
                os.remove(temp_path)
                if wav_path != temp_path:
                    os.remove(wav_path)

                return redirect(url_for('main.view_result', result_id=result.id))

            except Exception as e:
                flash(f'Error processing file: {str(e)}', 'error')
                return redirect(request.url)
        else:
            flash('Invalid file type', 'error')
            return redirect(request.url)

    return render_template('upload.html')


@main.route('/results/<int:result_id>')
def view_result(result_id):
    result = ThreatAnalysisResult.query.get_or_404(result_id)

    # Add severity class for bootstrap styling
    result.severity_class = {
        'low': 'info',
        'medium': 'warning',
        'high': 'danger',
        'critical': 'dark'
    }.get(result.severity.lower(), 'secondary')

    return render_template('results.html', result=result)


@main.route('/history')
def history():
    page = request.args.get('page', 1, type=int)
    per_page = 10

    pagination = ThreatAnalysisResult.query.order_by(
        ThreatAnalysisResult.created_at.desc()
    ).paginate(page=page, per_page=per_page)

    results = pagination.items

    # Add severity class for bootstrap styling
    for result in results:
        result.severity_class = {
            'low': 'info',
            'medium': 'warning',
            'high': 'danger',
            'critical': 'dark'
        }.get(result.severity.lower(), 'secondary')

    return render_template(
        'history.html',
        results=results,
        page=page,
        total_pages=pagination.pages
    )


@main.route('/download/<int:result_id>')
def download_audio(result_id):
    result = ThreatAnalysisResult.query.get_or_404(result_id)

    return send_file(
        io.BytesIO(result.audio_file),
        mimetype='audio/wav',
        as_attachment=True,
        download_name=f"{result.audio_input.filename}.wav"
    )


@main.route('/api/status/<int:audio_id>')
def check_status(audio_id):
    audio = AudioInput.query.get_or_404(audio_id)
    return jsonify({
        'processed': audio.processed,
        'result_id': audio.results[0].id if audio.processed and audio.results else None
    })