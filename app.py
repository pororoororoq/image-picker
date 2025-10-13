# At the top of the file, update imports
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json
import shutil
import zipfile
from pathlib import Path
from datetime import datetime
import threading
import uuid
import gc
import requests
import base64
from PIL import Image
from io import BytesIO
import random
import traceback
import tempfile
import time

def update_progress(job_dir, processed, total):
    """Write incremental progress updates to status.json."""
    import json, os
    status = {
        "status": "processing",
        "progress": round((processed / total) * 100, 2),
        "processed_files": processed,
        "total_files": total
    }
    with open(os.path.join(job_dir, "status.json"), "w") as f:
        json.dump(status, f)

# Try to import gradio_client (but we won't actually use it on Render)
try:
    from gradio_client import Client, handle_file
    GRADIO_CLIENT_AVAILABLE = True
    print("✓ gradio_client is available (but won't be used on Render due to Cloudflare issues)")
except ImportError:
    GRADIO_CLIENT_AVAILABLE = False
    print("⚠ gradio_client not available (not needed - using HTTP API)")

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev-key-change-in-production')

# Configure CORS
CORS(app, 
     origins="*",
     allow_headers="*",
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     supports_credentials=False,
     resources={r"/*": {"origins": "*"}})

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'tif'}

# Create necessary folders
for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULTS_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# Global processing queue
processing_jobs = {}

# HuggingFace Space URL - CORRECT ENDPOINT!
HF_SPACE_URL = 'https://pororoororoq-photo-analyzer.hf.space'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class AnalysisJob:
    def __init__(self, job_id, total_files):
        self.job_id = job_id
        self.total_files = total_files
        self.processed_files = 0
        self.status = 'processing'
        self.results = {}
        self.error = None
        self.current_file = None
        
    def to_dict(self):
        return {
            'job_id': self.job_id,
            'total_files': self.total_files,
            'processed_files': self.processed_files,
            'status': self.status,
            'progress': (self.processed_files / self.total_files * 100) if self.total_files > 0 else 0,
            'current_file': self.current_file,
            'error': self.error
        }

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({'status': 'ok'})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response

# @app.after_request
# def after_request(response):
#     response.headers.add('Access-Control-Allow-Origin', '*')
#     response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
#     response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
#     return response

@app.route('/')
def index():
    """API root endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'Yearbook Photo Analyzer API',
        'version': '4.0.0',
        'mode': 'Direct HuggingFace Integration',
        'hf_space': HF_SPACE_URL,
        'endpoints': {
            'upload': '/upload',
            'status': '/status/<job_id>',
            'results': '/results/<job_id>',
            'image': '/image/<job_id>/<filename>',
            'download': '/download_selection/<job_id>',
            'export': '/export_report/<job_id>'
        }
    })

UPLOAD_DIR = "uploads"

@app.route("/upload", methods=["POST"])
def upload_files():
    """
    Accepts chunked uploads.
    Each request may contain a subset of files belonging to the same job_id.
    The first request (no job_id) creates a new job folder.
    Subsequent requests append more files to that folder.
    """
    try:
        # job_id passed as query param for subsequent chunks
        job_id = request.args.get("job_id")
        is_new_job = False
        if not job_id:
            job_id = str(uuid.uuid4())
            is_new_job = True

        job_dir = os.path.join(UPLOAD_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)

        # get files from the current chunk
        uploaded_files = request.files.getlist("files[]")
        saved_files = []

        for i, file in enumerate(uploaded_files):
            filename = secure_filename(file.filename)
            save_path = os.path.join(job_dir, filename)
            file.save(save_path)
            saved_files.append(filename)
            update_progress(job_dir, i + 1, len(filename))

        # if this is the first chunk, you can initialize metadata
        if is_new_job:
            # create a simple status tracker file
            with open(os.path.join(job_dir, "meta.txt"), "w") as f:
                f.write("initialized")

        # ✅ Return same job_id for all chunks
        return jsonify({
            "job_id": job_id,
            "saved": len(saved_files),
            "message": f"Received {len(saved_files)} files for job {job_id}"
        }), 200

    except Exception as e:
        print("Upload error:", e)
        return jsonify({"error": str(e)}), 500

import json
import os
from flask import jsonify

@app.route("/status/<job_id>", methods=["GET"])
def check_status(job_id):
    """
    Returns the progress of the given job.
    Reads from uploads/<job_id>/status.json if it exists.
    """
    job_dir = os.path.join("uploads", job_id)
    status_file = os.path.join(job_dir, "status.json")

    # Check if the job directory exists
    if not os.path.exists(job_dir):
        return jsonify({"status": "error", "error": "Job not found"}), 404

    # If a status file exists, return its contents
    if os.path.exists(status_file):
        try:
            with open(status_file, "r") as f:
                data = json.load(f)
            return jsonify(data)
        except Exception as e:
            print("Error reading status.json:", e)
            return jsonify({"status": "error", "error": "Failed to read progress"}), 500

    # Default (no file yet)
    return jsonify({
        "status": "processing",
        "progress": 0,
        "processed_files": 0,
        "total_files": 0
    })

def call_huggingface_api(image_path):
    """Call HuggingFace Space API using gradio_client with file() wrapper"""
    try:
        from gradio_client import Client, handle_file
        
        print(f"  📡 Calling HuggingFace using gradio_client...")

        # Warm-up + retry to avoid cold-start JSON decode errors
        import time, requests, os
        SPACE_URL = HF_SPACE_URL  # use full URL
        last_err = None
        for _try in range(3):
            try:
                try:
                    requests.get(SPACE_URL, timeout=10)  # wake Space
                except Exception as _e:
                    pass
                client = Client(SPACE_URL, hf_token=os.environ.get('HF_TOKEN'))
                _ = client.view_api(return_format='dict')
                break
            except Exception as e:
                last_err = e
                time.sleep(2)
        else:
            raise last_err if last_err else RuntimeError('Failed to initialize gradio Client')
        
        # Create client
        # client initialized with SPACE_URL above (with retries)
        
        # Call the API using file() wrapper - THIS IS THE KEY!
        result = client.predict(
            handle_file(image_path),   # IMPORTANT: wrap with file()
            True,               # enhance_option
            api_name="/predict"
        )
        
        print(f"  ✅ HF response received!")
        print(f"  Response type: {type(result)}")
        
        if isinstance(result, dict):
            print(f"  Response keys: {list(result.keys())[:10]}")
            return result
        elif isinstance(result, str):
            try:
                parsed = json.loads(result)
                print(f"  Parsed string response to dict")
                return parsed
            except:
                print(f"  Could not parse string response")
                return None
        else:
            print(f"  Unexpected response type: {type(result)}")
            return result
            
    except ImportError:
        print(f"  ❌ gradio_client not installed! Installing it...")
        # Fallback to HTTP API if gradio_client not available
        return call_huggingface_api_http_fallback(image_path)
    except Exception as e:
        print(f"  ❌ Error calling HuggingFace: {e}")
        import traceback
        traceback.print_exc()
        return None

def call_huggingface_api_http_fallback(image_path):
    """Fallback HTTP API call if gradio_client fails"""
    try:
        print(f"  📡 Trying HTTP API fallback...")
        
        # Load and prepare image
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            
            # Resize if too large
            max_size = 1024
            if img.width > max_size or img.height > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffered = BytesIO()
            img.save(buffered, format="PNG", optimize=True)
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        api_url = "https://pororoororoq-photo-analyzer.hf.space/api/predict"
        
        response = requests.post(
            api_url,
            json={
                "data": [
                    f"data:image/png;base64,{img_base64}",
                    True
                ]
            },
            headers={"Content-Type": "application/json"},
            timeout=45
        )
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, dict) and 'data' in result:
                return result['data'][0] if result['data'] else {}
            return result
        
        return None
        
    except Exception as e:
        print(f"  HTTP fallback also failed: {e}")
        return None

def analyze_photos_background(job_id, folder_path):
    """Analyze photos using direct HuggingFace API"""
    job = processing_jobs.get(job_id)
    if not job:
        return
    
    try:
        print(f"\n{'='*60}")
        print(f"Starting analysis for job {job_id}")
        print(f"HuggingFace Space: {HF_SPACE_URL}")
        print(f"Using endpoint: {HF_SPACE_URL}/api/predict")
        print(f"{'='*60}")
        
        # Wake up HuggingFace Space if it's sleeping
        print("Checking if HuggingFace Space is awake...")
        try:
            wake_response = requests.get(HF_SPACE_URL, timeout=10)
            if wake_response.status_code == 200:
                print("✓ HuggingFace Space is responding")
                # Wait a bit for it to fully wake up
                import time
                time.sleep(2)
            else:
                print(f"⚠ HuggingFace Space returned {wake_response.status_code}")
        except Exception as e:
            print(f"⚠ Could not reach HuggingFace Space: {e}")
        
        # Get list of images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        files = [f for f in os.listdir(folder_path) 
                if os.path.splitext(f)[1].lower() in image_extensions]
        
        total_files = len(files)
        job.total_files = total_files
        job.status = 'processing'
        
        results = {}
        
        # Process each file
        for i, filename in enumerate(files):
            filepath = os.path.join(folder_path, filename)
            print(f"\n[{i+1}/{total_files}] Processing: {filename}")
            
            # Update job progress
            job.processed_files = i
            job.current_file = filename
            
            try:
                # Call HuggingFace API
                hf_result = call_huggingface_api(filepath)
                
                if hf_result and isinstance(hf_result, dict):
                    # Extract scores from HuggingFace response
                    scores = hf_result.get('scores', {})
                    analysis = hf_result.get('analysis', {})
                    
                    # Check if scores are at root level
                    if not scores:
                        if 'aesthetic_score' in hf_result:
                            scores = {
                                'aesthetic_score': hf_result.get('aesthetic_score', 5.0),
                                'blur_score': hf_result.get('blur_score', 100),
                                'composition_score': hf_result.get('composition_score', 5.0),
                                'combined_score': hf_result.get('combined_score', 5.0)
                            }
                        if 'blur_category' in hf_result:
                            analysis = {
                                'blur_category': hf_result.get('blur_category', 'unknown'),
                                'aesthetic_rating': hf_result.get('aesthetic_rating', 'fair'),
                                'recommendation': hf_result.get('recommendation', 'maybe'),
                                'action': hf_result.get('action', ''),
                                'face_detected': hf_result.get('face_detected', False)
                            }
                    
                    if scores:  # We got valid scores from HuggingFace
                        results[filepath] = {
                            'filename': filename,
                            'aesthetic_score': float(scores.get('aesthetic_score', 5.0)),
                            'blur_score': float(scores.get('blur_score', 100)),
                            'blur_category': analysis.get('blur_category', 'unknown'),
                            'composition_score': float(scores.get('composition_score', 5.0)),
                            'combined_score': float(scores.get('combined_score', 5.0)),
                            'aesthetic_rating': analysis.get('aesthetic_rating', 'fair'),
                            'recommendation': analysis.get('recommendation', 'maybe'),
                            'action': analysis.get('action', ''),
                            'ml_source': 'huggingface',
                            'face_detected': analysis.get('face_detected', False)
                        }
                        
                        print(f"  ✓ HF scores - A:{results[filepath]['aesthetic_score']:.1f}, "
                              f"B:{results[filepath]['blur_score']:.0f}, "
                              f"C:{results[filepath]['composition_score']:.1f}")
                    else:
                        # Fallback if HuggingFace didn't return proper scores
                        results[filepath] = simple_fallback_analysis(filepath, filename)
                        print(f"  → Using fallback (no valid HF scores)")
                else:
                    # Use fallback if HuggingFace failed
                    results[filepath] = simple_fallback_analysis(filepath, filename)
                    print(f"  → Using fallback (HF call failed)")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                results[filepath] = {
                    'filename': filename,
                    'blur_score': 100,
                    'blur_category': 'unknown',
                    'aesthetic_score': 5,
                    'aesthetic_rating': 'error',
                    'composition_score': 5,
                    'combined_score': 5,
                    'recommendation': 'skip',
                    'action': f'Error: {str(e)}',
                    'ml_source': 'error'
                }
            
            # Clean up memory
            gc.collect()
        
        # Update final status
        job.processed_files = total_files
        job.results = results
        job.status = 'completed'
        
        # Save results to file
        results_file = os.path.join(app.config['RESULTS_FOLDER'], f"{job_id}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Job {job_id} completed!")
        
        # Print summary
        hf_count = sum(1 for r in results.values() if r.get('ml_source') == 'huggingface')
        fallback_count = sum(1 for r in results.values() if 'fallback' in r.get('ml_source', ''))
        print(f"Summary: {hf_count} via HuggingFace, {fallback_count} via fallback")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"Fatal error in job {job_id}: {e}")
        traceback.print_exc()
        job.status = 'error'
        job.error = str(e)

def simple_fallback_analysis(filepath, filename):
    """Simple image analysis without ML - used when HF fails"""
    try:
        with Image.open(filepath) as img:
            width, height = img.size
            megapixels = (width * height) / 1_000_000
            
            # Basic scoring based on resolution
            aesthetic_score = 5.0 + min(megapixels / 2, 2.5)
            
            # Add some randomness for variety
            import hashlib
            file_hash = hashlib.md5(filename.encode()).hexdigest()
            hash_value = int(file_hash[:4], 16)
            variety = (hash_value % 20 - 10) / 10.0
            aesthetic_score = max(1, min(10, aesthetic_score + variety))
            
            # Simple blur estimation based on file size
            file_size = os.path.getsize(filepath) / 1024  # KB
            size_per_megapixel = file_size / megapixels if megapixels > 0 else 0
            
            if size_per_megapixel > 150:
                blur_score = 180  # Good quality
                blur_category = 'sharp'
            elif size_per_megapixel > 80:
                blur_score = 100  # Medium quality
                blur_category = 'slightly_blurry'
            else:
                blur_score = 50  # Low quality
                blur_category = 'blurry'
            
            composition_score = 5.0 + (hash_value % 30 - 15) / 5.0
            composition_score = max(1, min(10, composition_score))
            
            combined_score = (aesthetic_score * 0.4 + (blur_score/20) * 0.3 + composition_score * 0.3)
            
            return {
                'filename': filename,
                'aesthetic_score': round(aesthetic_score, 2),
                'blur_score': round(blur_score, 2),
                'blur_category': blur_category,
                'composition_score': round(composition_score, 2),
                'combined_score': round(combined_score, 2),
                'aesthetic_rating': 'good' if aesthetic_score >= 7 else 'fair' if aesthetic_score >= 5 else 'poor',
                'recommendation': 'use' if combined_score >= 7 else 'maybe' if combined_score >= 5 else 'skip',
                'action': 'Fallback analysis - manual review recommended',
                'ml_source': 'fallback',
                'face_detected': False
            }
            
    except Exception as e:
        print(f"  Error in fallback: {e}")
        return {
            'filename': filename,
            'aesthetic_score': 5.0,
            'blur_score': 100,
            'blur_category': 'unknown',
            'composition_score': 5.0,
            'combined_score': 5.0,
            'aesthetic_rating': 'error',
            'recommendation': 'skip',
            'action': 'Error processing',
            'ml_source': 'error',
            'face_detected': False
        }

@app.route('/status/<job_id>')
def get_job_status(job_id):
    """Get the status of an analysis job"""
    job = processing_jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(job.to_dict())

@app.route('/results/<job_id>')
def get_results(job_id):
    """Get analysis results for a job"""
    job = processing_jobs.get(job_id)
    
    if not job:
        results_file = os.path.join(app.config['RESULTS_FOLDER'], f"{job_id}.json")
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
                
            processed_results = []
            for filepath, data in results.items():
                processed_results.append({
                    'filename': os.path.basename(filepath),
                    'filepath': filepath,
                    **data,
                    'job_id': job_id
                })
            
            stats = {
                'total': len(processed_results),
                'use': sum(1 for r in processed_results if r.get('recommendation') == 'use'),
                'enhance': sum(1 for r in processed_results if r.get('recommendation') == 'enhance'),
                'maybe': sum(1 for r in processed_results if r.get('recommendation') == 'maybe'),
                'skip': sum(1 for r in processed_results if r.get('recommendation') == 'skip'),
            }
            
            return jsonify({
                'results': processed_results,
                'stats': stats,
                'job_id': job_id
            })
        
        return jsonify({'error': 'Job not found'}), 404
    
    if job.status != 'completed':
        return jsonify({'error': 'Analysis not yet completed', 'status': job.status}), 202
    
    processed_results = []
    for filepath, data in job.results.items():
        filename = os.path.basename(filepath)
        processed_results.append({
            'filename': filename,
            'filepath': filepath,
            **data,
            'job_id': job_id
        })
    
    processed_results.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
    
    stats = {
        'total': len(processed_results),
        'use': sum(1 for r in processed_results if r.get('recommendation') == 'use'),
        'enhance': sum(1 for r in processed_results if r.get('recommendation') == 'enhance'),
        'maybe': sum(1 for r in processed_results if r.get('recommendation') == 'maybe'),
        'skip': sum(1 for r in processed_results if r.get('recommendation') == 'skip'),
    }
    
    return jsonify({
        'results': processed_results,
        'stats': stats,
        'job_id': job_id
    })

@app.route('/image/<job_id>/<filename>')
def serve_image(job_id, filename):
    """Serve an uploaded image"""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], job_id, filename)
    if os.path.exists(filepath):
        return send_file(filepath)
    return jsonify({'error': 'Image not found'}), 404

@app.route('/download_selection/<job_id>', methods=['POST'])
def download_selection(job_id):
    """Download selected images as a zip file"""
    data = request.json
    selected_files = data.get('selected', [])
    
    if not selected_files:
        return jsonify({'error': 'No files selected'}), 400
    
    zip_filename = f"yearbook_selection_{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    zip_path = os.path.join(app.config['RESULTS_FOLDER'], zip_filename)
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for filename in selected_files:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], job_id, filename)
            if os.path.exists(filepath):
                zipf.write(filepath, filename)
    
    return send_file(zip_path, as_attachment=True, download_name=zip_filename)

@app.route('/export_report/<job_id>')
def export_report(job_id):
    """Export analysis report as JSON"""
    results_file = os.path.join(app.config['RESULTS_FOLDER'], f"{job_id}.json")
    if os.path.exists(results_file):
        return send_file(results_file, as_attachment=True, 
                        download_name=f"analysis_report_{job_id}.json")
    return jsonify({'error': 'Report not found'}), 404

@app.route('/cleanup/<job_id>', methods=['DELETE'])
def cleanup_job(job_id):
    """Clean up files for a completed job"""
    upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
    if os.path.exists(upload_folder):
        shutil.rmtree(upload_folder)
    
    if job_id in processing_jobs:
        del processing_jobs[job_id]
    
    gc.collect()
    
    return jsonify({'message': 'Cleanup successful'})

@app.route('/test_huggingface')
def test_huggingface():
    """Test HuggingFace connection with a small test image"""
    import numpy as np
    
    try:
        # Create a small test image
        test_image = Image.new('RGB', (100, 100), color='red')
        
        # Save to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            test_image.save(tmp.name)
            tmp_path = tmp.name
        
        # Test the API
        result = call_huggingface_api(tmp_path)
        
        # Clean up
        os.unlink(tmp_path)
        
        return jsonify({
            'status': 'success' if result else 'failed',
            'result': result,
            'hf_url': HF_SPACE_URL
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'hf_url': HF_SPACE_URL
        })

@app.route('/health')
def health():
    """Health check endpoint"""
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'memory_usage_mb': round(memory_mb, 2),
        'hf_space_url': HF_SPACE_URL
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)