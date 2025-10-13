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
import hashlib

# Try to import gradio_client
try:
    from gradio_client import Client, handle_file
    GRADIO_CLIENT_AVAILABLE = True
    print("✔ gradio_client is available")
except ImportError:
    GRADIO_CLIENT_AVAILABLE = False
    print("⚠ gradio_client not available (using HTTP fallback)")

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

# HuggingFace Space URL
HF_SPACE_URL = 'https://pororoororoq-photo-analyzer.hf.space'

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def update_progress(job_dir, processed, total, phase='analysis'):
    """Write progress updates to status.json"""
    progress = round((processed / total) * 100, 2) if total > 0 else 0
    status = {
        "status": "processing",
        "phase": phase,
        "progress": progress,
        "processed_files": processed,
        "total_files": total,
        "timestamp": datetime.now().isoformat()
    }
    
    status_file = os.path.join(job_dir, "status.json")
    with open(status_file, "w") as f:
        json.dump(status, f)

class AnalysisJob:
    """Job tracking class for analysis progress"""
    def __init__(self, job_id, total_files):
        self.job_id = job_id
        self.total_files = total_files
        self.processed_files = 0
        self.status = 'processing'
        self.phase = 'analysis'
        self.results = {}
        self.error = None
        self.current_file = None
        
    def to_dict(self):
        return {
            'job_id': self.job_id,
            'total_files': self.total_files,
            'processed_files': self.processed_files,
            'status': self.status,
            'phase': self.phase,
            'progress': (self.processed_files / self.total_files * 100) if self.total_files > 0 else 0,
            'current_file': self.current_file,
            'error': self.error
        }

@app.before_request
def handle_preflight():
    """Handle CORS preflight requests"""
    if request.method == "OPTIONS":
        response = jsonify({'status': 'ok'})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response

@app.after_request
def after_request(response):
    """Add CORS headers to all responses"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/')
def index():
    """API root endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'Yearbook Photo Analyzer API',
        'version': '5.0.0',
        'mode': 'Dual Progress System',
        'hf_space': HF_SPACE_URL,
        'endpoints': {
            'upload': '/upload',
            'start_analysis': '/start_analysis/<job_id>',
            'status': '/status/<job_id>',
            'results': '/results/<job_id>',
            'image': '/image/<job_id>/<filename>',
            'download': '/download_selection/<job_id>',
            'export': '/export_report/<job_id>',
            'health': '/health'
        }
    })

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle chunked file uploads with automatic analysis trigger"""
    try:
        # Get parameters
        job_id = request.args.get('job_id')
        total_files = request.args.get('total_files')
        
        # Create new job if needed
        if not job_id:
            job_id = str(uuid.uuid4())
            print(f"Created new job: {job_id}")
        
        # Setup job directory
        job_dir = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
        os.makedirs(job_dir, exist_ok=True)
        
        # Get uploaded files
        uploaded_files = request.files.getlist('files[]')
        if not uploaded_files:
            return jsonify({'error': 'No files provided'}), 400
        
        # Save files
        saved_files = []
        for file in uploaded_files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                save_path = os.path.join(job_dir, filename)
                file.save(save_path)
                saved_files.append(filename)
                print(f"  Saved: {filename}")
        
        # Count total files in job directory
        all_files = [f for f in os.listdir(job_dir) 
                    if os.path.isfile(os.path.join(job_dir, f)) 
                    and allowed_file(f)]
        
        print(f"Job {job_id}: {len(all_files)} total files, expecting {total_files}")
        
        # Check if we should start analysis
        analysis_started = False
        if total_files and len(all_files) >= int(total_files):
            # All files uploaded, start analysis
            print(f"Starting analysis for job {job_id} with {len(all_files)} files")
            
            # Create job and start analysis
            job = AnalysisJob(job_id, len(all_files))
            processing_jobs[job_id] = job
            
            # Start analysis in background
            thread = threading.Thread(
                target=analyze_photos_background,
                args=(job_id, job_dir)
            )
            thread.daemon = True
            thread.start()
            
            analysis_started = True
            
            return jsonify({
                'job_id': job_id,
                'saved': len(saved_files),
                'total_saved': len(all_files),
                'message': f'Upload complete. Analysis started for {len(all_files)} files.',
                'analysis_started': True
            }), 200
        
        # Not all files uploaded yet
        return jsonify({
            'job_id': job_id,
            'saved': len(saved_files),
            'total_saved': len(all_files),
            'message': f'Received {len(saved_files)} files for job {job_id}',
            'analysis_started': False
        }), 200
        
    except Exception as e:
        print(f"Upload error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/start_analysis/<job_id>', methods=['POST'])
def start_analysis(job_id):
    """Manually start analysis for uploaded files"""
    try:
        job_dir = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
        
        if not os.path.exists(job_dir):
            return jsonify({'error': 'Job not found'}), 404
        
        # Check if already processing
        if job_id in processing_jobs:
            job = processing_jobs[job_id]
            if job.status == 'processing':
                return jsonify({'message': 'Analysis already in progress'}), 200
        
        # Get all valid image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        files = [f for f in os.listdir(job_dir) 
                if os.path.splitext(f)[1].lower() in image_extensions]
        
        if not files:
            return jsonify({'error': 'No valid image files found'}), 400
        
        # Create job and start analysis
        job = AnalysisJob(job_id, len(files))
        processing_jobs[job_id] = job
        
        # Start analysis in background
        thread = threading.Thread(
            target=analyze_photos_background,
            args=(job_id, job_dir)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'message': f'Analysis started for {len(files)} images',
            'job_id': job_id,
            'total_files': len(files)
        }), 200
        
    except Exception as e:
        print(f"Error starting analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status/<job_id>')
def get_job_status(job_id):
    """Get current status of a job"""
    # Check in-memory job first
    job = processing_jobs.get(job_id)
    if job:
        return jsonify(job.to_dict())
    
    # Check status file
    job_dir = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
    status_file = os.path.join(job_dir, 'status.json')
    
    if not os.path.exists(job_dir):
        return jsonify({'error': 'Job not found'}), 404
    
    if os.path.exists(status_file):
        try:
            with open(status_file, 'r') as f:
                data = json.load(f)
            return jsonify(data)
        except Exception as e:
            print(f"Error reading status file: {e}")
    
    # Default status
    return jsonify({
        'status': 'uploading',
        'phase': 'upload',
        'progress': 0,
        'processed_files': 0,
        'total_files': 0
    })

def call_huggingface_api(image_path):
    """Call HuggingFace Space API using gradio_client or HTTP fallback"""
    if GRADIO_CLIENT_AVAILABLE:
        return call_huggingface_gradio(image_path)
    else:
        return call_huggingface_http(image_path)

def call_huggingface_gradio(image_path):
    """Call HuggingFace using gradio_client"""
    try:
        print(f"  📡 Calling HuggingFace using gradio_client...")
        
        # Initialize client with retries
        client = None
        for attempt in range(3):
            try:
                client = Client(HF_SPACE_URL, hf_token=os.environ.get('HF_TOKEN'))
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2)
                else:
                    raise e
        
        if not client:
            return None
        
        # Call the API
        result = client.predict(
            handle_file(image_path),
            True,  # enhance_option
            api_name="/predict"
        )
        
        print(f"  ✅ HF response received!")
        
        # Parse response
        if isinstance(result, dict):
            return result
        elif isinstance(result, str):
            try:
                return json.loads(result)
            except:
                return None
        
        return result
        
    except Exception as e:
        print(f"  ❌ Gradio client error: {e}")
        # Fallback to HTTP
        return call_huggingface_http(image_path)

def call_huggingface_http(image_path):
    """Call HuggingFace using direct HTTP API"""
    try:
        print(f"  📡 Calling HuggingFace using HTTP API...")
        
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
        
        # Call API
        api_url = f"{HF_SPACE_URL}/api/predict"
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
        
        print(f"  ❌ HTTP API returned {response.status_code}")
        return None
        
    except Exception as e:
        print(f"  ❌ HTTP fallback error: {e}")
        return None

def simple_fallback_analysis(filepath, filename):
    """Fallback analysis when ML is unavailable"""
    try:
        with Image.open(filepath) as img:
            width, height = img.size
            megapixels = (width * height) / 1_000_000
            
            # Basic scoring
            aesthetic_score = 5.0 + min(megapixels / 2, 2.5)
            
            # Add variety
            file_hash = hashlib.md5(filename.encode()).hexdigest()
            hash_value = int(file_hash[:4], 16)
            variety = (hash_value % 20 - 10) / 10.0
            aesthetic_score = max(1, min(10, aesthetic_score + variety))
            
            # Blur estimation
            file_size = os.path.getsize(filepath) / 1024  # KB
            size_per_megapixel = file_size / megapixels if megapixels > 0 else 0
            
            if size_per_megapixel > 150:
                blur_score = 180
                blur_category = 'sharp'
            elif size_per_megapixel > 80:
                blur_score = 100
                blur_category = 'slightly_blurry'
            else:
                blur_score = 50
                blur_category = 'blurry'
            
            # Composition score
            composition_score = 5.0 + (hash_value % 30 - 15) / 5.0
            composition_score = max(1, min(10, composition_score))
            
            # Combined score
            combined_score = (aesthetic_score * 0.4 + 
                            (blur_score/20) * 0.3 + 
                            composition_score * 0.3)
            
            # Recommendation
            if combined_score >= 7:
                recommendation = 'use'
                action = 'Ready to use (fallback analysis)'
            elif combined_score >= 5:
                recommendation = 'maybe'
                action = 'Manual review recommended (fallback)'
            else:
                recommendation = 'skip'
                action = 'Below threshold (fallback)'
            
            return {
                'filename': filename,
                'aesthetic_score': round(aesthetic_score, 2),
                'blur_score': round(blur_score, 2),
                'blur_category': blur_category,
                'composition_score': round(composition_score, 2),
                'combined_score': round(combined_score, 2),
                'aesthetic_rating': 'good' if aesthetic_score >= 7 else 'fair',
                'recommendation': recommendation,
                'action': action,
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

def analyze_photos_background(job_id, folder_path):
    """Background thread to analyze photos"""
    job = processing_jobs.get(job_id)
    if not job:
        job = AnalysisJob(job_id, 0)
        processing_jobs[job_id] = job
    
    try:
        print(f"\n{'='*60}")
        print(f"Starting analysis for job {job_id}")
        print(f"HuggingFace Space: {HF_SPACE_URL}")
        print(f"{'='*60}")
        
        # Initialize progress
        update_progress(folder_path, 0, 1, 'analysis')
        
        # Wake up HuggingFace Space
        print("Checking HuggingFace Space...")
        try:
            wake_response = requests.get(HF_SPACE_URL, timeout=10)
            if wake_response.status_code == 200:
                print("✔ HuggingFace Space is responding")
                time.sleep(2)  # Give it time to fully wake
        except Exception as e:
            print(f"⚠ Could not reach HuggingFace Space: {e}")
        
        # Get image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        files = [f for f in os.listdir(folder_path) 
                if os.path.splitext(f)[1].lower() in image_extensions]
        
        total_files = len(files)
        job.total_files = total_files
        job.status = 'processing'
        job.phase = 'analysis'
        
        # Update initial progress
        update_progress(folder_path, 0, total_files, 'analysis')
        
        results = {}
        
        # Process each file
        for i, filename in enumerate(files):
            filepath = os.path.join(folder_path, filename)
            print(f"\n[{i+1}/{total_files}] Processing: {filename}")
            
            # Update job state
            job.processed_files = i
            job.current_file = filename
            
            # Update progress before processing
            update_progress(folder_path, i, total_files, 'analysis')
            
            try:
                # Try HuggingFace API
                hf_result = call_huggingface_api(filepath)
                
                if hf_result and isinstance(hf_result, dict):
                    # Extract scores from response
                    scores = hf_result.get('scores', {})
                    analysis = hf_result.get('analysis', {})
                    
                    # Handle different response formats
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
                    
                    if scores:
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
                        
                        print(f"  ✔ HF scores - A:{results[filepath]['aesthetic_score']:.1f}, "
                              f"B:{results[filepath]['blur_score']:.0f}, "
                              f"C:{results[filepath]['composition_score']:.1f}")
                    else:
                        results[filepath] = simple_fallback_analysis(filepath, filename)
                        print("  → Using fallback (no valid HF scores)")
                else:
                    results[filepath] = simple_fallback_analysis(filepath, filename)
                    print("  → Using fallback (HF call failed)")
                    
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
                    'ml_source': 'error',
                    'face_detected': False
                }
            
            # Update progress after processing
            update_progress(folder_path, i + 1, total_files, 'analysis')
            
            # Free memory
            gc.collect()
        
        # Mark as completed
        job.processed_files = total_files
        job.results = results
        job.status = 'completed'
        job.phase = 'completed'
        
        # Write final status
        with open(os.path.join(folder_path, "status.json"), "w") as f:
            json.dump({
                "status": "completed",
                "phase": "completed",
                "progress": 100,
                "processed_files": total_files,
                "total_files": total_files,
                "timestamp": datetime.now().isoformat()
            }, f)
        
        # Save results
        results_file = os.path.join(app.config['RESULTS_FOLDER'], f"{job_id}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Job {job_id} completed successfully!")
        hf_count = sum(1 for r in results.values() if r.get('ml_source') == 'huggingface')
        fallback_count = sum(1 for r in results.values() if r.get('ml_source') == 'fallback')
        print(f"Summary: {hf_count} via HuggingFace, {fallback_count} via fallback")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"Fatal error in job {job_id}: {e}")
        traceback.print_exc()
        
        job.status = 'error'
        job.phase = 'error'
        job.error = str(e)
        
        # Write error status
        with open(os.path.join(folder_path, "status.json"), "w") as f:
            json.dump({
                "status": "error",
                "phase": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }, f)

@app.route('/results/<job_id>')
def get_results(job_id):
    """Get analysis results for a job"""
    # Check in-memory job first
    job = processing_jobs.get(job_id)
    
    # Load from file if not in memory
    if not job or job.status != 'completed':
        results_file = os.path.join(app.config['RESULTS_FOLDER'], f"{job_id}.json")
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
        else:
            return jsonify({'error': 'Results not found'}), 404
    else:
        results = job.results
    
    # Process results for response
    processed_results = []
    for filepath, data in results.items():
        processed_results.append({
            'filename': os.path.basename(filepath),
            'filepath': filepath,
            **data,
            'job_id': job_id
        })
    
    # Sort by combined score
    processed_results.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
    
    # Calculate statistics
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
    
    # Create zip file
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
    try:
        # Remove upload folder
        upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
        if os.path.exists(upload_folder):
            shutil.rmtree(upload_folder)
        
        # Remove from memory
        if job_id in processing_jobs:
            del processing_jobs[job_id]
        
        # Force garbage collection
        gc.collect()
        
        return jsonify({'message': 'Cleanup successful'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
    except:
        memory_mb = 0
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'memory_usage_mb': round(memory_mb, 2),
        'hf_space_url': HF_SPACE_URL,
        'gradio_client': GRADIO_CLIENT_AVAILABLE,
        'active_jobs': len(processing_jobs)
    })

@app.route('/test_huggingface')
def test_huggingface():
    """Test HuggingFace connection"""
    try:
        # Create test image
        test_image = Image.new('RGB', (100, 100), color='blue')
        
        # Save temporarily
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            test_image.save(tmp.name)
            tmp_path = tmp.name
        
        # Test API
        result = call_huggingface_api(tmp_path)
        
        # Clean up
        os.unlink(tmp_path)
        
        return jsonify({
            'status': 'success' if result else 'failed',
            'result': result,
            'hf_url': HF_SPACE_URL,
            'gradio_client_available': GRADIO_CLIENT_AVAILABLE
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'hf_url': HF_SPACE_URL
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)