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
        start_analysis_now = request.args.get('start_analysis') == 'true'
        
        # Create new job if needed
        if not job_id:
            job_id = str(uuid.uuid4())
            print(f"Created new job: {job_id}")
        
        # Setup job directory
        job_dir = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
        os.makedirs(job_dir, exist_ok=True)
        print(f"Job directory: {job_dir}")
        
        # Get uploaded files
        uploaded_files = request.files.getlist('files[]')
        if not uploaded_files:
            return jsonify({'error': 'No files provided'}), 400
        
        print(f"Received {len(uploaded_files)} files in this chunk")
        
        # Save files
        saved_files = []
        for file in uploaded_files:
            if file and allowed_file(file.filename):
                # Ensure we keep the original extension
                original_filename = file.filename
                filename = secure_filename(original_filename)
                
                # If secure_filename removed the extension, add it back
                if '.' not in filename:
                    original_ext = os.path.splitext(original_filename)[1]
                    if original_ext:
                        filename = filename + original_ext
                
                save_path = os.path.join(job_dir, filename)
                file.save(save_path)
                saved_files.append(filename)
                print(f"  Saved: {filename} -> {save_path}")
            else:
                print(f"  Skipped invalid file: {file.filename}")
        
        # Count total files in job directory
        all_files = []
        for f in os.listdir(job_dir):
            full_path = os.path.join(job_dir, f)
            if os.path.isfile(full_path) and allowed_file(f):
                all_files.append(f)
        
        print(f"Job {job_id}: {len(all_files)} total files saved, expecting {total_files}")
        print(f"Files in directory: {all_files}")
        
        # Check if we should start analysis
        analysis_started = False
        should_start = False
        
        # Determine if we should start analysis
        if start_analysis_now:
            should_start = True
            print(f"Explicit request to start analysis")
        elif total_files and len(all_files) >= int(total_files):
            should_start = True
            print(f"All expected files uploaded ({len(all_files)}/{total_files})")
        elif not total_files:
            # No total_files specified - use different logic
            if len(uploaded_files) < 10 and len(all_files) == len(uploaded_files):
                # First and only chunk
                should_start = True
                print(f"Single chunk upload detected - starting analysis")
            elif len(all_files) > 0 and len(uploaded_files) < 5:
                # Small chunk, might be last
                should_start = True
                print(f"Small chunk detected - starting analysis")
        
        # Additional failsafe: if we have files but no total_files after 2 seconds, start anyway
        if not should_start and not total_files and len(all_files) > 0:
            print(f"WARNING: No total_files specified but {len(all_files)} files uploaded")
            print(f"Consider starting analysis manually with /start_analysis/{job_id}")
            # For now, let's start it anyway if this looks like a complete upload
            if len(uploaded_files) == len(all_files):
                should_start = True
                print(f"Assuming this is a complete upload - starting analysis")
        
        if should_start:
            # Check if already processing
            if job_id in processing_jobs and processing_jobs[job_id].status == 'processing':
                print(f"Job {job_id} already processing")
                analysis_started = False
            else:
                # Start analysis
                print(f"Starting analysis for job {job_id} with {len(all_files)} files")
                print(f"Files to analyze: {all_files}")
                
                # Create job and add to tracking BEFORE starting thread
                job = AnalysisJob(job_id, len(all_files))
                processing_jobs[job_id] = job
                
                # Write initial status file with file count
                with open(os.path.join(job_dir, "status.json"), "w") as f:
                    json.dump({
                        "status": "processing",
                        "phase": "analysis",
                        "progress": 0,
                        "processed_files": 0,
                        "total_files": len(all_files),
                        "timestamp": datetime.now().isoformat()
                    }, f)
                
                # Start analysis in background
                thread = threading.Thread(
                    target=analyze_photos_background,
                    args=(job_id, job_dir)
                )
                thread.daemon = True
                thread.start()
                
                analysis_started = True
                print(f"Analysis thread started for job {job_id}")
            
            return jsonify({
                'job_id': job_id,
                'saved': len(saved_files),
                'total_saved': len(all_files),
                'message': f'Upload complete. Analysis started for {len(all_files)} files.',
                'analysis_started': True,
                'files': all_files  # Include list of files for debugging
            }), 200
        
        # Not starting analysis yet
        return jsonify({
            'job_id': job_id,
            'saved': len(saved_files),
            'total_saved': len(all_files),
            'message': f'Received {len(saved_files)} files for job {job_id}',
            'analysis_started': False,
            'hint': 'Call /start_analysis/{job_id} to begin processing' if len(all_files) > 0 else 'Upload more files',
            'files': all_files  # Include list of files for debugging
        }), 200
        
    except Exception as e:
        print(f"Upload error: {e}")
        import traceback
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
    
    # If status file exists, read it
    if os.path.exists(status_file):
        try:
            with open(status_file, 'r') as f:
                data = json.load(f)
            
            # Add debug info
            image_files = [f for f in os.listdir(job_dir) 
                          if os.path.splitext(f)[1].lower() in ALLOWED_EXTENSIONS]
            data['debug_files_found'] = len(image_files)
            
            return jsonify(data)
        except Exception as e:
            print(f"Error reading status file: {e}")
    
    # No status file yet - check if files exist
    image_files = [f for f in os.listdir(job_dir) 
                  if os.path.splitext(f)[1].lower() in ALLOWED_EXTENSIONS]
    
    # Default status with debug info
    return jsonify({
        'status': 'uploading',
        'phase': 'upload',
        'progress': 0,
        'processed_files': 0,
        'total_files': len(image_files),
        'debug_files_found': len(image_files),
        'debug_job_dir': job_dir
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
            
            # More conservative aesthetic scoring (centered around 5-6)
            base_score = 5.0
            
            # Small bonus for resolution (max +1.0 instead of +2.5)
            if megapixels >= 8:
                resolution_bonus = 1.0
            elif megapixels >= 4:
                resolution_bonus = 0.7
            elif megapixels >= 2:
                resolution_bonus = 0.4
            else:
                resolution_bonus = 0
            
            # Add controlled variety using hash (±1.5 range instead of ±1.0)
            file_hash = hashlib.md5(filename.encode()).hexdigest()
            hash_value = int(file_hash[:8], 16)  # Use more bits for better distribution
            variety = ((hash_value % 300) - 150) / 100.0  # Range: -1.5 to +1.5
            
            aesthetic_score = base_score + resolution_bonus + variety
            aesthetic_score = max(2, min(8.5, aesthetic_score))  # Cap at 8.5 for fallback
            
            # Blur estimation remains the same
            file_size = os.path.getsize(filepath) / 1024  # KB
            size_per_megapixel = file_size / megapixels if megapixels > 0 else 0
            
            if size_per_megapixel > 150:
                blur_score = 150
                blur_category = 'sharp'
            elif size_per_megapixel > 80:
                blur_score = 90
                blur_category = 'slightly_blurry'
            else:
                blur_score = 40
                blur_category = 'blurry'
            
            # More conservative composition score (centered around 5)
            composition_variety = ((hash_value >> 8) % 200 - 100) / 50.0  # Range: -2 to +2
            composition_score = 5.0 + composition_variety
            composition_score = max(3, min(7, composition_score))  # Cap at 7 for fallback
            
            # Combined score
            combined_score = (aesthetic_score * 0.4 + 
                            (blur_score/20) * 0.3 + 
                            composition_score * 0.3)
            
            # More conservative recommendations
            if combined_score >= 7.5:
                recommendation = 'use'
                action = 'Acceptable (fallback analysis - review recommended)'
            elif combined_score >= 5.5:
                recommendation = 'maybe'
                action = 'Manual review needed (fallback analysis)'
            else:
                recommendation = 'skip'
                action = 'Below threshold (fallback analysis)'
            
            # Rating based on aesthetic score
            if aesthetic_score >= 7:
                aesthetic_rating = 'good'
            elif aesthetic_score >= 5:
                aesthetic_rating = 'fair'
            else:
                aesthetic_rating = 'poor'
            
            return {
                'filename': filename,
                'aesthetic_score': round(aesthetic_score, 2),
                'blur_score': round(blur_score, 2),
                'blur_category': blur_category,
                'composition_score': round(composition_score, 2),
                'combined_score': round(combined_score, 2),
                'aesthetic_rating': aesthetic_rating,
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
    print(f"\n{'='*60}")
    print(f"analyze_photos_background started for job {job_id}")
    print(f"Folder path: {folder_path}")
    print(f"{'='*60}")
    
    # Get or create job
    job = processing_jobs.get(job_id)
    if not job:
        print(f"Creating new job object for {job_id}")
        job = AnalysisJob(job_id, 0)
        processing_jobs[job_id] = job
    
    try:
        print(f"Starting analysis for job {job_id}")
        print(f"HuggingFace Space: {HF_SPACE_URL}")
        
        # Initialize progress
        update_progress(folder_path, 0, 1, 'analysis')
        
        # Wake up HuggingFace Space
        print("Waking up HuggingFace Space...")
        try:
            wake_response = requests.get(HF_SPACE_URL, timeout=15)
            if wake_response.status_code == 200:
                print("✔ HuggingFace Space is responding")
                time.sleep(3)  # Give it more time to fully wake
            else:
                print(f"⚠ HuggingFace returned status {wake_response.status_code}")
        except Exception as e:
            print(f"⚠ Could not reach HuggingFace Space: {e}")
            print("Continuing anyway...")
        
        # Get image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        files = []
        
        # Check if folder exists
        if not os.path.exists(folder_path):
            print(f"ERROR: Folder does not exist: {folder_path}")
            raise FileNotFoundError(f"Upload folder not found: {folder_path}")
        
        # List all files in directory
        all_files_in_dir = os.listdir(folder_path)
        print(f"Files in directory: {all_files_in_dir}")
        
        # Filter for image files
        for f in all_files_in_dir:
            if os.path.splitext(f)[1].lower() in image_extensions:
                if os.path.isfile(os.path.join(folder_path, f)):
                    files.append(f)
        
        print(f"Found {len(files)} image files to process")
        
        if len(files) == 0:
            print("ERROR: No image files found!")
            job.status = 'error'
            job.error = 'No image files found to process'
            update_progress(folder_path, 0, 0, 'error')
            return
        
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
            print(f"  Full path: {filepath}")
            
            # Update job state
            job.processed_files = i
            job.current_file = filename
            
            # Update progress before processing
            update_progress(folder_path, i, total_files, 'analysis')
            
            try:
                # Check if file exists
                if not os.path.exists(filepath):
                    print(f"  WARNING: File not found: {filepath}")
                    continue
                
                # Try HuggingFace API
                print("  Calling HuggingFace API...")
                hf_result = call_huggingface_api(filepath)
                
                if hf_result and isinstance(hf_result, dict):
                    print(f"  Raw HF result keys: {list(hf_result.keys())[:20]}")  # Limit output
                    
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
                        # Validate and sanity-check the scores
                        aesthetic = float(scores.get('aesthetic_score', 5.0))
                        
                        # Sanity check
                        if aesthetic > 9.5 or aesthetic < 1:
                            print(f"  ⚠ Suspicious aesthetic score: {aesthetic}, using fallback")
                            results[filepath] = simple_fallback_analysis(filepath, filename)
                        else:
                            results[filepath] = {
                                'filename': filename,
                                'aesthetic_score': aesthetic,
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
                        print(f"  ⚠ No valid scores in HF response, using fallback")
                        results[filepath] = simple_fallback_analysis(filepath, filename)
                else:
                    print(f"  ⚠ Invalid HF response or call failed, using fallback")
                    results[filepath] = simple_fallback_analysis(filepath, filename)
                    
            except Exception as e:
                print(f"  ✗ Error processing {filename}: {e}")
                import traceback
                traceback.print_exc()
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
            
            # Free memory periodically
            if i % 5 == 0:
                gc.collect()
        
        # Mark as completed
        job.processed_files = total_files
        job.results = results
        job.status = 'completed'
        job.phase = 'completed'
        
        print(f"\nAnalysis complete. Processed {total_files} files")
        
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
        error_count = sum(1 for r in results.values() if r.get('ml_source') == 'error')
        print(f"Summary: {hf_count} via HuggingFace, {fallback_count} via fallback, {error_count} errors")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"Fatal error in job {job_id}: {e}")
        import traceback
        traceback.print_exc()
        
        job.status = 'error'
        job.phase = 'error'
        job.error = str(e)
        
        # Write error status
        try:
            with open(os.path.join(folder_path, "status.json"), "w") as f:
                json.dump({
                    "status": "error",
                    "phase": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }, f)
        except:
            print("Could not write error status file")

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