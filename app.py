
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
from flask import Flask, request, jsonify, send_file

def call_huggingface_direct(image_path):
    """Call HuggingFace Space directly (fallback)"""
    try:
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
        
        # Direct HuggingFace API callfrom flask import Flask, request, jsonify, send_file
    except Exception as e:  
            print(f"  Error calling HuggingFace direct: {e}")
            import traceback
            traceback.print_exc()
            return None 
# Bridge server URL (Vercel deployment)
BRIDGE_URL = os.getenv('BRIDGE_URL', 'https://yearbook-bridge.vercel.app')

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev-key-change-in-production')

print(f"=== STARTUP CONFIG ===")
print(f"Bridge URL: {BRIDGE_URL}")
print(f"======================")

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
HF_SPACE_URL = os.getenv('HF_SPACE_URL', 'https://pororoororoq-photo-analyzer.hf.space')

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

@app.after_request
def after_request(response):
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
        'version': '3.1.0',
        'mode': 'HTTP Direct with Bridge',
        'bridge_url': BRIDGE_URL,
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

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads and start analysis"""
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files[]')
    
    # Filter valid files
    valid_files = []
    for file in files:
        if file and allowed_file(file.filename):
            valid_files.append(file)
    
    if not valid_files:
        return jsonify({'error': 'No valid image files provided'}), 400
    
    # Limit files to prevent memory issues
    if len(valid_files) > 30:
        return jsonify({'error': 'Too many files. Please upload 30 or fewer images at once.'}), 400
    
    # Create unique job ID
    job_id = str(uuid.uuid4())
    
    # Create job folder
    job_folder = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
    os.makedirs(job_folder, exist_ok=True)
    
    # Save files
    saved_files = []
    for file in valid_files:
        filename = secure_filename(file.filename)
        filepath = os.path.join(job_folder, filename)
        file.save(filepath)
        saved_files.append(filepath)
    
    # Create job object
    job = AnalysisJob(job_id, len(saved_files))
    processing_jobs[job_id] = job
    
    # Start analysis in background thread
    thread = threading.Thread(target=analyze_photos_background, args=(job_id, job_folder))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'job_id': job_id,
        'message': f'Analysis started for {len(saved_files)} images',
        'total_files': len(saved_files)
    })

def call_bridge_server(image_path):
    """Call Vercel bridge server for HuggingFace analysis"""
    try:
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
        
        # Call bridge server (with fallback URLs)
        bridge_urls = [
            f"{BRIDGE_URL}/api/analyze",
            f"{BRIDGE_URL}/api/analyze.py",
            f"{BRIDGE_URL}/api/analyze.ts"
        ]
        
        response = None
        for api_url in bridge_urls:
            try:
                response = requests.post(
                    api_url,
                    json={
                        "image": img_base64,
                        "enhance": True
                    },
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                
                if response.status_code == 200:
                    break
            except:
                continue
        
        if response and response.status_code == 200:
            result = response.json()
            
            print(f"  🔍 Bridge raw response type: {type(result)}")
            if isinstance(result, dict):
                print(f"  🔍 Bridge response keys: {list(result.keys())}")
                if 'scores' in result:
                    print(f"  🔍 Scores found: {result['scores']}")
                if 'analysis' in result:
                    print(f"  🔍 Analysis found: {result['analysis']}")
            
            return result
        else:
            print(f"  Bridge server returned {response.status_code if response else 'no response'}")
            return None
            
    except requests.exceptions.Timeout:
        print(f"  Bridge server timeout")
        return None
    except Exception as e:
        print(f"  Error calling bridge server: {e}")
        import traceback
        traceback.print_exc()
        return None
        
        print(f"  Bridge server status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            # Parse the response properly
            if isinstance(result, dict):
                # Check if it has the expected structure
                if 'scores' in result or 'status' in result:
                    return result
                # Handle wrapped response
                elif 'data' in result and isinstance(result['data'], list):
                    if len(result['data']) > 0:
                        return result['data'][0]
                # Handle other formats
                elif 'result' in result:
                    return result['result']
            
            print(f"  Unexpected response format from bridge: {result}")
            return result  # Return it anyway, will be parsed later
        else:
            print(f"  Bridge server returned {response.status_code}")
            return None
            
    except requests.exceptions.Timeout:
        print(f"  Bridge server timeout")
        return None
    except Exception as e:
        print(f"  Error calling bridge server: {e}")
        return None

def call_huggingface_direct(image_path):
    """Call HuggingFace Space directly (fallback)"""
    try:
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
        
        # Direct HuggingFace API call
        api_url = f"{HF_SPACE_URL}/run/predict"
        
        response = requests.post(
            api_url,
            json={
                "data": [
                    f"data:image/png;base64,{img_base64}",
                    True  # enhance_option
                ]
            },
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"  HF Direct API status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"  HF Response type: {type(result)}, keys: {result.keys() if isinstance(result, dict) else 'not a dict'}")
            
            # Handle Gradio response format
            if isinstance(result, dict) and 'data' in result:
                data = result.get('data', [])
                if data and len(data) > 0:
                    actual_data = data[0]
                    print(f"  HF data[0] type: {type(actual_data)}")
                    
                    # If it's a string, try to parse it as JSON
                    if isinstance(actual_data, str):
                        try:
                            parsed = json.loads(actual_data)
                            print(f"  Parsed JSON from string response")
                            return parsed
                        except:
                            print(f"  Could not parse string response as JSON")
                            # Try to extract values from the string
                            return parse_string_response(actual_data)
                    
                    return actual_data
            
            # Already in the right format
            elif isinstance(result, dict) and ('scores' in result or 'aesthetic_score' in result):
                return result
            
            print(f"  Unexpected HF response format")
            return result  # Return anyway, let the caller handle it
        
        return None
            
    except Exception as e:
        print(f"  Error calling HuggingFace direct: {e}")
        import traceback
        traceback.print_exc()
        return None

def parse_string_response(response_str):
    """Try to parse values from a string response"""
    try:
        # Initialize result
        result = {
            'scores': {},
            'analysis': {}
        }
        
        # Try to find numeric values in the string
        import re
        
        # Look for aesthetic score
        aesthetic_match = re.search(r'aesthetic[_\s]*score[:\s]*([0-9.]+)', response_str, re.IGNORECASE)
        if aesthetic_match:
            result['scores']['aesthetic_score'] = float(aesthetic_match.group(1))
        
        # Look for blur score
        blur_match = re.search(r'blur[_\s]*score[:\s]*([0-9.]+)', response_str, re.IGNORECASE)
        if blur_match:
            result['scores']['blur_score'] = float(blur_match.group(1))
        
        # Look for composition score
        comp_match = re.search(r'composition[_\s]*score[:\s]*([0-9.]+)', response_str, re.IGNORECASE)
        if comp_match:
            result['scores']['composition_score'] = float(comp_match.group(1))
        
        # Look for combined score
        combined_match = re.search(r'combined[_\s]*score[:\s]*([0-9.]+)', response_str, re.IGNORECASE)
        if combined_match:
            result['scores']['combined_score'] = float(combined_match.group(1))
        
        # Look for blur category
        if 'sharp' in response_str.lower():
            result['analysis']['blur_category'] = 'sharp'
        elif 'slightly_blurry' in response_str.lower() or 'slight' in response_str.lower():
            result['analysis']['blur_category'] = 'slightly_blurry'
        elif 'blurry' in response_str.lower():
            result['analysis']['blur_category'] = 'blurry'
        
        # Look for recommendations
        if 'use' in response_str.lower():
            result['analysis']['recommendation'] = 'use'
        elif 'enhance' in response_str.lower():
            result['analysis']['recommendation'] = 'enhance'
        elif 'skip' in response_str.lower():
            result['analysis']['recommendation'] = 'skip'
        else:
            result['analysis']['recommendation'] = 'maybe'
        
        return result if result['scores'] else None
        
    except Exception as e:
        print(f"  Error parsing string response: {e}")
        return None

def analyze_photos_background(job_id, folder_path):
    """Analyze photos using bridge server or direct HuggingFace"""
    job = processing_jobs.get(job_id)
    if not job:
        return
    
    try:
        print(f"\n{'='*60}")
        print(f"Starting analysis for job {job_id}")
        print(f"Bridge Server: {BRIDGE_URL}")
        print(f"HuggingFace Space: {HF_SPACE_URL}")
        print(f"{'='*60}")
        
        # Test which services are available
        bridge_available = True  # FORCE bridge to always be tried first!
        hf_direct_available = False  # NEVER try direct HF from Render
        
        # Test bridge server (but don't block if it fails the health check)
        try:
            test_response = requests.get(f"{BRIDGE_URL}/api/health", timeout=5)
            if test_response.status_code == 200:
                print(f"✓ Bridge server health check passed at {BRIDGE_URL}")
            else:
                print(f"⚠ Bridge health check returned {test_response.status_code}, but will try anyway")
        except Exception as e:
            print(f"⚠ Bridge health check failed ({e}), but will try anyway")
        
        print(f"📡 Will use bridge server at: {BRIDGE_URL}/api/analyze")
        
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
                score_data = None
                
                # Try bridge server first (THIS IS THE ONLY WAY THAT WORKS FROM RENDER!)
                if bridge_available:
                    print(f"  📡 Calling bridge server at {BRIDGE_URL}")
                    bridge_result = call_bridge_server(filepath)
                    if bridge_result:
                        print(f"  🔍 Bridge result type: {type(bridge_result)}")
                        if isinstance(bridge_result, dict):
                            print(f"  🔍 Bridge result keys: {list(bridge_result.keys())[:10]}")  # First 10 keys
                        
                        # Extract scores and analysis from the result
                        scores = {}
                        analysis = {}
                        
                        # Handle different response structures
                        if isinstance(bridge_result, dict):
                            # Best case: Direct scores/analysis structure
                            if 'scores' in bridge_result and isinstance(bridge_result['scores'], dict):
                                scores = bridge_result['scores']
                                print(f"  ✅ Found scores in 'scores' key: {scores}")
                            
                            if 'analysis' in bridge_result and isinstance(bridge_result['analysis'], dict):
                                analysis = bridge_result['analysis']
                                print(f"  ✅ Found analysis in 'analysis' key: {analysis}")
                            
                            # Fallback: Look for scores at root level
                            if not scores:
                                for key in ['aesthetic_score', 'blur_score', 'composition_score', 'combined_score']:
                                    if key in bridge_result:
                                        scores[key] = bridge_result[key]
                                        print(f"  📍 Found {key} at root: {bridge_result[key]}")
                            
                            # Fallback: Look for analysis at root level
                            if not analysis:
                                for key in ['blur_category', 'aesthetic_rating', 'recommendation', 'action', 'face_detected']:
                                    if key in bridge_result:
                                        analysis[key] = bridge_result[key]
                                        print(f"  📍 Found {key} at root: {bridge_result[key]}")
                        
                        # Debug: Print exactly what we extracted
                        print(f"  🎯 Final extracted scores: {scores}")
                        print(f"  🎯 Final extracted analysis: {analysis}")
                        
                        # Only use the result if we found actual numeric scores
                        if scores and any(k in scores for k in ['aesthetic_score', 'blur_score', 'composition_score']):
                            # CRITICAL: Use the ACTUAL values from HuggingFace, not defaults!
                            score_data = {
                                'aesthetic_score': float(scores['aesthetic_score']) if 'aesthetic_score' in scores else 5.0,
                                'blur_score': float(scores['blur_score']) if 'blur_score' in scores else 100.0,
                                'blur_category': analysis.get('blur_category', 'unknown'),
                                'composition_score': float(scores['composition_score']) if 'composition_score' in scores else 5.0,
                                'combined_score': float(scores['combined_score']) if 'combined_score' in scores else 5.0,
                                'aesthetic_rating': analysis.get('aesthetic_rating', 'fair'),
                                'recommendation': analysis.get('recommendation', 'maybe'),
                                'action': analysis.get('action', ''),
                                'ml_source': 'huggingface_via_bridge',
                                'face_detected': analysis.get('face_detected', False)
                            }
                            
                            print(f"  ✅✅✅ USING HUGGINGFACE SCORES:")
                            print(f"       Aesthetic: {score_data['aesthetic_score']:.1f}")
                            print(f"       Blur: {score_data['blur_score']:.0f} (THIS SHOULD NOT BE 200!)")
                            print(f"       Composition: {score_data['composition_score']:.1f}")
                            print(f"       Category: {score_data['blur_category']}")
                        else:
                            print(f"  ❌❌❌ NO VALID SCORES FOUND - WILL USE FALLBACK")
                            print(f"  ❌❌❌ This is why blur score shows 200!")
                            score_data = None
                
                # Try direct HuggingFace if bridge failed
                if not score_data and hf_direct_available:
                    hf_result = call_huggingface_direct(filepath)
                    if hf_result:
                        # Extract scores and analysis from the result
                        scores = {}
                        analysis = {}
                        
                        # Handle different response structures
                        if isinstance(hf_result, dict):
                            # Direct scores/analysis structure
                            if 'scores' in hf_result:
                                scores = hf_result.get('scores', {})
                                analysis = hf_result.get('analysis', {})
                            # Everything at root level
                            else:
                                # Extract score-like fields
                                for key in ['aesthetic_score', 'blur_score', 'composition_score', 'combined_score']:
                                    if key in hf_result:
                                        scores[key] = hf_result[key]
                                
                                # Extract analysis fields
                                for key in ['blur_category', 'aesthetic_rating', 'recommendation', 'action', 'face_detected']:
                                    if key in hf_result:
                                        analysis[key] = hf_result[key]
                        
                        # Only use the result if we found actual scores
                        if scores or analysis:
                            score_data = {
                                'aesthetic_score': float(scores.get('aesthetic_score', 5.0)),
                                'blur_score': float(scores.get('blur_score', 100)),
                                'blur_category': analysis.get('blur_category', 'unknown'),
                                'composition_score': float(scores.get('composition_score', 5.0)),
                                'combined_score': float(scores.get('combined_score', 5.0)),
                                'aesthetic_rating': analysis.get('aesthetic_rating', 'fair'),
                                'recommendation': analysis.get('recommendation', 'maybe'),
                                'action': analysis.get('action', ''),
                                'ml_source': 'huggingface_direct',
                                'face_detected': analysis.get('face_detected', False)
                            }
                            
                            print(f"  ✓ HF Direct scores - A:{score_data['aesthetic_score']:.1f}, "
                                  f"B:{score_data['blur_score']:.0f}, "
                                  f"C:{score_data['composition_score']:.1f}")
                        else:
                            print(f"  ⚠ HF returned data but no scores found")
                            score_data = None
                
                # Use fallback if all ML methods failed
                if not score_data:
                    score_data = simple_fallback_analysis(filepath)
                    print(f"  → Using fallback - A:{score_data['aesthetic_score']:.1f}, "
                          f"B:{score_data['blur_score']:.0f}, "
                          f"C:{score_data['composition_score']:.1f}")
                
                results[filepath] = {
                    'filename': filename,
                    **score_data
                }
                
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
        bridge_count = sum(1 for r in results.values() if r.get('ml_source') == 'bridge_server')
        hf_count = sum(1 for r in results.values() if r.get('ml_source') == 'huggingface_direct')
        fallback_count = sum(1 for r in results.values() if 'fallback' in r.get('ml_source', ''))
        
        print(f"Summary: {bridge_count} via Bridge, {hf_count} via HF Direct, {fallback_count} via fallback")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"Fatal error in job {job_id}: {e}")
        traceback.print_exc()
        job.status = 'error'
        job.error = str(e)

def simple_fallback_analysis(filepath):
    """Simple image analysis without ML libraries"""
    print(f"    🚨🚨🚨 FALLBACK ANALYSIS ACTIVATED - NO ML SCORES!")
    print(f"    🚨🚨🚨 THIS IS WHY YOU SEE BLUR SCORE = 200!")
    
    try:
        with Image.open(filepath) as img:
            width, height = img.size
            megapixels = (width * height) / 1_000_000
            
            # Basic aesthetic scoring
            aesthetic_score = 5.0
            if megapixels >= 4:
                aesthetic_score += 2.5
            elif megapixels >= 2:
                aesthetic_score += 1.5
            elif megapixels >= 1:
                aesthetic_score += 0.5
            
            # Add variety
            import hashlib
            file_hash = hashlib.md5(os.path.basename(filepath).encode()).hexdigest()
            hash_value = int(file_hash[:4], 16)
            variety = (hash_value % 20 - 10) / 10.0
            aesthetic_score += variety
            aesthetic_score = max(1, min(10, aesthetic_score))
            
            # Simple blur estimation
            file_size = os.path.getsize(filepath) / 1024  # KB
            size_per_megapixel = file_size / megapixels if megapixels > 0 else 0
            
            # THIS IS WHERE THE 200 COMES FROM!
            if size_per_megapixel > 150:
                blur_score = 200  # <-- THE CULPRIT!
                blur_category = 'sharp'
                print(f"    📍 Fallback set blur_score = 200 (file size/megapixel > 150)")
            elif size_per_megapixel > 80:
                blur_score = 100
                blur_category = 'slightly_blurry'
                print(f"    📍 Fallback set blur_score = 100")
            else:
                blur_score = 50
                blur_category = 'blurry'
                print(f"    📍 Fallback set blur_score = 50")
            
            # Basic composition score
            composition_score = 5.0 + (hash_value % 30 - 15) / 5.0
            composition_score = max(1, min(10, composition_score))
            
            # Calculate combined score
            blur_normalized = min(blur_score / 50, 10)
            combined_score = (blur_normalized * 0.4) + (aesthetic_score * 0.3) + (composition_score * 0.3)
            
            # Determine recommendation
            if blur_category == 'sharp' and aesthetic_score >= 7:
                recommendation = 'use'
                action = 'Ready to use (FALLBACK ANALYSIS)'
            elif aesthetic_score >= 6:
                recommendation = 'maybe'
                action = 'Manual review needed (FALLBACK ANALYSIS)'
            else:
                recommendation = 'skip'
                action = 'Below quality threshold (FALLBACK ANALYSIS)'
            
            result = {
                'aesthetic_score': round(aesthetic_score, 2),
                'blur_score': round(blur_score, 2),
                'blur_category': blur_category,
                'composition_score': round(composition_score, 2),
                'combined_score': round(combined_score, 2),
                'aesthetic_rating': 'excellent' if aesthetic_score >= 7 else 'good' if aesthetic_score >= 5 else 'fair',
                'recommendation': recommendation,
                'action': action,
                'ml_source': 'FALLBACK_NOT_ML',
                'face_detected': False
            }
            
            print(f"    🚨 FALLBACK RESULT: blur={result['blur_score']}, aesthetic={result['aesthetic_score']}")
            return result
            
    except Exception as e:
        print(f"  Error in fallback: {e}")
        return {
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
        'bridge_url': BRIDGE_URL,
        'hf_space_url': HF_SPACE_URL
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)