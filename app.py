from flask import Flask, render_template, request, jsonify, send_file, session
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
from queue import Queue

# Import your existing modules
from blur_detector import BlurDetector
from aesthetic_scorer import AestheticScorer
from photo_analyzer import PhotoAnalyzer

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'

# Configure CORS to allow requests from Netlify
CORS(app, 
     origins="*",
     allow_headers="*",
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     supports_credentials=False,
     resources={r"/*": {"origins": "*"}})

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['ENHANCED_FOLDER'] = 'enhanced'

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'tif'}

# Create necessary folders
for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULTS_FOLDER'], app.config['ENHANCED_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# Global processing queue to track analysis jobs
processing_jobs = {}

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
        'version': '1.0.0',
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
    """Handle multiple file uploads and start analysis"""
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
    thread.start()
    
    return jsonify({
        'job_id': job_id,
        'message': f'Analysis started for {len(saved_files)} images',
        'total_files': len(saved_files)
    })

def analyze_photos_background(job_id, folder_path):
    """Run photo analysis in background with HuggingFace"""
    job = processing_jobs.get(job_id)
    if not job:
        print(f"Job {job_id} not found!")
        return
    
    try:
        print(f"Starting analysis for job {job_id}")
        
        # Import modules
        from blur_detector import BlurDetector
        from aesthetic_scorer import AestheticScorer
        import os
        
        # Get HuggingFace URL
        hf_url = os.getenv('HF_SPACE_URL', 'https://pororoororoq-yearbook-photo-analyzer.hf.space')
        print(f"Using HuggingFace Space: {hf_url}")
        
        # Initialize scorers
        blur_detector = BlurDetector()
        aesthetic_scorer = AestheticScorer(hf_url)
        
        # Check if HuggingFace is connected
        print(f"HuggingFace ML available: {aesthetic_scorer.ml_available}")
        print(f"Gradio client connected: {aesthetic_scorer.client is not None}")
        
        # Get list of images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        files = [f for f in os.listdir(folder_path) 
                if os.path.splitext(f)[1].lower() in image_extensions]
        
        total_files = len(files)
        print(f"Found {total_files} images to process")
        
        job.total_files = total_files
        job.status = 'processing'
        
        results = {}
        
        # Process each file
        for i, filename in enumerate(files):
            filepath = os.path.join(folder_path, filename)
            print(f"\nProcessing {i+1}/{total_files}: {filename}")
            
            # Update job progress
            job.processed_files = i
            job.current_file = filename
            
            try:
                # Get blur score
                blur_score, blur_category = blur_detector.detect_blur_laplacian(filepath)
                print(f"  Blur: {blur_score:.2f} ({blur_category})")
                
                # Get aesthetic score from HuggingFace
                aesthetic_data = aesthetic_scorer.score_image(filepath)
                print(f"  Aesthetic: {aesthetic_data.get('aesthetic_score', 0):.2f} (via {aesthetic_data.get('ml_source', 'unknown')})")
                
                # Calculate combined score
                blur_normalized = min(blur_score / 100, 10) if blur_score > 0 else 0
                aesthetic = aesthetic_data.get('aesthetic_score', 5)
                combined = (blur_normalized * 0.4) + (aesthetic * 0.6)
                
                # Determine recommendation
                if blur_category == 'sharp' and aesthetic >= 7:
                    recommendation = 'use'
                    action = 'Ready to use - high quality'
                elif blur_category == 'slightly_blurry' and aesthetic >= 7:
                    recommendation = 'enhance'
                    action = 'Good photo - enhance for sharpness'
                elif blur_category == 'sharp' and aesthetic >= 5:
                    recommendation = 'maybe'
                    action = 'Sharp but average aesthetics'
                elif aesthetic >= 8:
                    recommendation = 'enhance'
                    action = 'Great aesthetics but needs sharpening'
                else:
                    recommendation = 'skip'
                    action = 'Below quality threshold'
                
                results[filepath] = {
                    'filename': filename,
                    'blur_score': blur_score,
                    'blur_category': blur_category,
                    'aesthetic_score': aesthetic_data.get('aesthetic_score', 5),
                    'aesthetic_rating': aesthetic_data.get('aesthetic_rating', 'unknown'),
                    'composition_score': aesthetic_data.get('composition_score', 5),
                    'combined_score': round(combined, 2),
                    'recommendation': recommendation,
                    'action': action,
                    'ml_source': aesthetic_data.get('ml_source', 'unknown')
                }
                
            except Exception as e:
                print(f"  Error: {e}")
                results[filepath] = {
                    'filename': filename,
                    'blur_score': -1,
                    'blur_category': 'error',
                    'aesthetic_score': 5,
                    'aesthetic_rating': 'error',
                    'combined_score': 0,
                    'recommendation': 'skip',
                    'action': f'Error: {str(e)}',
                    'ml_source': 'error'
                }
        
        # Update final progress
        job.processed_files = total_files
        job.results = results
        job.status = 'completed'
        
        # Save results
        results_file = os.path.join(app.config['RESULTS_FOLDER'], f"{job_id}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nJob {job_id} completed!")
        print(f"Results summary:")
        ml_sources = {}
        for r in results.values():
            source = r.get('ml_source', 'unknown')
            ml_sources[source] = ml_sources.get(source, 0) + 1
        for source, count in ml_sources.items():
            print(f"  {source}: {count} images")
        
    except Exception as e:
        print(f"Fatal error in job {job_id}: {e}")
        import traceback
        traceback.print_exc()
        job.status = 'error'
        job.error = str(e)

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
        # Try to load from file if job object doesn't exist
        results_file = os.path.join(app.config['RESULTS_FOLDER'], f"{job_id}.json")
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            return jsonify({'results': results})
        return jsonify({'error': 'Job not found'}), 404
    
    if job.status != 'completed':
        return jsonify({'error': 'Analysis not yet completed', 'status': job.status}), 202
    
    # Process results for frontend
    processed_results = []
    for filepath, data in job.results.items():
        # Create relative path for image serving
        filename = os.path.basename(filepath)
        
        processed_results.append({
            'filename': filename,
            'filepath': filepath,
            'blur_score': data.get('blur_score', 0),
            'blur_category': data.get('blur_category', 'unknown'),
            'aesthetic_score': data.get('aesthetic_score', 0),
            'aesthetic_rating': data.get('aesthetic_rating', 'unknown'),
            'combined_score': data.get('combined_score', 0),
            'recommendation': data.get('recommendation', 'unknown'),
            'action': data.get('action', ''),
            'job_id': job_id
        })
    
    # Sort by combined score
    processed_results.sort(key=lambda x: x['combined_score'], reverse=True)
    
    # Calculate statistics
    stats = {
        'total': len(processed_results),
        'use': sum(1 for r in processed_results if r['recommendation'] == 'use'),
        'enhance': sum(1 for r in processed_results if r['recommendation'] == 'enhance'),
        'maybe': sum(1 for r in processed_results if r['recommendation'] == 'maybe'),
        'skip': sum(1 for r in processed_results if r['recommendation'] == 'skip'),
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
    # Remove upload folder
    upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
    if os.path.exists(upload_folder):
        shutil.rmtree(upload_folder)
    
    # Remove from processing jobs
    if job_id in processing_jobs:
        del processing_jobs[job_id]
    
    return jsonify({'message': 'Cleanup successful'})

@app.route('/api/enhance', methods=['POST'])
def enhance_image():
    """Placeholder for image enhancement endpoint"""
    # This will be implemented when you add the enhancement feature
    return jsonify({'message': 'Enhancement feature coming soon'}), 501

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Run in debug mode for development
    app.run(debug=True, port=5000, threaded=True)
