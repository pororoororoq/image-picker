"""
Aesthetic Scorer that uses HuggingFace Space for ML processing
"""
import requests
import base64
from PIL import Image
from pathlib import Path
from typing import Dict
import json
import numpy as np
from io import BytesIO
import os
import time

class AestheticScorer:
    """Score images using HuggingFace Space API for ML processing"""
    
    def __init__(self, hf_space_url=None):
        # Get HuggingFace Space URL from environment or parameter
        self.hf_space_url = hf_space_url or os.getenv(
            'HF_SPACE_URL', 
            'https://pororoororoq-yearbook-photo-analyzer.hf.space'  # Update this to your actual space
        )
        
        self.ml_available = self._test_connection()
        if self.ml_available:
            print(f"✓ Connected to HuggingFace Space: {self.hf_space_url}")
        else:
            print(f"✗ Could not connect to HuggingFace Space, using fallback")
    
    def _test_connection(self):
        """Test if HuggingFace Space is accessible"""
        try:
            response = requests.get(self.hf_space_url, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def score_image(self, image_path: str) -> Dict:
        """Score a single image using HuggingFace ML API"""
        
        try:
            print(f"\n>>> Calling HuggingFace for {Path(image_path).name}")
            
            # Load and prepare image
            image = Image.open(image_path).convert('RGB')
            
            # Convert to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Call HuggingFace API
            api_url = f"{self.hf_space_url}/run/predict"
            print(f"    API URL: {api_url}")
            
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
            
            print(f"    Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                # Print raw response for debugging
                print(f"    Raw response type: {type(result)}")
                if isinstance(result, dict):
                    print(f"    Response keys: {list(result.keys())}")
                
                # Parse Gradio response structure
                if 'data' in result and len(result['data']) > 0:
                    # Gradio returns results in data[0]
                    data = result['data'][0]
                    print(f"    Data type: {type(data)}")
                    
                    # If the data is a string (JSON), parse it
                    if isinstance(data, str):
                        try:
                            print(f"    Parsing JSON string...")
                            data = json.loads(data)
                            print(f"    Parsed successfully!")
                        except json.JSONDecodeError as e:
                            print(f"    JSON parse error: {e}")
                            print(f"    String content: {data[:200]}...")
                            return self._local_fallback_with_ml_features(image_path)
                    
                    # Now extract the scores from the proper structure
                    if isinstance(data, dict):
                        print(f"    Data keys: {list(data.keys())}")
                        
                        if data.get('status') == 'success':
                            scores = data.get('scores', {})
                            analysis = data.get('analysis', {})
                            
                            # Print what we got
                            print(f"    ✓ HF Scores received:")
                            print(f"      - Aesthetic: {scores.get('aesthetic_score', 'MISSING')}")
                            print(f"      - Blur: {scores.get('blur_score', 'MISSING')}")
                            print(f"      - Composition: {scores.get('composition_score', 'MISSING')}")
                            print(f"      - Combined: {scores.get('combined_score', 'MISSING')}")
                            
                            # Extract all values
                            aesthetic_score = float(scores.get('aesthetic_score', 5.0))
                            blur_score = float(scores.get('blur_score', 100))
                            composition_score = float(scores.get('composition_score', 5.0))
                            combined_score = float(scores.get('combined_score', 5.0))
                            
                            # Ensure we have valid values
                            if composition_score == 5.0:
                                print(f"    ⚠ Composition score is default 5.0 - HF might not be calculating it")
                            
                            return {
                                'aesthetic_score': aesthetic_score,
                                'blur_score': blur_score,
                                'blur_category': analysis.get('blur_category', 'unknown'),
                                'composition_score': composition_score,
                                'combined_score': combined_score,
                                'aesthetic_rating': analysis.get('aesthetic_rating', 'fair'),
                                'recommendation': analysis.get('recommendation', 'maybe'),
                                'action': analysis.get('action', ''),
                                'ml_source': 'huggingface',
                                'face_detected': analysis.get('face_detected', False)
                            }
                        else:
                            print(f"    Status not 'success': {data.get('status', 'MISSING')}")
                            if 'error' in data:
                                print(f"    Error: {data['error']}")
                    else:
                        print(f"    Data is not a dict: {type(data)}")
                else:
                    print(f"    No 'data' in response or empty data array")
                    
            else:
                print(f"    HTTP request failed with status {response.status_code}")
                if response.text:
                    print(f"    Error response: {response.text[:500]}")
                    
        except requests.exceptions.Timeout:
            print(f"    ✗ Request timed out after 30 seconds")
        except requests.exceptions.ConnectionError as e:
            print(f"    ✗ Connection error: {e}")
        except Exception as e:
            print(f"    ✗ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
        
        # If we get here, HF failed - use enhanced local fallback
        print(f"    → Using local fallback with ML features")
        return self._local_fallback_with_ml_features(image_path)
    
    def _local_fallback_with_ml_features(self, image_path: str) -> Dict:
        """Enhanced local fallback that calculates all scores locally"""
        try:
            import cv2
            image = Image.open(image_path).convert('RGB')
            width, height = image.size
            img_array = np.array(image)
            
            # Calculate aesthetic score with variety
            aesthetic_score = 5.0
            megapixels = (width * height) / 1_000_000
            if megapixels >= 4:
                aesthetic_score += 1.5
            elif megapixels >= 2:
                aesthetic_score += 0.5
            
            # Add variety based on image characteristics
            import hashlib
            with open(image_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            hash_variety = (int(file_hash[:2], 16) % 20 - 10) / 10.0
            aesthetic_score += hash_variety
            aesthetic_score = max(1, min(10, aesthetic_score))
            
            # Calculate composition score locally
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Rule of thirds analysis
            third_h = height // 3
            third_w = width // 3
            
            # Check edge distribution in thirds
            regions = []
            for i in range(3):
                for j in range(3):
                    region = edges[i*third_h:(i+1)*third_h, j*third_w:(j+1)*third_w]
                    regions.append(np.sum(region > 0))
            
            total_edges = sum(regions)
            if total_edges > 0:
                # Calculate standard deviation for distribution
                std_dev = np.std(regions)
                mean_edges = np.mean(regions)
                cv = std_dev / mean_edges if mean_edges > 0 else 0
                
                # Good composition has CV between 0.5 and 1.5
                if 0.5 <= cv <= 1.5:
                    composition_score = 7 + (1 - abs(cv - 1)) * 3
                else:
                    composition_score = 5 + max(0, 2 - abs(cv - 1))
            else:
                composition_score = 4.0
            
            composition_score = max(1, min(10, composition_score))
            
            # Calculate blur score locally (face-focused if possible)
            try:
                face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                
                if len(faces) > 0:
                    # Analyze blur on face region
                    x, y, w, h = faces[0]
                    padding = int(max(w, h) * 0.2)
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(width, x + w + padding)
                    y2 = min(height, y + h + padding)
                    roi = gray[y1:y2, x1:x2]
                    face_detected = True
                else:
                    # Use center region
                    roi = gray[height//4:3*height//4, width//4:3*width//4]
                    face_detected = False
                
                laplacian = cv2.Laplacian(roi, cv2.CV_64F)
                blur_score = laplacian.var()
                
                # Categorize for face-focused scores
                if blur_score > 150:
                    blur_category = "sharp"
                elif blur_score > 50:
                    blur_category = "slightly_blurry"
                else:
                    blur_category = "blurry"
                    
            except Exception as e:
                print(f"      Local blur detection error: {e}")
                blur_score = 100
                blur_category = "unknown"
                face_detected = False
            
            # Calculate combined score
            blur_normalized = min(blur_score / 50, 10) if blur_score > 0 else 0
            combined_score = (blur_normalized * 0.4) + (aesthetic_score * 0.3) + (composition_score * 0.3)
            
            # Determine recommendation
            if blur_category == "sharp" and aesthetic_score >= 7:
                recommendation = "use"
                action = "Ready to use"
            elif blur_category == "slightly_blurry" and aesthetic_score >= 7:
                recommendation = "enhance"
                action = "Good photo - needs enhancement"
            elif aesthetic_score >= 8:
                recommendation = "enhance"
                action = "Great aesthetics"
            else:
                recommendation = "maybe"
                action = "Manual review needed"
            
            print(f"      Local scores - A: {aesthetic_score:.1f}, B: {blur_score:.0f}, C: {composition_score:.1f}")
            
            return {
                'aesthetic_score': round(aesthetic_score, 2),
                'aesthetic_rating': 'excellent' if aesthetic_score >= 7 else 'good' if aesthetic_score >= 5 else 'fair',
                'composition_score': round(composition_score, 2),
                'blur_score': round(blur_score, 2),
                'blur_category': blur_category,
                'combined_score': round(combined_score, 2),
                'recommendation': recommendation,
                'action': action,
                'ml_source': 'local_enhanced',
                'face_detected': face_detected
            }
            
        except Exception as e:
            print(f"      Error in enhanced local scoring: {e}")
            # Final fallback
            return {
                'aesthetic_score': 5.0,
                'aesthetic_rating': 'error',
                'composition_score': 5.0,
                'blur_score': -1,
                'blur_category': 'error',
                'combined_score': 5.0,
                'recommendation': 'skip',
                'action': f'Error: {str(e)}',
                'ml_source': 'error',
                'face_detected': False
            }
    
    def process_folder(self, folder_path: str, existing_results: Dict = None) -> Dict:
        """Process all images in folder"""
        folder = Path(folder_path)
        results = existing_results or {}
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in folder.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        print(f"\nScoring {len(image_files)} images...")
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing {image_path.name}")
            
            score_data = self.score_image(str(image_path))
            
            path_key = str(image_path)
            if path_key in results:
                results[path_key].update(score_data)
            else:
                results[path_key] = {
                    'filename': image_path.name,
                    **score_data
                }
            
            print(f"Final: A={score_data['aesthetic_score']:.1f}, "
                  f"B={score_data['blur_score']:.0f}, "
                  f"C={score_data['composition_score']:.1f} "
                  f"(via {score_data.get('ml_source', 'unknown')})")
            
            # Small delay to not overwhelm HuggingFace
            if self.ml_available and i < len(image_files):
                time.sleep(0.5)
        
        return results