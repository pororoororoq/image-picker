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
            'https://pororoororoq-photo-analyzer.hf.space'  # Update this to your actual space
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
            print(f"Calling HuggingFace for {Path(image_path).name}")
            
            # Load and prepare image
            image = Image.open(image_path).convert('RGB')
            
            # Convert to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Call HuggingFace API
            response = requests.post(
                f"{self.hf_space_url}/run/predict",
                json={
                    "data": [
                        f"data:image/png;base64,{img_base64}",
                        True  # enhance_option
                    ]
                },
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            print(f"HTTP response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Raw response: {json.dumps(result, indent=2)[:500]}...")  # Debug print
                
                # Parse Gradio response structure
                if 'data' in result and len(result['data']) > 0:
                    # Gradio returns results in data[0]
                    data = result['data'][0]
                    
                    # If the data is a string (JSON), parse it
                    if isinstance(data, str):
                        try:
                            data = json.loads(data)
                        except:
                            print(f"Could not parse JSON from string: {data[:100]}...")
                            return self._local_simple_score(image_path)
                    
                    # Now extract the scores from the proper structure
                    if isinstance(data, dict) and data.get('status') == 'success':
                        scores = data.get('scores', {})
                        analysis = data.get('analysis', {})
                        
                        print(f"✓ Got scores - Aesthetic: {scores.get('aesthetic_score')}, "
                              f"Blur: {scores.get('blur_score')}, "
                              f"Composition: {scores.get('composition_score')}")
                        
                        # Calculate recommendation based on the scores
                        aesthetic_score = scores.get('aesthetic_score', 5.0)
                        blur_score = scores.get('blur_score', 100)
                        blur_category = analysis.get('blur_category', 'unknown')
                        
                        return {
                            'aesthetic_score': aesthetic_score,
                            'blur_score': blur_score,
                            'blur_category': blur_category,
                            'composition_score': scores.get('composition_score', 5.0),
                            'combined_score': scores.get('combined_score', 5.0),
                            'aesthetic_rating': analysis.get('aesthetic_rating', 'fair'),
                            'recommendation': analysis.get('recommendation', 'maybe'),
                            'action': analysis.get('action', ''),
                            'ml_source': 'huggingface'
                        }
                    else:
                        print(f"Unexpected data structure: {type(data)}, keys: {data.keys() if isinstance(data, dict) else 'not a dict'}")
                        return self._local_simple_score(image_path)
                else:
                    print(f"No data in response or unexpected structure")
                    return self._local_simple_score(image_path)
            else:
                print(f"HTTP request failed with status {response.status_code}")
                if response.text:
                    print(f"Error response: {response.text[:200]}")
                return self._local_simple_score(image_path)
                
        except Exception as e:
            print(f"Error calling HuggingFace API: {e}")
            import traceback
            traceback.print_exc()
            return self._local_simple_score(image_path)
    
    def _local_simple_score(self, image_path: str) -> Dict:
        """Fallback scoring without ML"""
        try:
            image = Image.open(image_path).convert('RGB')
            width, height = image.size
            
            # Very basic scoring as fallback
            score = 5.0
            megapixels = (width * height) / 1_000_000
            
            if megapixels >= 4:
                score += 1.5
            elif megapixels >= 2:
                score += 0.5
            
            # Add some variety based on file hash
            import hashlib
            with open(image_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            hash_variety = (int(file_hash[:2], 16) % 20 - 10) / 10.0
            score += hash_variety
            
            score = max(1, min(10, score))
            
            return {
                'aesthetic_score': round(score, 2),
                'aesthetic_rating': 'fair',
                'composition_score': 5.0,
                'blur_score': 100,  # Default blur score
                'blur_category': 'unknown',
                'combined_score': round(score, 2),
                'recommendation': 'maybe',
                'action': 'Manual review needed',
                'ml_source': 'local_fallback'
            }
            
        except Exception as e:
            print(f"Error in fallback scoring: {e}")
            return {
                'aesthetic_score': 5.0,
                'aesthetic_rating': 'error',
                'composition_score': 5.0,
                'blur_score': -1,
                'blur_category': 'error',
                'combined_score': 5.0,
                'recommendation': 'skip',
                'action': f'Error: {str(e)}',
                'ml_source': 'error'
            }
    
    def process_folder(self, folder_path: str, existing_results: Dict = None) -> Dict:
        """Process all images in folder"""
        folder = Path(folder_path)
        results = existing_results or {}
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in folder.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        print(f"\nScoring {len(image_files)} images with HuggingFace...")
        
        for i, image_path in enumerate(image_files, 1):
            print(f"  [{i}/{len(image_files)}] {image_path.name}...", end="")
            
            score_data = self.score_image(str(image_path))
            
            path_key = str(image_path)
            if path_key in results:
                results[path_key].update(score_data)
            else:
                results[path_key] = {
                    'filename': image_path.name,
                    **score_data
                }
            
            print(f" Score: {score_data['aesthetic_score']:.1f}/10 (via {score_data.get('ml_source', 'unknown')})")
            
            # Small delay to not overwhelm HuggingFace
            if self.ml_available and i < len(image_files):
                time.sleep(0.5)
        
        return results