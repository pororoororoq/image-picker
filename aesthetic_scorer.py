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
            'https://pororoororoq-yearbook-photo-analyzer.hf.space'  # ← UPDATE THIS
        )
        
        # For Gradio Client API
        from gradio_client import Client
        self.client = None
        try:
            from gradio_client import Client
            self.client = Client(self.hf_space_url)
            self.ml_available = True
            print(f"Connected to HuggingFace Space via Gradio Client: {self.hf_space_url}")
        except ImportError:
            print("gradio_client not installed, using HTTP API")
            self.ml_available = self._test_connection()
        except Exception as e:
            print(f"Could not connect via Gradio Client: {e}")
            self.ml_available = self._test_connection()
    
    def _test_connection(self):
        """Test if HuggingFace Space is accessible"""
        try:
            response = requests.get(self.hf_space_url, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def score_image(self, image_path: str) -> Dict:
        """Score a single image using HuggingFace ML API"""
        
        # Skip Gradio Client for now, use direct HTTP
        try:
            # Read the image
            from PIL import Image
            image = Image.open(image_path)
            
            # Convert to base64
            import base64
            from io import BytesIO
            
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Call HuggingFace directly via HTTP
            import requests
            
            response = requests.post(
                f"{self.hf_space_url}/run/predict",
                json={
                    "data": [
                        f"data:image/png;base64,{img_base64}",
                        True
                    ]
                },
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # The response should have 'data' key
                if 'data' in result and len(result['data']) > 0:
                    data = result['data'][0]
                    
                    # Parse the JSON response
                    if isinstance(data, str):
                        data = json.loads(data)
                    
                    if data.get('status') == 'success':
                        scores = data.get('scores', {})
                        analysis = data.get('analysis', {})
                        
                        return {
                            'aesthetic_score': scores.get('aesthetic_score', 5.0),
                            'blur_score': scores.get('blur_score', 100),
                            'blur_category': analysis.get('blur_category', 'unknown'),
                            'composition_score': scores.get('composition_score', 5.0),
                            'combined_score': scores.get('combined_score', 5.0),
                            'aesthetic_rating': analysis.get('aesthetic_rating', 'fair'),
                            'recommendation': analysis.get('recommendation', 'maybe'),
                            'action': analysis.get('action', ''),
                            'ml_source': 'huggingface_http'
                        }
            
        except Exception as e:
            print(f"Error calling HuggingFace: {e}")
        
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
            
            return {
                'aesthetic_score': round(score, 2),
                'aesthetic_rating': 'fair',
                'composition_score': 5.0,
                'ml_source': 'local_fallback'
            }
            
        except Exception as e:
            return {
                'aesthetic_score': 5.0,
                'aesthetic_rating': 'error',
                'composition_score': 5.0,
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