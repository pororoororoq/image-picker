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

class AestheticScorer:
    """Score images using HuggingFace Space API for ML processing"""
    
    def __init__(self, hf_space_url=None):
        """
        Initialize with HuggingFace Space URL
        Format: https://username-spacename.hf.space
        """
        # Get from environment variable or parameter
        self.hf_space_url = hf_space_url or os.getenv(
            'HF_SPACE_URL', 
            'https://your-username-yearbook-photo-analyzer.hf.space'
        )
        self.api_endpoint = f"{self.hf_space_url}/run/predict"
        
        # Test connection
        self.ml_available = self._test_connection()
        
        if self.ml_available:
            print(f"Connected to HuggingFace Space: {self.hf_space_url}")
        else:
            print("Warning: HuggingFace Space not available, using fallback scoring")
    
    def _test_connection(self):
        """Test if HuggingFace Space is accessible"""
        try:
            response = requests.get(self.hf_space_url, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _image_to_base64(self, image_path):
        """Convert image to base64 for API transmission"""
        with Image.open(image_path) as img:
            # Resize if too large to reduce transmission time
            max_size = 1024
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
        return img_base64
    
    def score_image(self, image_path: str) -> Dict:
        """
        Score a single image using HuggingFace ML API
        Falls back to local scoring if API unavailable
        """
        if self.ml_available:
            try:
                # Convert image to base64
                img_base64 = self._image_to_base64(image_path)
                
                # Prepare request
                payload = {
                    "data": [
                        f"data:image/png;base64,{img_base64}",
                        True  # enhance_option
                    ]
                }
                
                # Call HuggingFace API
                response = requests.post(
                    self.api_endpoint,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extract scores from response
                    if 'data' in result and isinstance(result['data'], list):
                        ml_result = result['data'][0]
                        
                        if isinstance(ml_result, str):
                            # Parse JSON string if needed
                            ml_result = json.loads(ml_result)
                        
                        if ml_result.get('status') == 'success':
                            scores = ml_result.get('scores', {})
                            analysis = ml_result.get('analysis', {})
                            
                            return {
                                'aesthetic_score': scores.get('aesthetic_score', 5.0),
                                'aesthetic_rating': analysis.get('aesthetic_rating', 'unknown'),
                                'blur_score': scores.get('blur_score', 100),
                                'blur_category': analysis.get('blur_category', 'unknown'),
                                'composition_score': scores.get('composition_score', 5.0),
                                'combined_score': scores.get('combined_score', 5.0),
                                'recommendation': analysis.get('recommendation', 'maybe'),
                                'action': analysis.get('action', ''),
                                'ml_source': 'huggingface'
                            }
                
                # If API call failed, fall back
                print(f"HuggingFace API returned status {response.status_code}")
                
            except Exception as e:
                print(f"Error calling HuggingFace API: {e}")
        
        # Fallback to local simple scoring
        return self._local_simple_score(image_path)
    
    def _local_simple_score(self, image_path: str) -> Dict:
        """
        Fallback scoring without ML - runs locally
        Used when HuggingFace Space is unavailable
        """
        try:
            image = Image.open(image_path).convert('RGB')
            width, height = image.size
            
            # Basic heuristic scoring
            score = 5.0
            
            # Resolution check
            megapixels = (width * height) / 1_000_000
            if megapixels >= 4:
                score += 1.5
            elif megapixels >= 2:
                score += 0.5
            elif megapixels < 1:
                score -= 1.5
            
            # Basic image statistics
            img_array = np.array(image)
            
            # Brightness check
            mean_brightness = np.mean(img_array)
            if 60 < mean_brightness < 200:
                score += 0.5
            
            # Contrast check
            std_dev = np.std(img_array)
            if std_dev > 40:
                score += 0.5
            
            # Color saturation (simple version)
            if len(img_array.shape) == 3:
                color_std = np.std(img_array, axis=2).mean()
                if color_std > 30:
                    score += 0.5
            
            # Clamp score
            final_score = max(1, min(10, score))
            
            # Categorize
            if final_score >= 7:
                rating = 'good'
                recommendation = 'use'
            elif final_score >= 5:
                rating = 'fair'
                recommendation = 'maybe'
            else:
                rating = 'poor'
                recommendation = 'skip'
            
            return {
                'aesthetic_score': round(final_score, 2),
                'aesthetic_rating': rating,
                'blur_score': 100,  # Default
                'blur_category': 'unknown',
                'composition_score': 5.0,
                'combined_score': final_score,
                'recommendation': recommendation,
                'action': f'Fallback scoring: {rating}',
                'ml_source': 'local_fallback'
            }
            
        except Exception as e:
            print(f"Error in local scoring: {e}")
            return {
                'aesthetic_score': 5.0,
                'aesthetic_rating': 'error',
                'blur_score': -1,
                'blur_category': 'error',
                'composition_score': 5.0,
                'combined_score': 5.0,
                'recommendation': 'error',
                'action': 'Error processing image',
                'ml_source': 'error'
            }
    
    def process_folder(self, folder_path: str, existing_results: Dict = None) -> Dict:
        """Process all images in a folder"""
        folder = Path(folder_path)
        results = existing_results or {}
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in folder.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        print(f"\nScoring {len(image_files)} images...")
        if self.ml_available:
            print(f"Using HuggingFace ML API: {self.hf_space_url}")
        else:
            print("Using local fallback scoring")
        
        for i, image_path in enumerate(image_files, 1):
            print(f"  [{i}/{len(image_files)}] Processing {image_path.name}...", end="")
            
            score_data = self.score_image(str(image_path))
            
            # Merge with existing results if available (e.g., from blur detector)
            path_key = str(image_path)
            if path_key in results:
                results[path_key].update(score_data)
            else:
                results[path_key] = {
                    'filename': image_path.name,
                    **score_data
                }
            
            print(f" Score: {score_data['aesthetic_score']:.1f}/10 ({score_data.get('ml_source', 'unknown')})")
            
            # Add small delay to avoid overwhelming the API
            if self.ml_available and i < len(image_files):
                import time
                time.sleep(0.5)  # Rate limiting
        
        return results
    
    def save_results(self, results: Dict, output_file: str = "aesthetic_results.json"):
        """Save results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")

# For backward compatibility
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = input("Enter folder path containing images: ")
    
    # Get HuggingFace Space URL from environment or user
    hf_url = os.getenv('HF_SPACE_URL')
    if not hf_url:
        print("\nEnter your HuggingFace Space URL")
        print("(e.g., https://username-yearbook-photo-analyzer.hf.space)")
        hf_url = input("URL: ").strip()
    
    scorer = AestheticScorer(hf_url)
    results = scorer.process_folder(folder_path)
    
    # Summary
    print("\n=== Summary ===")
    ml_sources = {}
    for result in results.values():
        source = result.get('ml_source', 'unknown')
        ml_sources[source] = ml_sources.get(source, 0) + 1
    
    for source, count in ml_sources.items():
        print(f"{source}: {count} images")
    
    scorer.save_results(results)