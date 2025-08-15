import torch
import clip
from PIL import Image
from pathlib import Path
from typing import Dict, List
import json
import numpy as np
from transformers import pipeline, AutoProcessor, AutoModel
import warnings
warnings.filterwarnings('ignore')

class AestheticScorer:
    """Score images based on aesthetic quality"""
    
    def __init__(self, method="simple"):
        """
        Initialize the aesthetic scoring model
        method: "simple", "clip", or "transformers"
        """
        self.method = method
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        if method == "transformers":
            try:
                print("Loading aesthetic model from HuggingFace...")
                # Use a model that's actually available on HuggingFace
                from transformers import pipeline
                self.pipe = pipeline("image-classification", 
                                    model="cafeai/cafe_aesthetic", 
                                    device=0 if self.device == "cuda" else -1)
                print("Model loaded successfully!")
            except Exception as e:
                print(f"Could not load HuggingFace model: {e}")
                print("Falling back to simple method...")
                self.method = "simple"
                
        elif method == "clip":
            try:
                print("Loading CLIP-based aesthetic scoring...")
                # This would require downloading the aesthetic predictor weights
                # For now, we'll use the simple method
                print("CLIP method requires additional setup. Using simple method...")
                self.method = "simple"
            except Exception as e:
                print(f"Could not load CLIP model: {e}")
                self.method = "simple"
    
    def score_image(self, image_path: str) -> Dict:
        """
        Score a single image for aesthetic quality.
        Returns dict with score and rating.
        """
        try:
            if self.method == "transformers" and hasattr(self, 'pipe'):
                # Use the HuggingFace model
                image = Image.open(image_path).convert('RGB')
                results = self.pipe(image)
                
                # Parse the results
                if results:
                    # Extract aesthetic score from the classification
                    aesthetic_score = self._parse_hf_results(results)
                else:
                    aesthetic_score = 5.0
            else:
                # Use simple heuristic scoring
                aesthetic_score = self._calculate_aesthetic_score(image_path)
            
            # Categorize the score
            if aesthetic_score >= 7:
                rating = 'excellent'
            elif aesthetic_score >= 5:
                rating = 'good'
            elif aesthetic_score >= 3:
                rating = 'fair'
            else:
                rating = 'poor'
            
            return {
                'aesthetic_score': round(aesthetic_score, 2),
                'aesthetic_rating': rating
            }
            
        except Exception as e:
            print(f"Error scoring {image_path}: {e}")
            return {
                'aesthetic_score': -1,
                'aesthetic_rating': 'error'
            }
    
    def _parse_hf_results(self, results):
        """Parse HuggingFace model results to get a score"""
        try:
            # Different models return results differently
            # This is a generic parser
            if isinstance(results, list) and len(results) > 0:
                first_result = results[0]
                if 'score' in first_result:
                    # Normalize confidence score to 1-10 scale
                    return first_result['score'] * 10
                elif 'label' in first_result:
                    # Parse label if it contains a score
                    label = first_result['label'].lower()
                    if 'high' in label or 'good' in label:
                        return 7.5
                    elif 'medium' in label or 'average' in label:
                        return 5.0
                    else:
                        return 2.5
            return 5.0
        except:
            return 5.0
    
    def _calculate_aesthetic_score(self, image_path: str) -> float:
        """
        Advanced heuristic scoring based on photographic principles.
        This is specifically tuned for yearbook photos.
        """
        try:
            image = Image.open(image_path).convert('RGB')
            img_array = np.array(image)
            height, width = img_array.shape[:2]
            
            score = 5.0  # Base score
            
            # 1. Resolution Quality (important for print)
            megapixels = (width * height) / 1_000_000
            if megapixels >= 8:  # 8MP+ is excellent
                score += 1.5
            elif megapixels >= 4:  # 4MP+ is good
                score += 1.0
            elif megapixels >= 2:  # 2MP+ is acceptable
                score += 0.5
            elif megapixels < 1:  # Less than 1MP is poor
                score -= 1.0
            
            # 2. Aspect Ratio (common photographic ratios)
            aspect_ratio = width / height
            golden_ratio = 1.618
            common_ratios = [1.0, 4/3, 3/2, 16/9, golden_ratio]
            
            # Check if aspect ratio is close to any common ratio
            ratio_scores = [abs(aspect_ratio - ratio) for ratio in common_ratios]
            if min(ratio_scores) < 0.1:
                score += 0.5
            
            # 3. Exposure Analysis
            # Convert to grayscale for histogram analysis
            gray = image.convert('L')
            histogram = np.array(gray.histogram())
            pixels_total = histogram.sum()
            
            # Check if image is well-exposed (not too dark or bright)
            dark_pixels = histogram[:50].sum() / pixels_total
            bright_pixels = histogram[205:].sum() / pixels_total
            mid_pixels = histogram[50:205].sum() / pixels_total
            
            if mid_pixels > 0.6:  # Well-balanced exposure
                score += 1.0
            if dark_pixels > 0.4:  # Too dark
                score -= 1.0
            if bright_pixels > 0.4:  # Too bright
                score -= 1.0
            
            # 4. Color Saturation (for yearbook vibrancy)
            hsv = image.convert('HSV')
            hsv_array = np.array(hsv)
            saturation = hsv_array[:,:,1]
            avg_saturation = np.mean(saturation)
            
            if 50 < avg_saturation < 180:  # Good color saturation
                score += 0.5
            elif avg_saturation < 30:  # Too desaturated
                score -= 0.5
            elif avg_saturation > 200:  # Oversaturated
                score -= 0.3
            
            # 5. Contrast Check
            std_dev = np.std(gray)
            if std_dev > 50:  # Good contrast
                score += 0.5
            elif std_dev < 20:  # Low contrast
                score -= 0.5
            
            # 6. Rule of Thirds Analysis (simplified)
            # Check if there's interesting content in the thirds
            third_h = height // 3
            third_w = width // 3
            
            # Sample regions at rule of thirds intersections
            regions = [
                img_array[third_h:third_h*2, third_w:third_w*2],  # Center
                img_array[:third_h, :third_w],  # Top-left
                img_array[:third_h, -third_w:],  # Top-right
                img_array[-third_h:, :third_w],  # Bottom-left
                img_array[-third_h:, -third_w:]  # Bottom-right
            ]
            
            # Check variance in these regions (higher variance = more interesting)
            region_variances = [np.var(region) for region in regions]
            if max(region_variances) > np.mean(region_variances) * 1.5:
                score += 0.3  # Has interesting composition
            
            # 7. Sharpness Indicator (edge detection)
            # This complements the blur detector
            edges = self._detect_edges(gray)
            edge_percentage = np.sum(edges > 0) / edges.size
            
            if edge_percentage > 0.1:  # Sharp, detailed image
                score += 0.5
            elif edge_percentage < 0.03:  # Lacks detail
                score -= 0.5
            
            # Clamp score between 1 and 10
            final_score = max(1, min(10, score))
            
            return final_score
            
        except Exception as e:
            print(f"Error in aesthetic calculation: {e}")
            return 5.0
    
    def _detect_edges(self, gray_image):
        """Simple edge detection for sharpness assessment"""
        try:
            import cv2
            gray_array = np.array(gray_image)
            edges = cv2.Canny(gray_array, 50, 150)
            return edges
        except:
            # If OpenCV is not available, return zeros
            return np.zeros_like(np.array(gray_image))
    
    def process_folder(self, folder_path: str, existing_results: Dict = None) -> Dict:
        """Process all images in a folder for aesthetic quality"""
        folder = Path(folder_path)
        results = existing_results or {}
        
        # Supported image formats
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in folder.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        print(f"\nScoring {len(image_files)} images for aesthetic quality...")
        print(f"Using method: {self.method}")
        
        for i, image_path in enumerate(image_files, 1):
            print(f"  [{i}/{len(image_files)}] Processing {image_path.name}...", end="")
            
            score_data = self.score_image(str(image_path))
            
            # Merge with existing results if available
            path_key = str(image_path)
            if path_key in results:
                results[path_key].update(score_data)
            else:
                results[path_key] = {
                    'filename': image_path.name,
                    **score_data
                }
            
            print(f" Score: {score_data['aesthetic_score']:.1f}/10 ({score_data['aesthetic_rating']})")
        
        return results
    
    def save_results(self, results: Dict, output_file: str = "aesthetic_results.json"):
        """Save results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")

# Quick test script
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = input("Enter folder path containing images: ")
    
    # Try different methods
    print("\nChoose scoring method:")
    print("1. Simple (fast, no external models)")
    print("2. Transformers (requires internet for model download)")
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        scorer = AestheticScorer(method="transformers")
    else:
        scorer = AestheticScorer(method="simple")
    
    results = scorer.process_folder(folder_path)
    
    # Summary statistics
    ratings = {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0, 'error': 0}
    total_score = 0
    valid_count = 0
    
    for result in results.values():
        ratings[result['aesthetic_rating']] += 1
        if result['aesthetic_score'] > 0:
            total_score += result['aesthetic_score']
            valid_count += 1
    
    print("\n=== Aesthetic Quality Summary ===")
    for rating, count in ratings.items():
        if count > 0:
            print(f"{rating}: {count} images")
    
    if valid_count > 0:
        print(f"Average score: {total_score/valid_count:.2f}/10")
    
    scorer.save_results(results)