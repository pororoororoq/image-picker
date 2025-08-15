import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json

class BlurDetector:
    """Detect blur in images using multiple methods"""
    
    def __init__(self):
        self.thresholds = {
            'sharp': 500,
            'slightly_blurry': 100,
            'blurry': 0
        }
    
    def detect_blur_laplacian(self, image_path: str) -> Tuple[float, str]:
        """
        Detect blur using Laplacian variance method.
        Returns: (blur_score, category)
        """
        try:
            # Read image in grayscale
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                return -1, "error"
            
            # Calculate Laplacian
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            variance = laplacian.var()
            
            # Categorize
            if variance >= self.thresholds['sharp']:
                category = 'sharp'
            elif variance >= self.thresholds['slightly_blurry']:
                category = 'slightly_blurry'
            else:
                category = 'blurry'
            
            return variance, category
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return -1, "error"
    
    def process_folder(self, folder_path: str) -> Dict:
        """Process all images in a folder"""
        folder = Path(folder_path)
        results = {}
        
        # Supported image formats
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in folder.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        print(f"Found {len(image_files)} images to process...")
        
        for image_path in image_files:
            score, category = self.detect_blur_laplacian(image_path)
            results[str(image_path)] = {
                'blur_score': score,
                'blur_category': category,
                'filename': image_path.name
            }
            print(f"  {image_path.name}: {category} (score: {score:.2f})")
        
        return results
    
    def save_results(self, results: Dict, output_file: str = "blur_results.json"):
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
    
    detector = BlurDetector()
    results = detector.process_folder(folder_path)
    
    # Summary statistics
    categories = {'sharp': 0, 'slightly_blurry': 0, 'blurry': 0, 'error': 0}
    for result in results.values():
        categories[result['blur_category']] += 1
    
    print("\n=== Summary ===")
    for category, count in categories.items():
        if count > 0:
            print(f"{category}: {count} images")
    
    detector.save_results(results)