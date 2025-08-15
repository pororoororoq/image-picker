#!/usr/bin/env python3
"""
Quick test script for the photo analyzer
"""

import os
from pathlib import Path
import urllib.request
from PIL import Image
import numpy as np

def create_test_images():
    """Create some test images with different qualities"""
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)
    
    print("Creating test images...")
    
    # Create a sharp, high-quality image
    img = Image.new('RGB', (1920, 1080), color='white')
    pixels = img.load()
    for i in range(0, 1920, 10):
        for j in range(0, 1080, 10):
            pixels[i, j] = (i % 255, j % 255, (i+j) % 255)
    img.save(test_dir / "sharp_good.jpg", quality=95)
    print("  ✓ Created sharp_good.jpg")
    
    # Create a blurry image
    from PIL import ImageFilter
    img_blur = img.filter(ImageFilter.BLUR).filter(ImageFilter.BLUR).filter(ImageFilter.BLUR)
    img_blur.save(test_dir / "blurry_image.jpg", quality=95)
    print("  ✓ Created blurry_image.jpg")
    
    # Create a low quality image
    img_small = Image.new('RGB', (320, 240), color='gray')
    for i in range(100):
        x, y = np.random.randint(0, 320), np.random.randint(0, 240)
        img_small.putpixel((x, y), (0, 0, 0))
    img_small.save(test_dir / "low_quality.jpg", quality=50)
    print("  ✓ Created low_quality.jpg")
    
    # Create a slightly blurry but good composition
    from PIL import ImageFilter
    img_slight = img.filter(ImageFilter.BLUR)
    img_slight.save(test_dir / "slight_blur_good.jpg", quality=90)
    print("  ✓ Created slight_blur_good.jpg")
    
    print(f"\nTest images created in '{test_dir}' directory")
    return str(test_dir)

def run_test():
    """Run the analyzer on test images"""
    # First, create test images
    test_folder = create_test_images()
    
    print("\n" + "="*50)
    print("Running Photo Analyzer on test images...")
    print("="*50)
    
    # Import and run the analyzer
    try:
        from photo_analyzer import PhotoAnalyzer
        
        analyzer = PhotoAnalyzer()
        results = analyzer.analyze_folder(test_folder)
        
        # Display results
        print("\n" + "="*50)
        print("TEST RESULTS:")
        print("="*50)
        
        for path, data in results.items():
            print(f"\nFile: {data.get('filename')}")
            print(f"  Blur Score: {data.get('blur_score', 0):.2f}")
            print(f"  Blur Category: {data.get('blur_category')}")
            print(f"  Aesthetic Score: {data.get('aesthetic_score', 0):.2f}/10")
            print(f"  Aesthetic Rating: {data.get('aesthetic_rating')}")
            print(f"  Combined Score: {data.get('combined_score', 0):.2f}/10")
            print(f"  Recommendation: {data.get('recommendation')}")
            print(f"  Action: {data.get('action')}")
        
        # Generate report
        analyzer.generate_report(".")
        
        print("\n✅ Test completed successfully!")
        print("Check the generated report files for detailed results.")
        
    except ImportError as e:
        print(f"\n❌ Error: Could not import photo_analyzer module.")
        print(f"Make sure all required files are in the same directory:")
        print("  - blur_detector.py")
        print("  - aesthetic_scorer.py")
        print("  - photo_analyzer.py")
        print(f"\nError details: {e}")
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()