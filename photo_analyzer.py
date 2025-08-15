#!/usr/bin/env python3
"""
Yearbook Photo Analyzer
Analyzes photos for blur and aesthetic quality, then recommends best selections.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from datetime import datetime

# Import our modules (these would be in the same directory)
from blur_detector import BlurDetector
from aesthetic_scorer import AestheticScorer

class PhotoAnalyzer:
    """Main analyzer that combines blur detection and aesthetic scoring"""
    
    def __init__(self):
        self.blur_detector = BlurDetector()
        self.aesthetic_scorer = AestheticScorer()
        self.results = {}
        
    def analyze_folder(self, folder_path: str) -> Dict:
        """Perform complete analysis on all images in folder"""
        print(f"\n{'='*50}")
        print(f"YEARBOOK PHOTO ANALYZER")
        print(f"{'='*50}")
        print(f"Analyzing folder: {folder_path}\n")
        
        # Step 1: Blur detection
        print("Step 1: Detecting blur...")
        self.results = self.blur_detector.process_folder(folder_path)
        
        # Step 2: Aesthetic scoring
        print("\nStep 2: Scoring aesthetic quality...")
        self.results = self.aesthetic_scorer.process_folder(folder_path, self.results)
        
        # Step 3: Calculate combined scores
        print("\nStep 3: Calculating recommendations...")
        self._calculate_recommendations()
        
        return self.results
    
    def _calculate_recommendations(self):
        """Calculate overall recommendations based on both metrics"""
        for path, data in self.results.items():
            # Skip if there were errors
            if data.get('blur_category') == 'error' or data.get('aesthetic_rating') == 'error':
                data['recommendation'] = 'error'
                data['combined_score'] = 0
                continue
            
            # Calculate combined score (weighted average)
            # Normalize blur score (0-1000 -> 0-10)
            blur_normalized = min(data.get('blur_score', 0) / 100, 10)
            aesthetic = data.get('aesthetic_score', 5)
            
            # Weight: 40% blur, 60% aesthetic (aesthetic is usually more important)
            combined = (blur_normalized * 0.4) + (aesthetic * 0.6)
            data['combined_score'] = round(combined, 2)
            
            # Make recommendation
            blur_cat = data.get('blur_category', 'unknown')
            aesthetic_rat = data.get('aesthetic_rating', 'unknown')
            
            if blur_cat == 'sharp' and aesthetic_rat in ['excellent', 'good']:
                data['recommendation'] = 'use'
                data['action'] = 'Ready to use'
            elif blur_cat == 'slightly_blurry' and aesthetic_rat in ['excellent', 'good']:
                data['recommendation'] = 'enhance'
                data['action'] = 'Good photo - enhance to remove slight blur'
            elif blur_cat == 'sharp' and aesthetic_rat == 'fair':
                data['recommendation'] = 'maybe'
                data['action'] = 'Sharp but average aesthetics - use if needed'
            elif aesthetic_rat == 'excellent' and blur_cat == 'blurry':
                data['recommendation'] = 'enhance'
                data['action'] = 'Great composition but blurry - try enhancement'
            else:
                data['recommendation'] = 'skip'
                data['action'] = 'Not recommended'
    
    def get_top_photos(self, n: int = 10) -> List[Tuple[str, Dict]]:
        """Get the top N photos by combined score"""
        sorted_photos = sorted(
            self.results.items(),
            key=lambda x: x[1].get('combined_score', 0),
            reverse=True
        )
        return sorted_photos[:n]
    
    def get_enhancement_candidates(self) -> List[Tuple[str, Dict]]:
        """Get photos that would benefit from enhancement"""
        candidates = [
            (path, data) for path, data in self.results.items()
            if data.get('recommendation') == 'enhance'
        ]
        return sorted(
            candidates,
            key=lambda x: x[1].get('aesthetic_score', 0),
            reverse=True
        )
    
    def generate_report(self, output_dir: str = "."):
        """Generate comprehensive analysis report"""
        output_dir = Path(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save complete results
        results_file = output_dir / f"photo_analysis_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate HTML report
        report_file = output_dir / f"photo_report_{timestamp}.html"
        self._generate_html_report(report_file)
        
        # Generate summary text file
        summary_file = output_dir / f"photo_summary_{timestamp}.txt"
        self._generate_text_summary(summary_file)
        
        print(f"\n{'='*50}")
        print(f"Reports generated:")
        print(f"  - Full results: {results_file}")
        print(f"  - HTML report: {report_file}")
        print(f"  - Text summary: {summary_file}")
        print(f"{'='*50}")
    
    def _generate_html_report(self, output_file: Path):
        """Generate an HTML report with recommendations"""
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Yearbook Photo Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        .stats { background: #f0f0f0; padding: 15px; border-radius: 5px; margin: 20px 0; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        .use { background-color: #d4edda; }
        .enhance { background-color: #fff3cd; }
        .maybe { background-color: #d1ecf1; }
        .skip { background-color: #f8d7da; }
    </style>
</head>
<body>
    <h1>Yearbook Photo Analysis Report</h1>
    <div class="stats">
        <h2>Summary Statistics</h2>
"""
        
        # Calculate statistics
        stats = {
            'total': len(self.results),
            'use': sum(1 for d in self.results.values() if d.get('recommendation') == 'use'),
            'enhance': sum(1 for d in self.results.values() if d.get('recommendation') == 'enhance'),
            'maybe': sum(1 for d in self.results.values() if d.get('recommendation') == 'maybe'),
            'skip': sum(1 for d in self.results.values() if d.get('recommendation') == 'skip'),
        }
        
        html_content += f"""
        <p>Total photos analyzed: <strong>{stats['total']}</strong></p>
        <p>Ready to use: <strong>{stats['use']}</strong></p>
        <p>Need enhancement: <strong>{stats['enhance']}</strong></p>
        <p>Maybe use: <strong>{stats['maybe']}</strong></p>
        <p>Skip: <strong>{stats['skip']}</strong></p>
    </div>
    
    <h2>Top Recommended Photos</h2>
    <table>
        <tr>
            <th>Filename</th>
            <th>Combined Score</th>
            <th>Blur Category</th>
            <th>Aesthetic Rating</th>
            <th>Recommendation</th>
            <th>Action</th>
        </tr>
"""
        
        # Add top photos to table
        for path, data in self.get_top_photos(20):
            rec_class = data.get('recommendation', 'unknown')
            html_content += f"""
        <tr class="{rec_class}">
            <td>{data.get('filename', 'unknown')}</td>
            <td>{data.get('combined_score', 0):.2f}</td>
            <td>{data.get('blur_category', 'unknown')}</td>
            <td>{data.get('aesthetic_rating', 'unknown')}</td>
            <td>{data.get('recommendation', 'unknown')}</td>
            <td>{data.get('action', '')}</td>
        </tr>
"""
        
        html_content += """
    </table>
</body>
</html>
"""
        
        with open(output_file, 'w') as f:
            f.write(html_content)
    
    def _generate_text_summary(self, output_file: Path):
        """Generate a text summary of recommendations"""
        with open(output_file, 'w') as f:
            f.write("YEARBOOK PHOTO SELECTION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Photos to definitely use
            f.write("RECOMMENDED PHOTOS (Ready to use):\n")
            f.write("-" * 30 + "\n")
            for path, data in self.results.items():
                if data.get('recommendation') == 'use':
                    f.write(f"  • {data.get('filename')} (score: {data.get('combined_score', 0):.2f})\n")
            
            # Photos to enhance
            f.write("\n\nENHANCEMENT CANDIDATES:\n")
            f.write("-" * 30 + "\n")
            for path, data in self.get_enhancement_candidates():
                f.write(f"  • {data.get('filename')} - {data.get('action')}\n")
            
            # Summary stats
            f.write("\n\nSUMMARY STATISTICS:\n")
            f.write("-" * 30 + "\n")
            total = len(self.results)
            for rec in ['use', 'enhance', 'maybe', 'skip']:
                count = sum(1 for d in self.results.values() if d.get('recommendation') == rec)
                percentage = (count / total * 100) if total > 0 else 0
                f.write(f"  {rec.upper()}: {count} photos ({percentage:.1f}%)\n")

def main():
    parser = argparse.ArgumentParser(description='Analyze photos for yearbook selection')
    parser.add_argument('folder', help='Path to folder containing photos')
    parser.add_argument('--top', type=int, default=10, help='Number of top photos to show')
    parser.add_argument('--output', default='.', help='Output directory for reports')
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = PhotoAnalyzer()
    results = analyzer.analyze_folder(args.folder)
    
    # Show top photos
    print(f"\n{'='*50}")
    print(f"TOP {args.top} RECOMMENDED PHOTOS:")
    print(f"{'='*50}")
    
    for i, (path, data) in enumerate(analyzer.get_top_photos(args.top), 1):
        print(f"\n{i}. {data.get('filename')}")
        print(f"   Combined Score: {data.get('combined_score', 0):.2f}/10")
        print(f"   Blur: {data.get('blur_category')} | Aesthetic: {data.get('aesthetic_rating')}")
        print(f"   Action: {data.get('action')}")
    
    # Show enhancement candidates
    candidates = analyzer.get_enhancement_candidates()
    if candidates:
        print(f"\n{'='*50}")
        print(f"PHOTOS THAT WOULD BENEFIT FROM ENHANCEMENT:")
        print(f"{'='*50}")
        
        for path, data in candidates[:5]:
            print(f"\n• {data.get('filename')}")
            print(f"  {data.get('action')}")
    
    # Generate reports
    analyzer.generate_report(args.output)

if __name__ == "__main__":
    main()