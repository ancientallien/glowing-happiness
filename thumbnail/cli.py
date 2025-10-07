#!/usr/bin/env python3
"""
Command-line interface for Animaker thumbnail generator.
Use this for testing or batch processing.
"""

import argparse
import os
import sys
from pathlib import Path

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Animaker - AI Thumbnail Generator CLI"
    )
    parser.add_argument(
        "prompt",
        help="Text prompt for the thumbnail"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: auto-generated)"
    )
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="Generate thumbnail without starting web server"
    )
    
    args = parser.parse_args()
    
    # Load environment
    from dotenv import load_dotenv
    load_dotenv()
    
    print(f"🎨 Generating thumbnail for: '{args.prompt}'")
    
    try:
        from .thumbnail_generator import ThumbnailGenerator
        
        generator = ThumbnailGenerator()
        result_path = generator.generate_thumbnail(args.prompt)
        
        if result_path and os.path.exists(result_path):
            print(f"✅ Thumbnail generated successfully!")
            print(f"📁 File saved to: {result_path}")
            
            # Copy to specified output path if provided
            if args.output:
                import shutil
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(result_path, output_path)
                print(f"📋 Copied to: {output_path}")
            
            return 0
        else:
            print("❌ Failed to generate thumbnail")
            return 1
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install dependencies with: uv sync")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
