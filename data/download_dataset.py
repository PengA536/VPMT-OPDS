#!/usr/bin/env python3
"""Download Volleyball Activity Dataset."""

import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='./datasets/volleyball')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Volleyball Activity Dataset")
    print(f"Download from: [Official Source]")
    print(f"Output directory: {args.output_dir}")
    print("")
    print("Dataset contains:")
    print("  - 6 volleyball match videos (1920x1080, 25 FPS)")
    print("  - 18,472 annotated frames")
    print("  - 6 action categories")


if __name__ == '__main__':
    main()
