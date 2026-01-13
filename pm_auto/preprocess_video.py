#!/usr/bin/env python3
"""
Video Preprocessor for OLED Display

Converts video files to optimized 128x64 monochrome frames for SSD1306 OLED display.
Frames are saved as a compressed numpy array for fast loading on Raspberry Pi.

Usage:
    python3 -m pm_auto.preprocess_video --input video.mp4 --output frames.npy --preview
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image

# OLED Display dimensions
OLED_WIDTH = 128
OLED_HEIGHT = 64


def extract_frames_from_video(video_path, target_fps=15):
    """
    Extract frames from video file using cv2.
    Returns list of PIL Images.
    """
    try:
        import cv2
    except ImportError:
        print("OpenCV not found. Installing...")
        os.system(f"{sys.executable} -m pip install opencv-python")
        import cv2
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_fps if original_fps > 0 else 0
    
    print(f"Video: {os.path.basename(video_path)}")
    print(f"  Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"  FPS: {original_fps:.2f}")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Total frames: {total_frames}")
    
    # Calculate frame skip to achieve target FPS
    frame_skip = max(1, int(original_fps / target_fps))
    target_frame_count = int(total_frames / frame_skip)
    
    print(f"\nTarget FPS: {target_fps}")
    print(f"Frame skip: {frame_skip}")
    print(f"Output frames: ~{target_frame_count}")
    
    frames = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_skip == 0:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            frames.append(pil_image)
        
        frame_idx += 1
    
    cap.release()
    print(f"Extracted {len(frames)} frames")
    return frames, target_fps


def process_frame_for_oled(frame, dither=True, threshold=128):
    """
    Process a single frame for OLED display:
    - Resize to 128x64 with aspect ratio preservation
    - Convert to 1-bit monochrome
    """
    # Calculate aspect ratio preserving resize
    original_width, original_height = frame.size
    aspect = original_width / original_height
    oled_aspect = OLED_WIDTH / OLED_HEIGHT
    
    if aspect > oled_aspect:
        # Video is wider - fit to width
        new_width = OLED_WIDTH
        new_height = int(OLED_WIDTH / aspect)
    else:
        # Video is taller - fit to height
        new_height = OLED_HEIGHT
        new_width = int(OLED_HEIGHT * aspect)
    
    # Resize with high quality
    resized = frame.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create black background at exact OLED size
    output = Image.new('L', (OLED_WIDTH, OLED_HEIGHT), 0)
    
    # Paste centered
    x_offset = (OLED_WIDTH - new_width) // 2
    y_offset = (OLED_HEIGHT - new_height) // 2
    
    # Convert to grayscale and paste
    resized_gray = resized.convert('L')
    output.paste(resized_gray, (x_offset, y_offset))
    
    # Convert to 1-bit
    if dither:
        # Floyd-Steinberg dithering for smoother gradients
        output_1bit = output.convert('1', dither=Image.Dither.FLOYDSTEINBERG)
    else:
        # Simple threshold
        output_1bit = output.point(lambda p: 255 if p > threshold else 0, mode='1')
    
    return output_1bit


def frames_to_numpy(processed_frames):
    """
    Convert list of PIL 1-bit images to numpy array.
    Shape: (num_frames, 64, 128) with dtype=bool
    """
    num_frames = len(processed_frames)
    # Preallocate array
    arr = np.zeros((num_frames, OLED_HEIGHT, OLED_WIDTH), dtype=bool)
    
    for i, frame in enumerate(processed_frames):
        # Convert to numpy (PIL 1-bit: 255=white, 0=black)
        frame_arr = np.array(frame)
        arr[i] = frame_arr > 0
    
    return arr


def save_frames(frames_array, output_path, fps):
    """
    Save frames array and metadata to compressed numpy file.
    """
    np.savez_compressed(
        output_path,
        frames=frames_array,
        fps=fps,
        width=OLED_WIDTH,
        height=OLED_HEIGHT
    )
    file_size = os.path.getsize(output_path) / 1024
    print(f"\nSaved {len(frames_array)} frames to {output_path}")
    print(f"File size: {file_size:.1f} KB")


def create_preview_gif(processed_frames, output_path, fps=15):
    """
    Create a preview GIF from processed frames.
    """
    # Convert 1-bit frames to RGB for GIF
    rgb_frames = []
    for frame in processed_frames:
        # Convert 1-bit to RGB (white on black)
        rgb = frame.convert('RGB')
        rgb_frames.append(rgb)
    
    # Scale up for easier viewing
    scale = 4
    scaled_frames = [
        f.resize((OLED_WIDTH * scale, OLED_HEIGHT * scale), Image.Resampling.NEAREST)
        for f in rgb_frames
    ]
    
    # Save as GIF
    duration_ms = int(1000 / fps)
    scaled_frames[0].save(
        output_path,
        save_all=True,
        append_images=scaled_frames[1:],
        duration=duration_ms,
        loop=0
    )
    print(f"Preview GIF saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess video for OLED display')
    parser.add_argument('--input', '-i', required=True, help='Input video file path')
    parser.add_argument('--output', '-o', default='video_frames.npz', help='Output frames file path')
    parser.add_argument('--fps', type=int, default=15, help='Target FPS (default: 15)')
    parser.add_argument('--preview', action='store_true', help='Generate preview GIF')
    parser.add_argument('--no-dither', action='store_true', help='Disable dithering (use threshold)')
    parser.add_argument('--threshold', type=int, default=128, help='Threshold for non-dithered mode (0-255)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    print("=" * 50)
    print("OLED Video Preprocessor")
    print("=" * 50)
    
    # Extract frames
    print("\n[1/3] Extracting frames from video...")
    raw_frames, fps = extract_frames_from_video(args.input, args.fps)
    
    # Process frames
    print("\n[2/3] Processing frames for OLED...")
    processed_frames = []
    for i, frame in enumerate(raw_frames):
        processed = process_frame_for_oled(frame, dither=not args.no_dither, threshold=args.threshold)
        processed_frames.append(processed)
        if (i + 1) % 10 == 0 or i == len(raw_frames) - 1:
            print(f"  Processed {i + 1}/{len(raw_frames)} frames", end='\r')
    print()
    
    # Convert to numpy and save
    print("\n[3/3] Saving frames...")
    frames_array = frames_to_numpy(processed_frames)
    save_frames(frames_array, args.output, fps)
    
    # Generate preview if requested
    if args.preview:
        preview_path = args.output.rsplit('.', 1)[0] + '_preview.gif'
        print("\nGenerating preview GIF...")
        create_preview_gif(processed_frames, preview_path, fps)
    
    print("\n" + "=" * 50)
    print("Done! Copy the .npz file to your Raspberry Pi.")
    print("=" * 50)


if __name__ == '__main__':
    main()
