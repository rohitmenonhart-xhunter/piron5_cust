"""
Example: OLED Video Intro Playback

This example demonstrates running the OLED with video intro.
The logo animation plays in a loop, with device info shown every 5 minutes.

Prerequisites:
1. Run `python3 -m pm_auto.preprocess_video --input video.mp4 --output pm_auto/video_frames.npz`
2. Ensure video_frames.npz is in the pm_auto directory
"""

from pm_auto.pm_auto import PMAuto
import time
import logging

# Configuration
config = {
    'temperature_unit': 'C',
    'oled_rotation': 0,
    # Video settings (optional - defaults work fine)
    'video_enabled': True,
    'info_display_interval': 300,  # Show device info every 5 minutes
    'info_display_duration': 10,   # Show info for 10 seconds
    'video_fps': 15,               # Playback FPS
}

peripherals = [
    'oled',
    # Add other peripherals as needed:
    # 'ws2812',
    # 'gpio_fan',
]

def get_logger(name):
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    log.addHandler(console_handler)
    return log

pm_auto = PMAuto(config, peripherals=peripherals, get_logger=get_logger)

def main():
    print("=" * 50)
    print("OLED Video Intro Demo")
    print("=" * 50)
    print("")
    print("Controls:")
    print("  Single click  - Switch to info pages / cycle pages")
    print("  Double click  - Previous page")
    print("  Long press    - Toggle between video and info mode")
    print("")
    print("Press Ctrl+C to exit")
    print("")
    
    try:
        pm_auto.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        pm_auto.stop()
        print("Goodbye!")

if __name__ == "__main__":
    main()
