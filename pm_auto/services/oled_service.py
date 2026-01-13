from pm_auto.services.pironman_mcu_service import INTERVAL
from ..libs.ssd1306 import SSD1306, Rect
from sf_rpi_status import \
    get_cpu_temperature, \
    get_cpu_percent, \
    get_memory_info, \
    get_disks_info, \
    get_ips
from ..libs.i2c import I2C
from ..libs.utils import format_bytes, log_error

import time
import threading
import os
from enum import Enum
from PIL import Image
import numpy as np
from importlib.resources import files as resource_files

INTERVAL = 1

OLED_DEFAULT_CONFIG = {
    'temperature_unit': 'C',
    'oled_enable': True,
    'oled_rotation': 0,
    'oled_disk': 'total',  # 'total' or the name of the disk, normally 'mmcblk0' for SD Card, 'nvme0n1' for NVMe SSD
    'oled_network_interface': 'all',  # 'all' or the name of the interface, normally 'wlan0' for WiFi, 'eth0' for Ethernet
    'oled_sleep_timeout': 0,
    # Video intro settings
    'video_enabled': True,              # Enable video intro mode
    'video_frames_path': None,          # Path to preprocessed frames file (auto-detected if None)
    'info_display_interval': 300,       # Show device info every N seconds (5 min = 300)
    'info_display_duration': 10,        # Show device info for N seconds
    'video_fps': 15,                    # Video playback FPS
}

class OLEDPage(Enum):
    POWER_OFF = 0
    ALL_INFO = 1
    LOGO = 2           # Still logo image
    GREETING = 3       # Welcome greeting
    SERVER_INFO = 4    # Custom server message
    CYCLE_INFO = 5     # System info in cycle

class OLEDService():
    @log_error
    def __init__(self, config, get_logger=None):
        if get_logger is None:
            import logging
            get_logger = logging.getLogger
        self.log = get_logger(__name__)
        self._is_ready = False

        try:
            self.oled = SSD1306()
        except Exception as e:
            self.log.error(f"Failed to initialize OLED service: {e}")
            return
        self._is_ready = self.oled.is_ready()

        self.temperature_unit = OLED_DEFAULT_CONFIG['temperature_unit']
        self.disk_mode = OLED_DEFAULT_CONFIG['oled_disk']
        self.ip_interface = OLED_DEFAULT_CONFIG['oled_network_interface']
        self.sleep_timeout = OLED_DEFAULT_CONFIG['oled_sleep_timeout']
        self.enable = OLED_DEFAULT_CONFIG['oled_enable']
        self.ip_index = 0
        self.ip_show_next_timestamp = 0
        self.ip_show_next_interval = 3
        self.wake_flag = True
        self.button = False
        self.wake_start_time = 0
        self.last_ips = {}

        self.running = False
        self.thread = None
        self.current_page = OLEDPage.LOGO
        
        # Display cycle settings
        self.cycle_enabled = True
        self.cycle_interval = 60  # Show cycle every 60 seconds
        self.last_cycle_time = time.time()  # Initialize to current time
        self.cycle_state = 0  # 0=logo, 1=greeting, 2=server_info, 3=info
        self.cycle_state_start_time = time.time()  # Initialize to current time
        self.greeting_duration = 3      # Show greeting for 3 sec
        self.server_info_duration = 5   # Show server info for 5 sec  
        self.info_duration = 8          # Show system info for 8 sec
        
        # Custom messages
        self.server_name = "FALCON 1"
        self.server_owner = "HITROO"
        self.server_description = "Private Server for Lightscape"
        
        # Logo frame
        self.logo_frame = None
        self._load_logo_frame(config.get('video_frames_path'))
        
        self.update_config(config)

    @log_error
    def set_debug_level(self, level):
        self.log.setLevel(level)

    @log_error
    def _load_logo_frame(self, frames_path=None):
        """Load the first frame from preprocessed video as a still logo."""
        # Try to find frames file
        if frames_path is None:
            package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            possible_paths = [
                os.path.join(package_dir, 'video_frames.npz'),
                os.path.join(package_dir, '..', 'video_frames.npz'),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    frames_path = path
                    break
        
        if frames_path is None or not os.path.exists(frames_path):
            self.log.warning("No logo frames file found")
            return
        
        try:
            data = np.load(frames_path)
            frames = data['frames']
            # Use the first frame as the logo
            self.logo_frame = frames[0]  # Shape: (64, 128) boolean
            self.log.info(f"Loaded logo frame from {frames_path}")
            self.current_page = OLEDPage.LOGO
            self.last_cycle_time = time.time()
        except Exception as e:
            self.log.error(f"Failed to load logo frame: {e}")

    @log_error
    def update_config(self, config):
        if "temperature_unit" in config:
            if config['temperature_unit'] not in ['C', 'F']:
                self.log.error("Invalid temperature unit")
                return
            self.log.debug(f"Update temperature_unit to {config['temperature_unit']}")
            self.temperature_unit = config['temperature_unit']
        if "oled_rotation" in config:
            self.log.debug(f"Update oled_rotation to {config['oled_rotation']}")
            self.set_rotation(config['oled_rotation'])
        if "oled_disk" in config:
            self.log.debug(f"Update oled_disk to {config['oled_disk']}")
            self.disk_mode = config['oled_disk']
        if "oled_network_interface" in config:
            self.log.debug(f"Update oled_network_interface to {config['oled_network_interface']}")
            self.ip_interface = config['oled_network_interface']
        if "oled_sleep_timeout" in config:
            self.log.debug(f"Update oled_sleep_timeout to {config['oled_sleep_timeout']}")
            self.sleep_timeout = config['oled_sleep_timeout']
        if "oled_enable" in config:
            self.log.debug(f"Update oled_enable to {config['oled_enable']}")
            if config['oled_enable']:
                self.wake()
            else:
                self.sleep()
        # Video settings
        if "video_enabled" in config:
            self.video_enabled = config['video_enabled']
        if "video_fps" in config:
            self.video_fps = config['video_fps']
        if "info_display_interval" in config:
            self.info_display_interval = config['info_display_interval']
        if "info_display_duration" in config:
            self.info_display_duration = config['info_display_duration']

    @log_error
    def set_rotation(self, rotation):
        self.oled.set_rotation(rotation)

    @log_error
    def is_ready(self):
        return self._is_ready

    @log_error
    def get_data(self):
        memory_info = get_memory_info()
        ips = get_ips()

        data = {
            'cpu_temperature': get_cpu_temperature(),
            'cpu_percent': get_cpu_percent(),
            'memory_total': memory_info.total,
            'memory_used': memory_info.used,
            'memory_percent': memory_info.percent,
            'ips': []
        }
        # Get disk info
        disks_info = get_disks_info()
        data['disk_total'] = 0
        data['disk_used'] = 0
        data['disk_percent'] = 0
        data['disk_mounted'] = False
        if self.disk_mode == 'total':
            for disk in disks_info.values():
                if disk.mounted:
                    data['disk_total'] += disk.total
                    data['disk_used'] += disk.used
                    data['disk_percent'] += disk.percent
                    data['disk_mounted'] = True
        else:
            disk = disks_info[self.disk_mode]
            if disk.mounted:
                data['disk_total'] = disk.total
                data['disk_used'] = disk.used
                data['disk_percent'] = disk.percent
                data['disk_mounted'] = True
            else:
                data['disk_total'] = disk.total
                data['disk_mounted'] = False
        
        # Get IPs
        for interface, ip in ips.items():
            if interface not in self.last_ips:
                self.log.info(f"Connected to {interface}: {ip}")
            elif self.last_ips[interface] != ip:
                self.log.info(f"IP changed for {interface}: {ip}")
            self.last_ips[interface] = ip
        for interface in self.last_ips.keys():
            if interface not in ips:
                self.log.info(f"Disconnected from {interface}")
                self.last_ips.pop(interface)

        if len(ips) > 0:
            if self.ip_interface == 'all':
                data['ips'] = list(ips.values())
            elif self.ip_interface in ips:
                data['ips'] = [ips[self.ip_interface]]
                self.ip_index = 0
            else:
                self.log.warning(f"Invalid interface: {self.ip_interface}, available interfaces: {list(ips.keys())}")

        return data

    @log_error
    def draw_all_info(self):
        data = self.get_data()
        # Get system status data
        cpu_temp_c = data['cpu_temperature']
        cpu_temp_f = cpu_temp_c * 9 / 5 + 32
        cpu_usage = data['cpu_percent']
        memory_total, memory_unit = format_bytes(data['memory_total'])
        memory_used = format_bytes(data['memory_used'], memory_unit)
        memory_percent = data['memory_percent']
        disk_total, disk_unit = format_bytes(data['disk_total'])
        if data['disk_mounted']:
            disk_used = format_bytes(data['disk_used'], disk_unit)
            disk_percent = data['disk_percent']
        else:
            disk_used = 'NA'
            disk_percent = 0
        ips = data['ips']
        ip = 'DISCONNECTED'

        if len(ips) > 0:
            ip = ips[self.ip_index]
            if time.time() - self.ip_show_next_timestamp > self.ip_show_next_interval:
                self.ip_show_next_timestamp = time.time()
                self.ip_index = (self.ip_index + 1) % len(ips)

        # Clear draw buffer
        self.oled.clear()

        # ---- display info ----
        ip_rect =           Rect(39,  0, 88, 10)
        memory_info_rect =  Rect(39, 17, 88, 10)
        memory_rect =       Rect(39, 29, 88, 10)
        disk_info_rect =    Rect(39, 41, 88, 10)
        disk_rect =         Rect(39, 53, 88, 10)

        LEFT_AREA_X = 18
        # cpu usage
        self.oled.draw_text('CPU', LEFT_AREA_X, 0, align='center')
        self.oled.draw_pieslice_chart(cpu_usage, LEFT_AREA_X, 27, 15, 180, 0)
        self.oled.draw_text(f'{cpu_usage} %', LEFT_AREA_X, 27, align='center')
        # cpu temp
        temp = cpu_temp_c if self.temperature_unit == 'C' else cpu_temp_f
        self.oled.draw_text(f'{temp:.1f}°{self.temperature_unit}', LEFT_AREA_X, 37, align='center')
        self.oled.draw_pieslice_chart(cpu_temp_c, LEFT_AREA_X, 48, 15, 0, 180)
        # RAM
        self.oled.draw_text(f'RAM:  {memory_used}/{memory_total} {memory_unit}', *memory_info_rect.coord())
        self.oled.draw_bar_graph_horizontal(memory_percent, *memory_rect.coord(), *memory_rect.size())
        # Disk
        self.oled.draw_text(f'DISK: {disk_used}/{disk_total} {disk_unit}', *disk_info_rect.coord())
        self.oled.draw_bar_graph_horizontal(disk_percent, *disk_rect.coord(), *disk_rect.size())
        # IP
        self.oled.draw.rectangle((ip_rect.x,ip_rect.y,ip_rect.x+ip_rect.width,ip_rect.height), outline=1, fill=1)
        self.oled.draw_text(ip, *ip_rect.topcenter(), fill=0, align='center')

        # draw the image buffer
        self.oled.display()

    @log_error
    def draw_power_off(self):
        self.oled.clear()
        self.oled.draw_text(f'POWER OFF', 64, 20, align='center', size=16)
        self.oled.display()

    @log_error
    def show_shutdown_screen(self, reason):
        self.log.info(f"Shutdown reason: {reason}")
        self.current_page = OLEDPage.POWER_OFF
        self.wake()

    @log_error
    def wake(self):
        self.wake_start_time = time.time()
        self.wake_flag = True

    def set_button(self, button_state):
        self.button = button_state

    @log_error
    def sleep(self):
        self.wake_flag = False
        self.oled.clear()
        self.oled.display()

    @log_error
    def draw_logo(self):
        """Draw the still logo image."""
        if self.logo_frame is None:
            return False
        
        # Convert numpy boolean array to PIL Image
        frame_array = (self.logo_frame * 255).astype(np.uint8)
        frame_image = Image.fromarray(frame_array, mode='L')
        frame_image = frame_image.convert('1')
        
        self.oled.image = frame_image
        self.oled.display()
        return True

    @log_error
    def draw_greeting(self):
        """Draw a welcome greeting with animation effect."""
        self.oled.clear()
        # Greeting text with stars
        self.oled.draw_text("★ WELCOME ★", 64, 10, align='center', size=12)
        self.oled.draw_text("to", 64, 28, align='center', size=8)
        self.oled.draw_text(self.server_name, 64, 40, align='center', size=14)
        self.oled.display()

    @log_error
    def draw_server_info(self):
        """Draw custom server information."""
        self.oled.clear()
        # Server name with box
        self.oled.draw.rectangle((0, 0, 127, 14), outline=1, fill=1)
        self.oled.draw_text(self.server_name, 64, 2, fill=0, align='center', size=10)
        # Owner info
        self.oled.draw_text(f"by {self.server_owner}", 64, 20, align='center', size=10)
        # Description with line
        self.oled.draw.line((10, 35, 117, 35), fill=1)
        self.oled.draw_text(self.server_description, 64, 42, align='center', size=8)
        # Footer
        self.oled.draw_text("━━━━━━━━━━━━━━━", 64, 55, align='center', size=6)
        self.oled.display()

    @log_error
    def loop(self):
        from ..oled_page.ips import oled_page_ips
        from ..oled_page.disk import oled_page_disk
        from ..oled_page.performance import oled_page_performance

        info_pages = [oled_page_performance, oled_page_ips, oled_page_disk]
        page_index = 0
        last_refresh_time = 0
        logo_drawn = False

        if self.oled is None or not self.oled.is_ready():
            self.log.error("OLED service not ready")
            return

        self.log.info(f"OLED display cycle started. Logo loaded: {self.logo_frame is not None}")
        self.last_cycle_time = time.time()

        while self.running:
            current_time = time.time()

            # Handle button presses
            if self.button == 'single_click':
                if not self.wake_flag:
                    self.wake_flag = True
                else:
                    # Manual cycle through info pages
                    page_index = (page_index + 1) % len(info_pages)
                    info_pages[page_index](self.oled)
                self.button = False
                self.wake_start_time = current_time
            elif self.button:
                self.button = False

            # Handle power off
            if self.current_page == OLEDPage.POWER_OFF:
                self.draw_power_off()
                time.sleep(0.1)
                continue

            if not self.wake_flag:
                time.sleep(0.1)
                continue

            # ===== DISPLAY CYCLE LOGIC =====
            time_since_cycle = current_time - self.last_cycle_time
            time_in_state = current_time - self.cycle_state_start_time

            # State 0: LOGO (showing most of the time)
            if self.cycle_state == 0:
                if not logo_drawn:
                    if self.logo_frame is not None:
                        self.draw_logo()
                    else:
                        # Fallback: show text logo
                        self.oled.clear()
                        self.oled.draw_text(self.server_name, 64, 25, align='center', size=16)
                        self.oled.display()
                    logo_drawn = True
                
                # Check if it's time to start the cycle
                if time_since_cycle >= self.cycle_interval:
                    self.cycle_state = 1
                    self.cycle_state_start_time = current_time
                    logo_drawn = False
                    self.log.debug("Starting display cycle: Greeting")

            # State 1: GREETING
            elif self.cycle_state == 1:
                if time_in_state < 0.1:  # Draw once at start
                    self.draw_greeting()
                
                if time_in_state >= self.greeting_duration:
                    self.cycle_state = 2
                    self.cycle_state_start_time = current_time
                    self.log.debug("Display cycle: Server Info")

            # State 2: SERVER INFO
            elif self.cycle_state == 2:
                if time_in_state < 0.1:  # Draw once at start
                    self.draw_server_info()
                
                if time_in_state >= self.server_info_duration:
                    self.cycle_state = 3
                    self.cycle_state_start_time = current_time
                    page_index = 0
                    self.log.debug("Display cycle: System Info")

            # State 3: SYSTEM INFO
            elif self.cycle_state == 3:
                # Refresh info page periodically
                if current_time - last_refresh_time > INTERVAL:
                    last_refresh_time = current_time
                    info_pages[page_index](self.oled)
                
                if time_in_state >= self.info_duration:
                    # Cycle complete, return to logo
                    self.cycle_state = 0
                    self.last_cycle_time = current_time
                    self.cycle_state_start_time = current_time
                    logo_drawn = False
                    self.log.debug("Display cycle complete, returning to logo")

            # Sleep timeout
            if self.sleep_timeout > 0 and current_time - self.wake_start_time > self.sleep_timeout:
                self.sleep()
                continue

            time.sleep(0.1)

    @log_error
    def start(self):
        if self.running:
            self.log.warning("OLED service already running")
            return
        self.running = True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    @log_error
    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()
        if self.oled is not None and self.oled.is_ready():
            self.oled.clear()
            self.oled.display()
            self.oled.off()
            self.log.debug("OLED service closed")

