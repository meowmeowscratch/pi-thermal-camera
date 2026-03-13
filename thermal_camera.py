#!/usr/bin/env python3
"""
Thermal Camera

Reads an 8x8 temperature grid from an AMG8833 thermal sensor and sends
each frame to meow meow scratch.  Optionally displays a live heat map
on a 1.3" OLED screen or in the terminal.
"""

import os                          # Access environment variables (API key)
import sys                         # Exit cleanly on missing config
import time                        # Timing control for the refresh loop
from dotenv import load_dotenv     # Load variables from .env file

# Read the .env file in the same directory as this script, so you
# don't have to export variables in your shell every time.
load_dotenv()
import board                       # Raspberry Pi pin definitions (from Adafruit Blinka)
import busio                       # I2C communication support
import adafruit_amg88xx            # Driver for the AMG8833 8x8 infrared thermal sensor
import numpy as np                 # Numerical operations for temperature processing
from meow_sdk import Meow, MeowError  # Client for the meow meow scratch API

# These imports are only needed when an OLED is connected.
# They are loaded here so the script still works without them.
try:
    from luma.core.interface.serial import i2c as luma_i2c   # I2C interface for the OLED
    from luma.oled.device import sh1106                      # SH1106 OLED controller driver
    from PIL import Image, ImageDraw, ImageFont              # Image creation and drawing
    OLED_AVAILABLE = True
except ImportError:
    OLED_AVAILABLE = False

# GlowBit library for the 8x8 LED matrix.
try:
    import glowbit                 # Core Electronics GlowBit matrix driver
    GLOWBIT_AVAILABLE = True
except ImportError:
    GLOWBIT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration — tweak these to suit your environment
# ---------------------------------------------------------------------------

# meow meow scratch settings
API_KEY = os.environ.get("MEOW_API_KEY", "")

APP = "heat"                       # Name of your app on meow meow scratch
ENDPOINT = "thermal"               # Endpoint where frames are stored

# Temperature range for the brightness mapping (degrees Celsius).
# Anything at or below TEMP_MIN appears black; at or above TEMP_MAX appears white.
# Narrow the range (e.g. 24–30) for more contrast indoors.
TEMP_MIN = 20.0
TEMP_MAX = 35.0

# AMG8833 I2C address.  The default is 0x69.  If you connect the sensor's
# address pin to ground, it changes to 0x68.  Run "sudo i2cdetect -y 1"
# to see which address your sensor is using.
AMG_ADDRESS = 0x69

# Set to True to skip sending data to the API.
# Useful for testing the sensor and display without network delays.
API_DISABLED = True

# How often to read and send a frame (in seconds).
# 1 second gives good coverage without flooding the API.
SEND_INTERVAL = 0.1

# OLED display dimensions (128x64 is standard for the 1.3" Duinotech OLED)
OLED_WIDTH = 128
OLED_HEIGHT = 64

# Size in pixels of the heat-map square drawn on the OLED.
HEATMAP_SIZE = 56

# GlowBit 8x8 LED matrix settings
GLOWBIT_PIN = 18                  # GPIO 18 — data pin for the LED matrix
GLOWBIT_BRIGHTNESS = 40           # 0–255, keep low to avoid blinding yourself

# Display orientation — how many times to rotate the 8x8 grid 90° clockwise.
#   0 = no rotation
#   1 = 90° clockwise
#   2 = 180° (upside down)
#   3 = 270° clockwise (90° counter-clockwise)
ROTATION = 0

# Flip the display vertically (top↔bottom) or horizontally (left↔right).
FLIP_VERTICAL = False
FLIP_HORIZONTAL = True

# Characters used to represent temperature in the terminal heat map,
# ordered from coldest to hottest.
SHADE_CHARS = " .-:=+*#%@"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def orient_grid(pixels):
    """
    Apply rotation and flip settings to the 8x8 grid for display.
    Rotation is clockwise in 90° steps.  Flips mirror the grid.
    """
    arr = np.array(pixels)
    if ROTATION:
        arr = np.rot90(arr, k=-ROTATION)
    if FLIP_VERTICAL:
        arr = np.flipud(arr)
    if FLIP_HORIZONTAL:
        arr = np.fliplr(arr)
    return arr.tolist()


def pixels_to_8bit(pixels):
    """
    Convert the 8x8 float temperature grid to a flat list of 64 integers
    in the 0–255 range.  This is the format needed for an 8-bit model
    input (e.g. Akida) and is compact for storage.

    The mapping:  TEMP_MIN → 0,  TEMP_MAX → 255.
    """
    arr = np.array(pixels, dtype=np.float32)
    arr = np.clip(arr, TEMP_MIN, TEMP_MAX)
    arr = ((arr - TEMP_MIN) / (TEMP_MAX - TEMP_MIN) * 255).astype(np.uint8)
    return arr.flatten().tolist()


def build_frame(pixels):
    """
    Build the data dict to send to meow meow scratch for one sensor reading.

    Fields match the collection schema set up by setup_endpoint():
      - pixels_raw  (json):   64 float temperatures, row by row
      - pixels_8bit (json):   64 uint8 values 0–255, for ML model input
      - temp_min    (number): coldest pixel in the frame
      - temp_max    (number): hottest pixel in the frame
      - temp_centre (number): average of the 4 centre pixels
      - temp_range  (json):   [TEMP_MIN, TEMP_MAX] used for 8-bit mapping
    """
    flat = [temp for row in pixels for temp in row]
    temp_min = round(min(flat), 2)
    temp_max = round(max(flat), 2)
    temp_centre = round(
        (pixels[3][3] + pixels[3][4] + pixels[4][3] + pixels[4][4]) / 4.0, 2
    )

    return {
        "pixels_raw": [round(t, 2) for t in flat],
        "pixels_8bit": pixels_to_8bit(pixels),
        "temp_min": temp_min,
        "temp_max": temp_max,
        "temp_centre": temp_centre,
        "temp_range": [TEMP_MIN, TEMP_MAX],
    }


def setup_endpoint(api):
    """
    Create the collection endpoint and its fields on meow meow scratch
    if they don't already exist.  Safe to call every time — the API
    will raise MeowError if the endpoint/field already exists, which
    we silently ignore.
    """
    # Create the collection endpoint
    try:
        api.create_endpoint(
            APP, "Thermal Frames", ENDPOINT,
            endpoint_type="collection",
            description="8x8 thermal sensor frames from the AMG8833",
            is_public=True,
        )
        print(f"Created endpoint: {APP}/{ENDPOINT}")
    except MeowError:
        pass  # Already exists

    # Define the fields that each record will contain
    fields = [
        ("pixels_raw",  "Raw Temperatures",    "json",   "64 float values in C, row by row"),
        ("pixels_8bit", "8-bit Pixels",        "json",   "64 uint8 values 0-255 for ML input"),
        ("temp_min",    "Min Temperature",     "number", "Coldest pixel in the frame"),
        ("temp_max",    "Max Temperature",     "number", "Hottest pixel in the frame"),
        ("temp_centre", "Centre Temperature",  "number", "Average of the 4 centre pixels"),
        ("temp_range",  "Temperature Range",   "json",   "Min/max C used for 8-bit mapping"),
    ]

    for name, label, field_type, help_text in fields:
        try:
            api.create_field(
                APP, ENDPOINT, name, label, field_type,
                help_text=help_text,
            )
        except MeowError:
            pass  # Already exists


def temp_to_rgb(temp):
    """
    Map a temperature value to an RGB colour using a thermal camera
    gradient:  blue → cyan → green → yellow → red.

    The gradient is split into 4 equal segments across the TEMP_MIN to
    TEMP_MAX range.  Each segment blends between two colours by ramping
    one channel up or down.
    """
    # Normalise temperature to 0.0–1.0
    ratio = (temp - TEMP_MIN) / (TEMP_MAX - TEMP_MIN)
    ratio = max(0.0, min(1.0, ratio))

    # Scale to 0–4 to pick which colour segment we're in
    segment = ratio * 4.0

    if segment <= 1.0:
        # Blue → Cyan:  red stays 0, green ramps up, blue stays 255
        r, g, b = 0, int(segment * 255), 255
    elif segment <= 2.0:
        # Cyan → Green:  red stays 0, green stays 255, blue ramps down
        r, g, b = 0, 255, int((2.0 - segment) * 255)
    elif segment <= 3.0:
        # Green → Yellow:  red ramps up, green stays 255, blue stays 0
        r, g, b = int((segment - 2.0) * 255), 255, 0
    else:
        # Yellow → Red:  red stays 255, green ramps down, blue stays 0
        r, g, b = 255, int((4.0 - segment) * 255), 0

    return (r, g, b)


def setup_glowbit():
    """
    Try to initialise the GlowBit 8x8 LED matrix.
    Returns the matrix object, or None if not available.
    Must be run as root (sudo) because WS2812B LEDs need hardware PWM.
    """
    if not GLOWBIT_AVAILABLE:
        return None

    try:
        matrix = glowbit.matrix8x8(pin=GLOWBIT_PIN, brightness=GLOWBIT_BRIGHTNESS)
        matrix.pixelsFill(0)
        matrix.pixelsShow()
        return matrix
    except Exception as e:
        print(f"GlowBit not available ({e}) — try running with sudo.")
        return None


def display_on_glowbit(matrix, pixels):
    """
    Map the 8x8 temperature grid directly onto the 8x8 LED matrix.
    Each LED gets a colour from the thermal gradient (blue=cold, red=hot).
    The glowbit library handles the pixel layout internally via pixelSetXY.
    """
    for row_idx, row in enumerate(pixels):
        for col_idx, temp in enumerate(row):
            r, g, b = temp_to_rgb(temp)
            # Pack RGB into the 24-bit integer format glowbit expects
            colour = (r << 16) | (g << 8) | b
            matrix.pixelSetXY(col_idx, row_idx, colour)

    matrix.pixelsShow()


def print_terminal_heatmap(pixels, temp_min, temp_max, temp_centre):
    """
    Print an ASCII heat map to the terminal.  Each temperature value is
    mapped to a shade character — spaces for cold, @ for hot.
    """
    # Move cursor up to overwrite the previous frame (8 data rows + 4 info lines)
    print("\033[12A", end="")

    for row in pixels:
        line = ""
        for temp in row:
            ratio = (temp - TEMP_MIN) / (TEMP_MAX - TEMP_MIN)
            ratio = max(0.0, min(1.0, ratio))
            idx = int(ratio * (len(SHADE_CHARS) - 1))
            line += SHADE_CHARS[idx] * 2
        print(line)

    print(f"  Hi {temp_max:.1f} C | Lo {temp_min:.1f} C | Ctr {temp_centre:.1f} C")
    print(f"  Range: {TEMP_MIN} C (space) to {TEMP_MAX} C (@)")
    print()
    print()


def setup_oled():
    """
    Try to initialise the OLED display.  Returns (device, font) if
    successful, or (None, None) if the display is not available.
    """
    if not OLED_AVAILABLE:
        return None, None

    try:
        serial = luma_i2c(port=1, address=0x3C)
        device = sh1106(serial, width=OLED_WIDTH, height=OLED_HEIGHT)
    except Exception as e:
        print(f"OLED not found ({e}) — using terminal output instead.")
        return None, None

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9
        )
    except OSError:
        font = ImageFont.load_default()

    return device, font


def display_on_oled(device, font, pixels, temp_min, temp_max, temp_centre):
    """Render the heat map and temperature stats on the OLED screen."""
    arr = np.array(pixels, dtype=np.float32)
    arr = np.clip(arr, TEMP_MIN, TEMP_MAX)
    arr = ((arr - TEMP_MIN) / (TEMP_MAX - TEMP_MIN) * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    heatmap = img.resize((HEATMAP_SIZE, HEATMAP_SIZE), Image.BILINEAR)

    canvas = Image.new("L", (OLED_WIDTH, OLED_HEIGHT), 0)
    draw = ImageDraw.Draw(canvas)

    y_offset = (OLED_HEIGHT - HEATMAP_SIZE) // 2
    canvas.paste(heatmap, (2, y_offset))

    draw.rectangle(
        [1, y_offset - 1, HEATMAP_SIZE + 2, y_offset + HEATMAP_SIZE],
        outline=255,
    )

    text_x = HEATMAP_SIZE + 8
    draw.text((text_x, 4),  f"Hi {temp_max:.1f}", fill=255, font=font)
    draw.text((text_x, 18), f"Lo {temp_min:.1f}", fill=255, font=font)
    draw.text((text_x, 36), "Ctr",               fill=255, font=font)
    draw.text((text_x, 48), f"{temp_centre:.1f}C", fill=255, font=font)

    device.display(canvas)


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------

def main():
    # --- API setup ---
    if not API_DISABLED and API_KEY:
        api = Meow(api_key=API_KEY)
        setup_endpoint(api)
    else:
        api = None
        if API_DISABLED:
            print("API disabled — running in local-only mode.")
        else:
            print("No MEOW_API_KEY set — running in local-only mode.")

    # --- Sensor setup ---
    # I2C ("Inter-Integrated Circuit") is a two-wire protocol that lets
    # multiple devices share the same pair of wires.  board.SCL is the
    # clock line and board.SDA is the data line on the Pi's GPIO header.
    i2c_bus = busio.I2C(board.SCL, board.SDA)

    # Create the AMG8833 sensor object.  The sensor has an 8x8 grid of tiny
    # infrared thermometers that each measure the temperature of whatever
    # they are pointing at.  AMG_ADDRESS is set in the config section above.
    sensor = adafruit_amg88xx.AMG88XX(i2c_bus, addr=AMG_ADDRESS)

    # --- Display setup (optional) ---
    strip = setup_glowbit()
    device, font = setup_oled()
    use_oled = device is not None

    if strip:
        print("Thermal camera running (GlowBit) — press Ctrl+C to stop.")
    if use_oled:
        print("Thermal camera running (OLED) — press Ctrl+C to stop.")
    if not strip and not use_oled:
        print("Thermal camera running (terminal) — press Ctrl+C to stop.")
        print(f"Display range: {TEMP_MIN} C (space) to {TEMP_MAX} C (@)")
        # Print blank lines so the first frame has something to overwrite
        print("\n" * 11, end="")

    frame_count = 0

    try:
        while True:
            # Read the 8x8 grid of temperatures (each value is a float in C)
            pixels = sensor.pixels

            # Calculate summary stats
            flat = [temp for row in pixels for temp in row]
            temp_min = min(flat)
            temp_max = max(flat)
            temp_centre = (
                pixels[3][3] + pixels[3][4] + pixels[4][3] + pixels[4][4]
            ) / 4.0

            # --- Send frame to meow meow scratch ---
            if api:
                frame = build_frame(pixels)
                try:
                    api.send(APP, ENDPOINT, frame)
                    frame_count += 1
                except MeowError as e:
                    print(f"Send failed: {e}")

            # Rotate the grid for display (doesn't affect the raw data sent to API)
            display_pixels = orient_grid(pixels)

            # --- Update local displays ---
            if strip:
                display_on_glowbit(strip, display_pixels)
            if use_oled:
                display_on_oled(device, font, display_pixels, temp_min, temp_max, temp_centre)
            if not strip and not use_oled:
                print_terminal_heatmap(display_pixels, temp_min, temp_max, temp_centre)

            time.sleep(SEND_INTERVAL)

    except KeyboardInterrupt:
        if strip:
            strip.pixelsFill(0)
            strip.pixelsShow()
        if use_oled:
            device.hide()
        if api:
            print(f"\nStopped. Sent {frame_count} frames to {APP}/{ENDPOINT}.")
        else:
            print("\nStopped.")


if __name__ == "__main__":
    main()
