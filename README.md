# Pi Thermal Camera

Turn your Raspberry Pi into a thermal camera! This project reads an 8x8 grid of temperature data from an AMG8833 infrared sensor and displays a live colour heat map on a GlowBit 8x8 LED matrix — blue for cold, red for hot. Auto-range "predator vision" mode stretches contrast to the actual temperatures in each frame, so you can spot a person at up to 7 metres. An optional HC-SR04 ultrasonic sensor measures the distance to whatever is in front of the camera. Each frame is also streamed to meow meow scratch with an 8-bit version ready for machine learning models.

## What you'll learn
- How I2C communication works (two devices sharing the same two wires)
- Reading temperature data from an infrared sensor
- Driving a WS2812B LED matrix with colour-mapped data
- Measuring distance with an ultrasonic sensor (HC-SR04)
- Auto-ranging a display for maximum contrast ("predator vision")
- Sending sensor data to an API for storage and analysis
- Mapping temperature values to 8-bit integers for ML model input

## What you'll need

### Hardware
- **Raspberry Pi Zero W** (or any Pi with GPIO pins)
- **AMG8833 thermal sensor** — a small board with an 8x8 grid of infrared thermometers. Each one measures the temperature of whatever it's pointed at, giving you 64 temperature readings at once. Detects 0–80°C at up to ~7 metres.
- **GlowBit Matrix 8x8** (Core Electronics) — an 8x8 grid of individually addressable RGB LEDs (WS2812B). Each LED can display any colour, so it maps perfectly to the 8x8 sensor grid. Uses a single data wire.
- **HC-SR04 ultrasonic distance sensor** *(optional)* — sends out a burst of ultrasound and listens for the echo to measure how far away something is (2 cm to ~4 m). Great for knowing how far a detected person is from the camera.
- **1.3" OLED display** *(optional)* (Duinotech v2.0 or similar SH1106-based OLED) — a tiny monochrome screen. If connected, it shows a smooth greyscale heat map alongside temperature stats.
- **2 resistors** *(for HC-SR04)* — a 1 k&Omega; and a 2 k&Omega; resistor to make a voltage divider. The HC-SR04's ECHO pin outputs 5 V, but the Pi's GPIO pins only tolerate 3.3 V. The voltage divider brings it down to a safe level.
- **Jumper wires** — short cables that connect components together without soldering.
- **Breadboard** *(optional)* — handy for splitting shared I2C pins out to multiple devices.

### Software
- Python 3 (comes pre-installed on your Pi)
- A free meow meow scratch account — sign up at meowmeowscratch.com

## Wiring diagram

The AMG8833 sensor uses **I2C** (pronounced "eye-squared-see"), a two-wire protocol. The GlowBit matrix uses a single **data pin** (GPIO 18). They don't share any wires.

```
Raspberry Pi            AMG8833 Sensor
──────────────          ──────────────
Pin 1  (3.3V) ────────── VIN
Pin 6  (GND)  ────────── GND
Pin 3  (SDA)  ────────── SDA
Pin 5  (SCL)  ────────── SCL

Raspberry Pi            GlowBit Matrix 8x8
──────────────          ───────────────────
Pin 2  (5V)   ────────── 5V
Pin 14 (GND)  ────────── GND
Pin 12 (GPIO 18) ─────── DIN

Raspberry Pi            HC-SR04 (with voltage divider on ECHO)
──────────────          ──────────────────────────────────────
Pin 2  (5V)   ────────── VCC
Pin 16 (GPIO 23) ─────── TRIG
Pin 22 (GPIO 25) ──┬──── ECHO (through voltage divider — see below)
Pin 20 (GND)  ────────── GND

Voltage divider for ECHO:
    ECHO pin ──── 1kΩ ──┬── GPIO 25
                        │
                       2kΩ
                        │
                       GND
```

### AMG8833 sensor wiring

| AMG8833 pin | Raspberry Pi pin | What it does |
|---|---|---|
| VIN | Pin 1 (3.3V Power) | Powers the sensor |
| GND | Pin 6 (Ground) | Completes the circuit |
| SDA | Pin 3 (GPIO 2 / SDA) | I2C data line |
| SCL | Pin 5 (GPIO 3 / SCL) | I2C clock line |

### GlowBit matrix wiring

| GlowBit pin | Raspberry Pi pin | What it does |
|---|---|---|
| 5V | Pin 2 (5V Power) | Powers the LEDs — use 5V, not 3.3V |
| GND | Pin 14 (Ground) | Completes the circuit |
| DIN | Pin 12 (GPIO 18) | Receives colour data for the LEDs |

> **Important:** Power the GlowBit from the Pi's **5V** pin (pin 2 or 4), not 3.3V. The LEDs need 5V to display colours correctly. Make sure the data wire goes to **DIN** (data in), not DOUT (data out).

### Optional HC-SR04 wiring

The HC-SR04 ultrasonic distance sensor measures how far away objects are. It needs a **voltage divider** on the ECHO pin because it outputs 5 V but the Pi's GPIO pins can only handle 3.3 V.

| HC-SR04 pin | Raspberry Pi pin | What it does |
|---|---|---|
| VCC | Pin 2 (5V Power) | Powers the sensor — needs 5 V |
| TRIG | Pin 16 (GPIO 23) | Trigger — the Pi sends a short pulse to start a measurement |
| ECHO | Pin 22 (GPIO 25) via voltage divider | Echo — goes HIGH for a duration proportional to the distance |
| GND | Pin 20 (Ground) | Completes the circuit |

> **Important:** Do NOT connect the ECHO pin directly to GPIO 25. Use a voltage divider (1 k&Omega; between ECHO and GPIO 25, 2 k&Omega; between GPIO 25 and GND) to drop the 5 V signal to ~3.3 V. Without this, you risk damaging your Pi.

### Optional OLED wiring

If you also have a 1.3" OLED, connect it to the same I2C bus as the sensor:

| OLED pin | Raspberry Pi pin |
|---|---|
| VCC | Pin 1 (3.3V) |
| GND | Pin 6 (GND) |
| SDA | Pin 3 (SDA) |
| SCL | Pin 5 (SCL) |

## Step-by-step setup

### 1. Enable I2C on your Pi

I2C is turned off by default. You need to enable it once:

```bash
sudo raspi-config
```

Navigate to **Interfacing Options → I2C → Yes**, then reboot:

```bash
sudo reboot
```

After rebooting, check that the Pi can see the sensor:

```bash
sudo apt install -y i2c-tools
i2cdetect -y 1
```

You should see **0x69** (the AMG8833). If you also have the OLED connected, you'll see **0x3c** too. If an expected address is missing, double-check your wiring.

### 2. Install system dependencies

```bash
sudo apt install -y libopenblas0 fonts-dejavu
```

### 3. Set up the project

```bash
cd ~/projects/pi-thermal-camera
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

This installs:

| Library | What it does |
|---|---|
| `meow-sdk` | Talks to the meow meow scratch API |
| `python-dotenv` | Loads your API key from the `.env` file |
| `RPi.GPIO` | Low-level access to the Pi's GPIO pins |
| `adafruit-circuitpython-amg88xx` | Reads temperature data from the AMG8833 sensor |
| `glowbit` | Core Electronics driver for the GlowBit LED matrix |
| `luma.oled` | Draws graphics on the OLED screen (optional) |
| `Pillow` | Creates and manipulates images in Python |
| `numpy` | Fast number crunching for the temperature grid |

### 4. Get your API key

1. Go to **meowmeowscratch.com** and create a free account (or log in).
2. Open your account settings and find your **API key** — a long string of letters and numbers that proves it's really you.
3. Copy the key.

### 5. Set your API key

Create a file called `.env` in the project folder. This keeps your key out of your code and out of git (it's in `.gitignore`).

```bash
echo 'MEOW_API_KEY=paste-your-key-here' >> .env
```

The script uses `python-dotenv` to read this file automatically when it starts.

> **Which kind of key should you use?** Your account offers two. A **platform token** works across every app you own. An **app API key** works for one app only. For a Pi that sits running for days, use an **app API key** — if it ever leaks, only this one app is affected, not your whole account. You'll find both in your account settings.

### 6. Create your app on meow meow scratch

1. Log in to **meowmeowscratch.com**.
2. Create a new app called **heat**.

The script will automatically create the **thermal** collection endpoint and its fields on first run — you don't need to set those up manually.

### 7. Run it

The GlowBit needs hardware PWM access, so you must run with `sudo`:

```bash
sudo venv/bin/python3 thermal_camera.py
```

Point the sensor at something — your hand, a cup of tea, a wall — and watch the GlowBit light up with a thermal colour gradient. Press **Ctrl+C** to stop.

## Configuration

All settings are at the top of `thermal_camera.py`:

| Setting | Default | Description |
|---|---|---|
| `API_DISABLED` | `True` | Set to `False` to send frames to meow meow scratch |
| `TEMP_MIN` | `20.0` | Temperature (°C) that maps to blue / black (used for API 8-bit mapping and when `AUTO_RANGE` is off) |
| `TEMP_MAX` | `35.0` | Temperature (°C) that maps to red / white |
| `AUTO_RANGE` | `True` | Stretches display contrast to each frame's actual min/max — "predator vision" mode for spotting warm bodies at distance |
| `AUTO_RANGE_MIN_SPAN` | `2.0` | Minimum temperature span (°C) in auto-range mode. Lower = more sensitive but more flicker |
| `AMG_ADDRESS` | `0x69` | I2C address of the AMG8833 (some boards use `0x68`) |
| `SEND_INTERVAL` | `0.1` | Seconds between readings (0.1 = ~10 fps) |
| `GLOWBIT_PIN` | `18` | GPIO pin connected to the GlowBit DIN |
| `GLOWBIT_BRIGHTNESS` | `40` | LED brightness (0–255) |
| `ROTATION` | `0` | Rotate the display 0/1/2/3 × 90° clockwise |
| `FLIP_VERTICAL` | `False` | Flip the display top-to-bottom |
| `FLIP_HORIZONTAL` | `True` | Flip the display left-to-right |
| `DISTANCE_ENABLED` | `True` | Set to `False` if no HC-SR04 is connected |
| `TRIG_PIN` | `23` | GPIO pin for the HC-SR04 trigger |
| `ECHO_PIN` | `25` | GPIO pin for the HC-SR04 echo (through voltage divider) |

## Display modes

The script auto-detects what's connected and uses the best available display:

| Priority | Display | What it looks like |
|---|---|---|
| 1 | **GlowBit 8x8** | Full-colour thermal heat map (blue→cyan→green→yellow→red) |
| 2 | **1.3" OLED** | Greyscale heat map with Hi/Lo/Ctr temperature stats |
| 3 | **Terminal** | ASCII heat map using shade characters (space→`@`) |

Multiple displays can run at the same time — e.g. GlowBit + OLED together.

## What gets sent to meow meow scratch

When `API_DISABLED = False`, each frame is sent as a JSON object:

| Field | Type | Description |
|---|---|---|
| `pixels_raw` | list of 64 floats | The raw temperature readings in °C, row by row |
| `pixels_8bit` | list of 64 ints (0–255) | Temperatures mapped to 8-bit values — ready for ML model input |
| `temp_min` | float | Coldest pixel in the frame |
| `temp_max` | float | Hottest pixel in the frame |
| `temp_centre` | float | Average of the 4 centre pixels |
| `temp_range` | [float, float] | The min/max °C used for the 8-bit mapping |
| `distance_cm` | float | Distance from the HC-SR04 ultrasonic sensor in cm (omitted if no sensor) |

The `pixels_8bit` field is designed for direct use as input to an 8-bit neural network (e.g. Akida) for tasks like person detection.

## How the code works

1. **Load config** — reads the API key from `.env` and connects to meow meow scratch (if enabled).
2. **Set up I2C** — opens the two-wire I2C bus on the Pi.
3. **Initialise the AMG8833** — connects to the thermal sensor at its I2C address.
4. **Initialise displays** — tries GlowBit (GPIO 18), then OLED (I2C 0x3C), falls back to terminal.
5. **Initialise HC-SR04** — sets up the TRIG and ECHO GPIO pins for distance measurement (if enabled).
5. **Main loop** (runs ~10 times per second):
   - Reads 64 temperature values (an 8x8 grid) from the sensor.
   - Reads the distance from the HC-SR04 (if connected).
   - Computes the effective display range — in auto-range mode, this stretches to the frame's own min/max for maximum contrast.
   - Builds a frame with raw temps, 8-bit values, distance, and summary stats.
   - Sends the frame to meow meow scratch (if API enabled).
   - Applies rotation and flip settings to the grid.
   - Maps each temperature to a colour and updates the display.
   - Prints a live one-line status to the terminal (temp + distance) even when GlowBit or OLED is the main display.
6. **Clean exit** — when you press Ctrl+C, LEDs are turned off, GPIO is cleaned up, and it reports how many frames were sent.

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `No I2C device at address: 0x69` | Sensor not wired or I2C not enabled | Check wiring and run `sudo raspi-config` to enable I2C |
| GlowBit shows nothing | Not running as root | Run with `sudo venv/bin/python3 thermal_camera.py` |
| GlowBit shows only 1 LED | Data wire on wrong pad, or bad solder | Check DIN (not DOUT), check 5V power, resolder headers |
| GlowBit shows wrong colours | Pixel order mismatch | The `glowbit` library handles this — make sure it's installed |
| `Send failed: ...` | API key wrong or app not created | Check `MEOW_API_KEY` in `.env` and create the app on meowmeowscratch.com |
| Heat map looks mirrored | Sensor/display orientation | Adjust `FLIP_VERTICAL`, `FLIP_HORIZONTAL`, or `ROTATION` in config |
| Heat map is all one colour | Temperature range doesn't match the scene | Enable `AUTO_RANGE = True`, or narrow `TEMP_MIN`/`TEMP_MAX` to match the scene |
| Heat map flickers wildly | `AUTO_RANGE_MIN_SPAN` is too small | Increase `AUTO_RANGE_MIN_SPAN` (try 3.0 or 4.0) |
| Distance shows `--` | Nothing in range or HC-SR04 not wired | Check wiring, make sure ECHO goes through the voltage divider |
| Distance readings are wrong | Voltage divider missing or wrong resistors | Use 1 k&Omega; + 2 k&Omega;. Without the divider, the 5 V signal can give bad readings or damage the Pi |
| `ModuleNotFoundError: No module named 'RPi'` | RPi.GPIO not installed | Run `pip install RPi.GPIO` |
| `ModuleNotFoundError: No module named '_rpi_ws281x'` | WS281x driver missing | Run `pip install rpi_ws281x` |
| `libopenblas.so.0: cannot open shared object file` | System library missing | Run `sudo apt install -y libopenblas0` |
| OLED stays blank | Wrong display driver | Change `sh1106` to `ssd1306` in the code |
