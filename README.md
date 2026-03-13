# Pi Thermal Camera

Read temperature data from an AMG8833 infrared sensor and stream it to meow meow scratch. Each frame includes the raw 8x8 temperature grid and an 8-bit version ready for machine learning models. If you have a 1.3" OLED screen it shows a live heat map; otherwise it prints an ASCII heat map in the terminal.

## What you'll learn
- How I2C communication works (two devices sharing the same two wires)
- Reading temperature data from an infrared sensor
- Sending sensor data to an API for storage and analysis
- Mapping temperature values to 8-bit integers for ML model input

## What you'll need

### Hardware
- **Raspberry Pi Zero W** (or any Pi with GPIO pins)
- **AMG8833 thermal sensor** — a small board with an 8x8 grid of infrared thermometers. Each one measures the temperature of whatever it's pointed at, giving you 64 temperature readings at once.
- **1.3" OLED display** *(optional)* (Duinotech v2.0 or similar SH1106-based OLED) — a tiny screen that shows crisp white graphics on a black background. If you don't have one, the script prints an ASCII heat map to the terminal instead.
- **Jumper wires** — short cables that connect components together without soldering.

### Software
- Python 3 (comes pre-installed on your Pi)
- A free meow meow scratch account — sign up at meowmeowscratch.com

## Wiring diagram

Both the sensor and the display use **I2C** (pronounced "eye-squared-see"), a communication protocol that only needs two data wires. Because each device has its own unique address, they can share the same wires.

> **No OLED?** Just wire the AMG8833 sensor (VIN, GND, SDA, SCL) and skip the display — the script will automatically fall back to terminal output.

```
Raspberry Pi            AMG8833 Sensor        OLED Display (optional)
──────────────          ──────────────        ──────────────
Pin 1  (3.3V) ────────── VIN ──────────────── VCC
Pin 6  (GND)  ────────── GND ──────────────── GND
Pin 3  (SDA)  ────────── SDA ──────────────── SDA
Pin 5  (SCL)  ────────── SCL ──────────────── SCL
```

| Component pin | Raspberry Pi pin | What it does |
|---|---|---|
| VIN / VCC | Pin 1 (3.3V Power) | Provides power to the component |
| GND | Pin 6 (Ground) | Completes the electrical circuit |
| SDA | Pin 3 (GPIO 2 / SDA) | Carries the data for I2C communication |
| SCL | Pin 5 (GPIO 3 / SCL) | Carries the clock signal that keeps I2C in sync |

> **Tip:** Both devices connect to the *same* four Pi pins. You can use a small breadboard to split each pin out to both devices, or daisy-chain the wires.

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

### 2. Install the required libraries

`pip` is Python's package installer — it downloads libraries from the internet and sets them up for you.

```bash
pip install -r requirements.txt
```

This installs:

| Library | What it does |
|---|---|
| `meow-sdk` | Talks to the meow meow scratch API |
| `python-dotenv` | Loads your API key from the `.env` file |
| `RPi.GPIO` | Low-level access to the Pi's GPIO pins |
| `adafruit-circuitpython-amg88xx` | Reads temperature data from the AMG8833 sensor |
| `luma.oled` | Draws graphics on the OLED screen (optional) |
| `Pillow` | Creates and manipulates images in Python |
| `numpy` | Fast number crunching for the temperature grid |

You may also need to install a system library and font:

```bash
sudo apt install -y libopenblas0 fonts-dejavu
```

### 3. Get your API key

1. Go to **meowmeowscratch.com** and create a free account (or log in).
2. Open your account settings and find your **API key** — a long string of letters and numbers that proves it's really you.
3. Copy the key — you'll need it in the next step.

### 4. Set your API key

Create a file called `.env` in the project folder. This keeps your key out of your code and out of git (it's in `.gitignore`).

```bash
echo 'MEOW_API_KEY=paste-your-key-here' >> .env
```

The script uses `python-dotenv` to read this file automatically when it starts.

### 5. Create your app on meow meow scratch

1. Log in to **meowmeowscratch.com**.
2. Create a new app called **heat**.

The script will automatically create the **thermal** collection endpoint and its fields on first run — you don't need to set those up manually.

### 6. Run it

Point the sensor at something (your hand, a cup of tea, a window) and start the script:

```bash
python3 thermal_camera.py
```

You'll see a heat map update in the terminal (or on the OLED if connected), and each frame is sent to meow meow scratch. Press **Ctrl+C** to stop — it will tell you how many frames were sent.

## What gets sent to meow meow scratch

Each frame is a JSON object with:

| Field | Type | Description |
|---|---|---|
| `pixels_raw` | list of 64 floats | The raw temperature readings in °C, row by row |
| `pixels_8bit` | list of 64 ints (0–255) | Temperatures mapped to 8-bit values — ready for ML model input |
| `temp_min` | float | Coldest pixel in the frame |
| `temp_max` | float | Hottest pixel in the frame |
| `temp_centre` | float | Average of the 4 centre pixels |
| `temp_range` | [float, float] | The min/max °C used for the 8-bit mapping |

The `pixels_8bit` field is designed for direct use as input to an 8-bit neural network (e.g. Akida) for tasks like person detection.

## How the code works

1. **Connect to the API** — authenticates with meow meow scratch using your API key.
2. **Set up I2C** — opens the two-wire I2C bus on the Pi.
3. **Initialise the AMG8833** — connects to the thermal sensor at its I2C address.
4. **Try to initialise the OLED** — if the display is connected and the libraries are installed, it sets up the screen. Otherwise it switches to terminal mode.
5. **Main loop** (runs every second):
   - Reads 64 temperature values (an 8x8 grid) from the sensor.
   - Builds a frame with raw temps, 8-bit values, and summary stats.
   - Sends the frame to meow meow scratch.
   - Updates the local display (OLED or terminal).
6. **Clean exit** — when you press Ctrl+C, it reports how many frames were sent.

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `No I2C device at address: 0x69` | Sensor not wired correctly or I2C not enabled | Check wiring and run `sudo raspi-config` to enable I2C |
| `No I2C device at address: 0x3c` | OLED not wired correctly | Check the four wires — or just skip the OLED |
| OLED stays blank but no errors | Wrong display driver | Change `sh1106` to `ssd1306` in `thermal_camera.py` |
| `Send failed: ...` | API key wrong or app/endpoint not created | Check `MEOW_API_KEY` and create the app on meowmeowscratch.com |
| Heat map is all black | Temperature range too high for the scene | Lower `TEMP_MAX` (e.g. to 30) |
| Heat map is all white | Temperature range too low | Raise `TEMP_MAX` |
| `ModuleNotFoundError: No module named 'RPi'` | RPi.GPIO not installed | Run `pip install RPi.GPIO` |
| `libopenblas.so.0: cannot open shared object file` | System library missing | Run `sudo apt install -y libopenblas0` |
| `i2cdetect` shows nothing | I2C not enabled or bad wiring | Re-run `raspi-config`, check wires |
| Image is upside down | Sensor or display is rotated | Add `device.rotate(2)` after the `device = sh1106(...)` line |
