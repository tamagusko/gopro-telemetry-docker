# GoPro Telemetry Extractor (Docker)

This project provides a Docker-based tool to extract telemetry data (GPS, Accelerometer, Gyroscope) from GoPro `.mp4` videos using the `py-gpmf-parser` library. The extracted data is saved as a single JSON file per video.

## Requirements

- Docker ([installation instructions](https://www.docker.com/get-started/))
- GoPro video files in `.mp4` format

## Folder Structure

```
gopro-telemetry-docker/
├── Dockerfile
├── extract.py
├── input/           # Place GoPro videos here
└── output/          # Extracted JSON files will be saved here
```

## Installation

Clone the repository and move into the project directory:

```
git clone https://github.com/tamagusko/gopro-telemetry-docker.git
cd gopro-telemetry-docker
```

Place your `.mp4` video files into the `input/` folder.

## Build the Docker Image

```
docker build -t gopro-telemetry .
```

## Run the Extraction

```
docker run --rm \
  -v "$PWD/input":/data/input \
  -v "$PWD/output":/data/output \
  gopro-telemetry
```

This command will process all `.mp4` files inside the `input/` folder and generate one `.json` file per video inside the `output/` folder.

## Output

For a file named `GX010001_PhoenixPark.mp4`, you will get:

```
output/GX010001_PhoenixPark.json
```

The JSON contains telemetry streams such as `GPS9`, `ACCL`, and `GYRO`.
