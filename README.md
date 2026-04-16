# Skin Tone Prediction (Frontend)

## Overview
This project is a browser-based facial analysis application built using MediaPipe Tasks Vision. It performs real-time face analysis directly in the browser using WebAssembly, with support for both image uploads and live webcam input.

The application focuses on feature extraction and visualization by integrating multiple external models and heuristic methods into a unified interface.

Note: This project uses outputs from external models (e.g., MediaPipe and other vision models). The predictions shown are derived from these tools and are not original machine learning models developed in this project.

## Features

### Face Detection
- Uses BlazeFace (short-range model)
- Runs continuously on each frame
- Displays detection confidence and face count

### Facial Landmarks and Expressions
- Detects 478 facial landmarks
- Extracts 52 blendshape values
- Updates expression data in real time

### Skin Tone Analysis
- Fitzpatrick scale classification
- Undertone estimation
- ITA (Individual Typology Angle) score visualization
- Dynamic color swatch

Note: Skin tone values are computed using heuristic and model-based outputs and may vary depending on lighting and model behavior.

### Scene Temperature
- Estimates lighting temperature from image data
- Displays value on a gradient scale

### Face Shape Detection
- Uses landmark geometry ratios
- Outputs face shape classification
- Displays measurement data

### Ornament Detection
- Detects accessories such as glasses, earrings, necklace, and bindi
- Displays results as tags

### Camera Distance Tracking
- Based on interpupillary distance (IPD)
- Shows distance status (close, optimal, far)
- Adjustable range using sliders

### Mesh Overlay Modes
- Full
- Contour
- Minimal
- Off

### Deep Analysis (Backend Integration)
- Captures current frame
- Sends image to backend for advanced analysis
- Displays results in a modal

Note: Backend analysis may rely on external models (e.g., vision-language models or LLMs). Outputs depend on those systems.

## Project Structure

```
frontend/
├── index.html
├── css/
│   └── styles.css
└── js/
    ├── imports.js
    └── script.js
```

## How It Works

1. MediaPipe models are loaded through `imports.js`
2. User selects input mode (upload or webcam)
3. Face detection runs first
4. If a face is detected:
   - Landmark detection is executed
   - Feature extraction is performed
5. UI updates continuously based on results

## Input Modes

### Upload
- Accepts image files
- Processes a single frame

### Webcam
- Uses live video stream
- Processes frames continuously
- Allows capture for analysis

## Rendering

- `<video>` is used for webcam input
- `<canvas>` is used for processed output

UI overlays display:
- Face mesh
- Detection indicators
- Distance tracking

## Tech Stack

- HTML, CSS, JavaScript (no frameworks)
- MediaPipe Tasks Vision
- WebAssembly (WASM)

## Notes

- No frameworks or build tools are used
- Runs entirely in the browser
- Designed for low-latency real-time processing
- Relies on external ML models for predictions
- Focus is on integration, visualization, and user experience rather than model development

## Run Frontend

```bash
python -m http.server 8080
```

Open in browser:
```
http://localhost:8080/frontend/
```
