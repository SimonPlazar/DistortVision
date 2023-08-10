# Video Processing App

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Distortion Options](#distortion-options)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

The Video Processing App is a graphical user interface (GUI) application built using Python and various libraries such as OpenCV, tkinter, and YOLO. This app allows you to process real-time video streams by applying object detection using YOLO and various distortion effects to the detected objects.

## Features

- Real-time video processing and distortion.
- Instance segmentation using YOLO model.
- Multiple distortion options including pixelation, JPEG artifact, and more.
- Adjustable video resolution, torch device selection and more.

## Getting Started

### Prerequisites

Before using the app, ensure you have the following prerequisites installed:

- Python (>= 3.6)
- OpenCV (cv2)
- tkinter
- torch
- ultralytics (YOLO library)
- numpy
- PIL

You can install the required packages using the following command:

```bash
pip install opencv-python tkinter torch ultralytics numpy pillow
```

### Installation

1. Clone this repository to your local machine using:

```bash
git clone https://github.com/yourusername/video-processing-app.git
```

2. Navigate to the cloned directory:

```bash
cd video-processing-app
```

3. Run the application:

```bash
python app.py
```

## Usage

1. Upon launching the application, you will be presented with a GUI where you can select various settings, including video resolution, torch device, distortion options, and YOLO weights.
2. Click the "Start" button to initiate the video processing.
3. The app will capture the webcam video stream and process it in real-time using the chosen settings.
4. Press the 'q' key to exit the application.

## Distortion Options

- **None**: No distortion is applied, and the video stream is displayed as is.
- **Jpeg**: Apply JPEG compression artifact distortion to the video stream.
- **Censoring**: Pixelate the detected objects to censor them.
- **Pink**: Replace detected objects with a pink color.

## Contributing

Contributions are welcome! If you find any bugs or want to improve the app, feel free to create issues or pull requests.

## Contact

If you have any questions or suggestions, please feel free to contact [Simon Plazar](mailto:simon.plazar@gmail.com).