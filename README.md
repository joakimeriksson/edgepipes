# Edgepipes - processing pipeline for Python and Edge AI/ML

Experimental pipeline processing for Python. Intended for developing basic AI or IoT pipelines for testing ideas and developing PoC. The idea is to have something that reminds of Google Mediapipes for edge computer vision and audio pipelines but implemented in plain Python rather than in C++. The pipeline configuration format is heavily inspired by Mediapipes.

## Installing prerequisites

You need Python 3, Package Installer for Python (PIP) and PyAudio.

### Linux

```
> sudo apt update
> sudo apt install python3 python3-pip
> sudo apt install python3-pyaudio
```

### OSX with Homebrew

```
> # If xcode not installed before (needed for portaudio)
> xcode-select --install

> # If Python 3 not already installed
> brew install python@3.8

> brew install portaudio
> pip3 install pyaudio
```

### Other operating systems

Please search online for instructions for your operating system.

### Virtual environments

It is highly recommended to use Python virtual environments to enable separated environments for different Python projects.

```
> pip3 install virtualenv
> pip3 install virtualenvwrapper
```
Some systems might require extra installation steps. Please search online for instructions for your operating system.

Create virtual environment for Edgepipes.

```
> mkvirtualenv edgepipes -p python3
> workon edgepipes
```

## Install required Python modules

```
> cd edgepipes
> pip3 install -r requirements.txt
```

Some examples have additional requirements. To run the face recognition example:
```
> pip3 install -r face/requirements.txt
```

## Run Edgepipes

Edgepipes are run with computer vision example by default (for now)
```
> ./edgepipes.py graphs/edge_detection.pbtxt
```

This will run the same graph as is available in Google's Mediapipes as an example of the similarities.
Press 'q' in the popup window to quit the example.

To specify the input video source:
```
> # First video device
> ./edgepipes.py graphs/edge_detection.pbtxt --input 0
> # Use IP camera via RTSP
> ./edgepipes.py graphs/edge_detection.pbtxt --input rtsp://192.168.1.237:7447/5c8d2bf990085177ff91c7a2_2
> # Screen capture (first monitor)
> ./edgepipes.py graphs/edge_detection.pbtxt --input screen
```
The input argument (except for `screen`) will be passed directly to OpenCV `VideoCapture` function. Please see OpenCV documentation for more options.

### Included Examples

All examples use the default camera unless told otherwise using the `--input` argument.
Press 'q' in the popup window to quit the example.

Edge detection
```
> ./edgepipes.py graphs/edge_detection.pbtxt
```
Yolo 3 detector
```
> ./edgepipes.py graphs/yolov3detector.pbtxt
```
Hand tracker
```
> ./edgepipes.py graphs/handtracker.pbtxt
```
Face recognition - more images of "known" people can be added in the directory `edgepipes/face/known`
```
> ./edgepipes.py graphs/face_recognition.pptxt
```

Yolo 3 detector publishing detections via MQTT. This example requires a running MQTT broker in local computer.
```
> ./edgepipes.py graphs/yolov3detector_mqtt.pbtxt
```

## Interactive CLI Version
Another option to start Edgepipes is to run it from an interactive CLI.
```
> ./pipecli.py
```

This will allow you to play with loading pipeline, starting stopping and printing the pipeline. Future features will be to add and remove parts of the pipeline at runtime, run at different speeds, debug, etc.

## Future features and ideas
* Add a way to distribute the pipeline processing over multiple threads and machines.
* Add a way to send messages over MQTT instead of passing results internally in Python.
* Serialize messages in Protobufs between processes (over network).
* Allow same config to run in different modes (e.g. local in single process or over MQTT with Protobufs) without
massive configuration change.
