# Edgepipes - processing pipeline for python and Edge AI/ML
Experimental pipeline processing for Python. Intended for developing basic AI or IoT pipelines for testing ideas and developing PoC. The idea is to have something that reminds of Google Mediapipes for edge computer vision and audio pipelines but implemented in plain python rather than in C++. The pipeline configuration format is heavily inspired by Mediapipes.

## Run Edgepipes
Edgepipes are run with computer vision example by default (for now)
>python3 edgepipes.py graphs/edge_detection.pbtxt

This will run the same graph as is available in googles mediapipes as an example of the similarities.

## Interactive CLI Version
Another option to start datapipes is to run it from an interactive cli.
>python3 pipecli.py

This will allow you to play with loading pipeline, starting stopping and printing the pipeline. Future features will be to add and remove parts of the pipeline at runtime, run at different speeds, debug, etc.

## Future features and ideas
* Add a way to distribute the pipeline processing over multiple threads and machines.
* Add a way to send messages over MQTT instead of passing results internally in python
* Serialize messages in protobufs between processes (over network)
* Allow same config to run in different modes (e.g. local in single process or over MQTT with protobufs) without
massive configuration change
