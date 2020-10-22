# Installation on NVIDIA Jetson Nano

1. Download image with JetPack 4.4: https://developer.nvidia.com/embedded/jetpack

2. Burn the image to a SD-card and boot the NVIDIA Jetson Nano with the image.
Please enter desired keyboard layout, user name, etc., when asked.
For detailed instructions please see: https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit

3. Setup system

   Add CUDA to command path by adding following lines to `.bashrc`:

   ```
   > cat >>~/.bashrc
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

   Press `CTRL-D` to close the file.

4. Install system packages

   ```
   > sudo apt update
   > sudo apt upgrade
   > sudo apt install python3-pip
   > sudo apt install python3-pyaudio
   > sudo apt install cython3
   > sudo apt install libcanberra-gtk-module
   ```

   Optionally remove some packets to reduce disk usage if you have a small SD-card.

   ```
   > sudo apt purge libreoffice*
   > sudo apt autoremove
   ```

5. Install TensorFlow

   ```
   > sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
   > pip3 install testresources setuptools
   > pip3 install h5py
   > pip3 install numpy future mock keras_preprocessing keras_applications futures protobuf pybind11
   > pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v44 tensorflow
   ```

6. Install PyTorch

   ```
   > wget https://nvidia.box.com/shared/static/9eptse6jyly1ggt9axbja2yrmj6pbarc.whl -O torch-1.6.0-cp36-cp36m-linux_aarch64.whl
   > sudo apt install libopenblas-base libopenmpi-dev
   > pip3 install Cython
   > pip3 install torch-1.6.0-cp36-cp36m-linux_aarch64.whl
   ```
7. Install TorchVision

   ```
   > sudo apt-get install libjpeg-dev zlib1g-dev libavcodec-dev
   > cd /tmp
   > export BUILD_VERSION=0.7.0
   > git clone --branch v$BUILD_VERSION https://github.com/pytorch/vision torchvision
   > cd torchvision
   > sudo python3 setup.py install
   ```

8. Install Vosk

   Install Vosk using a prebuilt binary:

   ```
   pip install vosk-0.3.10-cp36-cp36m-linux_aarch64.whl
   ```

   If you do not find or want to use a prebuilt binary, instructions to build Vosk from source code is available here:
   https://github.com/alphacep/vosk-api/issues/164

   Note that Vosk example requires a model!

   Download a model from https://alphacephei.com/vosk/models and unpack as 'model' in the current folder where Edgepipes are run.
   Recommended model to start with is `vosk-model-small-en-us-0.4`.

9. Install Edgepipes

   ```
   > git clone https://github.com/joakimeriksson/edgepipes.git
   > cd edgepipes
   > pip3 install mss paho-mqtt
   # For face recogniztion
   > pip3 install -r face/requirements.txt
   ```

10. Test Edgepipes

   This example assumes a USB web cam has been connected.

   ```
   > ./edgepipes.py graphs/edge_detection.pbtxt
   ```

   Run the example using a connected Raspberry Pi camera

   ```
   > ./edgepipes.py graphs/edge_detection.pbtxt --input rpicam
   ```
