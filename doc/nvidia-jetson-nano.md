# Installation on NVIDIA Jetson Nano

1. Download image with JetPack 4.4: https://developer.nvidia.com/embedded/jetpack

2. Burn the image to a SD-card and boot the NVIDIA Jetson Nano with the image.
Please enter desired keyboard layout, user name, etc., when asked.
For detailed instructions please see: https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit

3. Install TensorFlow

   ```
   > sudo apt update
   > sudo apt install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
   > sudo apt install python3-pip
   > sudo -H pip3 install -U pip testresources setuptools==49.6.0
   > sudo -H sudo pip3 install -U numpy==1.16.1 future==0.18.2 mock==3.0.5 h5py==2.10.0 keras_preprocessing==1.1.1 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11
   > sudo -H pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v44 tensorflow
   ```

   Check version and CUDA support
   ```
   > python3
   >>> import tensorflow as tf
   >>> tf.__version__
   '2.3.1'
   >>> tf.test.is_built_with_cuda()
   True
   >>> tf.test.is_gpu_available()
   [...]
   True
   ```

4. Setup system

   Add CUDA to command path by adding following lines to `.bashrc`:

   ```
   > cat >>~/.bashrc
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

   Press `CTRL-D` to close the file.

5. Install system packages

   ```
   > sudo apt install python3-pyaudio
   > sudo apt install cython3
   > sudo apt install libcanberra-gtk-module
   ```

   Optionally remove some packets to reduce disk usage if you have a small SD-card.

   ```
   > sudo apt purge libreoffice*
   > sudo apt autoremove
   ```

6. Install PyTorch

   ```
   > wget https://nvidia.box.com/shared/static/wa34qwrwtk9njtyarwt5nvo6imenfy26.whl -O torch-1.7.0-cp36-cp36m-linux_aarch64.whl
   > sudo apt install libopenblas-base libopenmpi-dev
   > sudo -H pip3 install Cython
   > pip3 install torch-1.7.0-cp36-cp36m-linux_aarch64.whl
   ```

   Verify PyTorch installation
   ```
   > python3
   >>> import torch
   >>> torch.__version__
   '1.7.0'
   >>> torch.cuda.is_available()
   True
   >>> torch.backends.cudnn.version()
   8000
   >>> a = torch.cuda.FloatTensor(2).zero_()
   >>> str(a)
   "tensor([0., 0.], device='cuda:0')"
   >>> b = torch.randn(2).cuda()
   >>> str(b)
   "tensor([-2.2662, -1.0085], device='cuda:0')"
   >>> c = a + b
   >>> str(c)
   "tensor([-2.2662, -1.0085], device='cuda:0')"
   ```

7. Install TorchVision

   ```
   > sudo apt install libjpeg-dev zlib1g-dev libavcodec-dev libavformat-dev libswscale-dev apt-utils
   > cd /tmp
   > export BUILD_VERSION=0.8.1
   > git clone --branch v$BUILD_VERSION https://github.com/pytorch/vision torchvision
   > cd torchvision
   > sudo python3 setup.py install
   ```

   Check TorchVision version
   ```
   > python3
   >>> import torchvision
   >>> print(torchvision.__version__)
   '0.8.0a0+45f960c'
   ```

8. Install Vosk

   Install Vosk using a prebuilt binary:

   ```
   pip3 install vosk-0.3.10-cp36-cp36m-linux_aarch64.whl
   ```

   If you do not find or want to use a prebuilt binary, instructions to build Vosk from source code is available here:
   https://github.com/alphacep/vosk-api/issues/164

   Note that Vosk example requires a model!

   Download a model from https://alphacephei.com/vosk/models and unpack as `model` in the current folder where Edgepipes are run.
   Recommended model to start with is `vosk-model-small-en-us-0.4`.

9. Install Edgepipes

   ```
   > git clone https://github.com/joakimeriksson/edgepipes.git
   > cd edgepipes
   > pip3 install mss paho-mqtt pycuda
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

11. Useful tools

   jtop is similar to htop but also shows GPU activity

   ```
   > sudo -H pip3 install -U jetson-stats
   > # Please reboot for the Jetson stats daemon to initiate
   > jtop
   > # See system information
   > jetson_release -v
   ```
