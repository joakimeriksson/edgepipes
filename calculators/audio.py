from datetime import datetime
from calculators.core import Calculator
import pyaudio

class AudioData:
    def __init__(self, audio, timestamp):
        self.audio = audio
        self.timestamp = timestamp

class AudioCaptureNode(Calculator):

    cap = None
    paud = None

    def __init__(self, name, s, options=None):
        super().__init__(name, s, options)
        self.output_data = [None]
        self.output_buffers = []
        if options is not None and 'audio' in options:
            self.audio = options['audio']
        else:
            self.audio = 0
        print("*** Capture from ", self.audio)
        self.paud = pyaudio.PyAudio()

    def callback(self, in_data, frame_count, time_info, status):
        now = datetime.now()
        timestamp = datetime.timestamp(now)
        print("Got callback")
        self.output_buffers += [AudioData(in_data, timestamp)]
        return (in_data, pyaudio.paContinue)


    def process(self):
        data = None
        if len(self.output_buffers) > 0:
            data = self.output_buffers.pop(0)
            if len(self.output_buffers) > 0:
                print("Warning - more audio left in buffer: ", len(self.output_buffers))
        elif self.cap is None:
            self.cap = self.paud.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000, stream_callback=self.callback)
            self.cap.start_stream()
        if data is not None:
            self.set_output(0, data)
            return True
        return False

    def close(self):
        if self.cap:
            self.cap.stop_stream()
            self.cap.close()
            self.cap = None
        if self.paud:
            self.paud.terminate()
            self.paud = None

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def spectrogram_image(y, sr, hop_length, n_mels):
    import librosa
    import numpy
    # use log-melspectrogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                          n_fft=hop_length*2, hop_length=hop_length)
    mels = numpy.log(mels + 1e-9) # add small number to avoid log(0)

    # min-max scale to fit inside 8-bit range
    img = scale_minmax(mels, 0, 255).astype(numpy.uint8)
    img = numpy.flip(img, axis=0) # put low frequencies at the bottom in image
    img = 255-img # invert. make black==more energy
    return img

class SpectrogramCalculator(Calculator):

    def __init__(self, name, s, options=None):
        import numpy
        super().__init__(name, s, options)
        self.output_data = [None]
        self.audio = None

    def process(self):
        import numpy
        from calculators.image import ImageData

        audio = self.get(0)
        if isinstance(audio, AudioData):
            hop_length = 256 # number of samples per time-step in spectrogram
            n_mels = 128 # number of bins in spectrogram. Height of image
            time_steps = 384 # number of time-steps. Width of image
            start_sample = 0 # starting at beginning
            length_samples = time_steps*hop_length

            sr = 16000
            y = numpy.fromstring(audio.audio, numpy.int16) / 32768.0
            if self.audio is None:
                self.audio = y
            else:
                self.audio = numpy.append(self.audio, y)

            if (len(self.audio) > length_samples):
                self.audio = self.audio[len(self.audio) - length_samples:]
            y = self.audio
            window = y[start_sample:start_sample+length_samples]
            img = spectrogram_image(window, sr=sr, hop_length=hop_length, n_mels=n_mels)
            self.set_output(0, ImageData(img, audio.timestamp))
            return True
        return False

class VoskVoiceToTextCalculator(Calculator):

    def __init__(self, name, s, options=None):
        from vosk import Model, KaldiRecognizer
        super().__init__(name, s, options)
        self.model = Model("model")
        self.rec = KaldiRecognizer(self.model, 16000)
        self.output_data = [None, None]

    def process(self):
        import vosk
        audio = self.get(0)
        if isinstance(audio, AudioData):
            if self.rec.AcceptWaveform(audio.audio):
                res = self.rec.Result()
                print("Voice2Text:", res)
                self.set_output(0, res)
            else:
                res = self.rec.PartialResult()
                print("Voice2Word:", res)
                self.set_output(1, res)
            return True
        return False
