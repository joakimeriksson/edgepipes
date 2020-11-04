from datetime import datetime
from queue import Queue
from calculators.core import Calculator, TextData
import json
import pyaudio


class AudioData:
    def __init__(self, audio, timestamp):
        self.audio = audio
        self.timestamp = timestamp

    def add_data(self, more_data):
        self.audio += more_data.audio


class VoiceTextData(TextData):
    def __init__(self, text, timestamp, info=None):
        super().__init__(text, timestamp)
        self.info = info


class AudioCaptureNode(Calculator):

    cap = None
    paud = None

    def __init__(self, name, s, options=None):
        super().__init__(name, s, options)
        self.output_data = [None]
        self.output_queue = Queue(maxsize=16)
        if options is not None and 'audio' in options:
            self.audio = options['audio']
        else:
            self.audio = 0
        print("*** Capture from ", self.audio)
        self.paud = pyaudio.PyAudio()

    def _callback(self, in_data, frame_count, time_info, status):
        # Drop sound if queue is full
        if not self.output_queue.full():
            now = datetime.now()
            timestamp = datetime.timestamp(now)
            self.output_queue.put(AudioData(in_data, timestamp))
        return in_data, pyaudio.paContinue

    def process(self):
        if not self.output_queue.empty():
            data = self.output_queue.get()
            while not self.output_queue.empty():
                data.add_data(self.output_queue.get())
                print("Added more audio: ", type(data.audio), len(data.audio))
            self.set_output(0, data)
            return True
        if self.cap is None:
            self.cap = self.paud.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000, stream_callback=self._callback)
            self.cap.start_stream()
        return False

    def close(self):
        if self.cap:
            self.cap.stop_stream()
            self.cap.close()
            self.cap = None
        if self.paud:
            self.paud.terminate()
            self.paud = None


def scale_minmax(x, min=0.0, max=1.0):
    x_min, x_max = x.min(), x.max()
    if x_min == x_max:
        # Scaling not possible
        print("audio scale not possible (min == max)")
        return x
    x_std = (x - x_min) / (x_max - x_min)
    x_scaled = x_std * (max - min) + min
    return x_scaled


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
            hop_length = 256  # number of samples per time-step in spectrogram
            n_mels = 128      # number of bins in spectrogram. Height of image
            time_steps = 384  # number of time-steps. Width of image
            start_sample = 0  # starting at beginning
            length_samples = time_steps*hop_length

            sr = 16000
            y = numpy.fromstring(audio.audio, numpy.int16) / 32768.0
            if self.audio is None:
                self.audio = y
            else:
                self.audio = numpy.append(self.audio, y)

            if len(self.audio) > length_samples:
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
        audio = self.get(0)
        if isinstance(audio, AudioData):
            if self.rec.AcceptWaveform(audio.audio):
                result = self.rec.Result()
                result_json = json.loads(result)
                if 'text' in result_json:
                    text = result_json['text']
                    if text:
                        print("Voice2Text:", repr(text), result_json)
                        self.set_output(0, VoiceTextData(text, audio.timestamp, info=result_json))
            else:
                partial_result = self.rec.PartialResult()
                partial_json = json.loads(partial_result)
                if 'partial' in partial_json:
                    text = partial_json['partial']
                    if text:
                        print("Voice2Text (partial): ", repr(text))
                        self.set_output(1, VoiceTextData(text, audio.timestamp, info=partial_json))
            return True
        return False
