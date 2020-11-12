from datetime import datetime
from queue import Queue
from calculators.core import Calculator, TextData
import json
import pyaudio
import re
import wave


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
    audio_index = None

    def __init__(self, name, s, options=None):
        super().__init__(name, s, options)
        self.output_data = [None]
        self.output_queue = Queue(maxsize=16)
        self.paud = pyaudio.PyAudio()
        if options is not None and 'audio' in options:
            audio_description = options['audio']
            self.audio_index = _find_audio_index(self.paud, audio_description, False)
        print("*** Capture from", 'default' if self.audio_index is None else self.audio_index)

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
            self.set_output(0, data)
            return True
        if self.cap is None:
            input_device_index = None if self.audio_index is None else self.audio_index
            self.cap = self.paud.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True,
                                      input_device_index=input_device_index, frames_per_buffer=8000,
                                      stream_callback=self._callback)
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


class PlaySound(Calculator):

    _paud = None
    _stream = None
    _wf = None

    def __init__(self, name, s, options=None):
        super().__init__(name, s, options)
        self._sound_table = {}
        if options is not None:
            for w in [w for w in options.keys() if w.startswith("on")]:
                self._sound_table[w[2:].lower()] = options[w]
            if 'audio' in options:
                audio_description = options['audio']
                paud = pyaudio.PyAudio()
                self.audio_index = _find_audio_index(paud, audio_description, True)
                paud.terminate()

    def process(self):
        text = self.get(0)
        if not isinstance(text, TextData):
            if self._stream and not self._stream.is_active():
                self.close()
            return False

        sound_file = None
        for w in self._sound_table.keys():
            if w in text.text:
                sound_file = self._sound_table[w]
                break

        if sound_file is None:
            return False

        if self._stream:
            self.stop_sound()

        print(f"On '{text.text}' playing {sound_file}")

        try:
            # open the file for reading.
            self._wf = wave.open(sound_file, 'rb')

            # create an audio object
            if self._paud is None:
                self._paud = pyaudio.PyAudio()

            # length of data to read.
            chunk = 1024
            output_device_index = None if self.audio_index is None else self.audio_index
            stream = self._paud.open(format=self._paud.get_format_from_width(self._wf.getsampwidth()),
                                     output_device_index=output_device_index,
                                     channels=self._wf.getnchannels(),
                                     rate=self._wf.getframerate(),
                                     frames_per_buffer=chunk,
                                     stream_callback=self._playing_callback,
                                     output=True)
            stream.start_stream()
            return True
        except FileNotFoundError:
            print("Could not open the sound file", sound_file)
        return False

    def _playing_callback(self, in_data, frame_count, time_info, status):
        data = self._wf.readframes(frame_count) if self._wf else b''
        return (data, pyaudio.paContinue) if data else (data, pyaudio.paComplete)

    def stop_sound(self):
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._wf:
            self._wf.close()
            self._wf = None

    def close(self):
        self.stop_sound()
        if self._paud:
            self._paud.terminate()
            self._paud = None


def _find_audio_index(paud, audio_description, is_output):
    if isinstance(audio_description, str):
        if audio_description.isnumeric():
            return int(audio_description)
        elif audio_description.startswith("name:"):
            name = audio_description[5:]
            audio_index = _find_audio_index_by_name(paud, name, is_output)
            if audio_index is None:
                print("Could not find an audio device matching", name)
            return audio_index
        else:
            print("Unknown audio description:", audio_description)
    elif isinstance(audio_description, int):
        return audio_description
    else:
        print(f"Illegal audio description '{audio_description} - must be int or string")
    return None


def _find_audio_index_by_name(paud, name, is_output):
    pattern = re.compile(name)
    info = paud.get_host_api_info_by_index(0)
    for i in range(0, info.get('deviceCount')):
        device_info = paud.get_device_info_by_host_api_device_index(0, i)
        channels = 'maxOutputChannels' if is_output else 'maxInputChannels'
        if device_info.get(channels) > 0:
            device_name = device_info.get('name')
            if pattern.search(device_name):
                print(f"Found audio device {device_name} at audio index {i}")
                return i
    return None
