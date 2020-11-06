import handtracker.hand_tracker
from datetime import datetime
from calculators.core import Calculator, TextData
from calculators.image import ImageData

palm_model_path = "handtracker/models/palm_detection.tflite"
landmark_model_path = "handtracker/models/hand_landmark.tflite"
anchors_path = "handtracker/anchors.csv"


class HandDetector(Calculator):
    def __init__(self, name, s, options=None):
        super().__init__(name, s)
        self.input_data = [None]
        self.detector = handtracker.hand_tracker.HandTracker(palm_model_path, landmark_model_path, anchors_path,
                                                             box_shift=0.2, box_enlarge=1.3)

    def process(self):
        image = self.get(0)
        if isinstance(image, ImageData):
            nf = image.image.copy()
            img = nf[:, :, ::-1]
            kp, box = self.detector(img)

            if kp is not None:
                handtracker.hand_tracker.draw_hand(nf, kp)
                handtracker.hand_tracker.draw_box(nf, box)
                self.set_output(0, ImageData(nf, image.timestamp))
                self.set_output(1, (kp, box))
            else:
                self.set_output(0, image)
                self.set_output(1, None)
            return True
        return False


class DrawHandDetections(Calculator):
    def __init__(self, name, s, options=None):
        super().__init__(name, s)
        self.input_data = [None, None]

    def process(self):
        if self.input_data[0] is not None and self.input_data[1] is not None:
            image = self.get(0)
            (kp, box) = self.get(1)
            if isinstance(image, ImageData):
                nf = image.image.copy()
                if kp is not None:
                    handtracker.hand_tracker.draw_hand(nf, kp)
                    handtracker.hand_tracker.draw_box(nf, box)
                self.set_output(0, ImageData(nf, image.timestamp))
                return True
        return False


class HandThumbDetector(Calculator):

    def __init__(self, name, s, options=None):
        super().__init__(name, s, options=options)

    def process(self):
        hand_data = self.get(0)
        if hand_data:
            kp, box = hand_data
            palm = kp[0][1]
            thumb = kp[4][1]
            fingers = [kp[8][1], kp[12][1], kp[16][1], kp[20][1]]
            if thumb < palm and thumb < min(fingers):
                print("Thumb up!")
                now = datetime.now()
                timestamp = datetime.timestamp(now)
                self.set_output(0, TextData("Thumb up!", timestamp))
            elif thumb > palm and thumb > max(fingers):
                print("Thumb down!")
                now = datetime.now()
                timestamp = datetime.timestamp(now)
                self.set_output(0, TextData("Thumb down!", timestamp))
        return True
