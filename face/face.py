import cv2
import face_recognition
import os
from calculators.core import Calculator
from calculators.image import ImageData

_known_face_encodings = []
_known_face_names = []
_known_images_loaded = False


# Load a sample picture and learn how to recognize it.
def _load_known_images():
    global _known_face_encodings
    global _known_face_names
    global _known_images_loaded
    if _known_images_loaded:
        return True
    image_path = 'face/known/'
    image_files = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]
    for image_file in image_files:
        image = face_recognition.load_image_file(os.path.join(image_path, image_file))
        _known_face_encodings.append(face_recognition.face_encodings(image)[0])
        _known_face_names.append(os.path.splitext(image_file)[0])
    _known_images_loaded = True


class FaceRecognizer(Calculator):

    def __init__(self, name, s, options=None):
        super().__init__(name, s)
        self.input_data = [None]
        _load_known_images()

    def process(self):
        image = self.get(0)
        if isinstance(image, ImageData):
            nf = image.image.copy()
            nf = self._process_image(nf)
            self.set_output(0, ImageData(nf, image.timestamp))
            return True
        return False

    def _process_image(self, frame):
        global _known_face_names
        global _known_face_encodings

        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        name = "Unknown"
        match = False

        # Loop through each face found in the unknown image
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(_known_face_encodings, face_encoding)
            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = _known_face_names[first_match_index]
                match = True
            cut = frame[top:bottom, left:right]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, name, (left, top - 5), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('cut', cut)
            if match:
                print("Name: ", name)

        return frame
