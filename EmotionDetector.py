import cv2 
from deepface import DeepFace

class EmotionDetector:
    def __init__(self):
        self.model = DeepFace.build_model("Emotion")
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
        self.emoticons = {
            'Happy': cv2.imread('happy.png', -1),
            'Sad': cv2.imread('sad.png', -1),
            'Angry': cv2.imread('angry.png', -1),
            'Fear': cv2.imread('fear.png', -1),
            'Neutral': cv2.imread('neutral.png', -1),
            'Surprise': cv2.imread('surprise.png', -1),
            'Disgust': cv2.imread('disgust.png', -1),
            # Tambahkan gambar emotikon untuk emosi lainnya
        }

    def detect_emotions(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)

        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (48, 48))
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                face_img = face_img.astype("float") / 255.0
                face_img = cv2.resize(face_img, (48, 48))
                face_img = face_img.reshape(1, 48, 48, 1)
                emotions = self.model.predict(face_img)[0]
                emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
                max_index = emotions.argmax()
                emotion_label = emotion_labels[max_index]
                emotion_probability = emotions[max_index]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{emotion_label}: {emotion_probability:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                emoticon = self.emoticons.get(emotion_label)

                if emoticon is not None:
                    emoticon_height, emoticon_width, _ = emoticon.shape
                    if emoticon_height > h or emoticon_width > w:
                        emoticon = cv2.resize(emoticon, (w, h))
                    x_offset = x
                    y_offset = y - emoticon.shape[0] - 10

                    for c in range(3):
                        if y_offset >= 0 and x_offset >= 0 and y_offset + emoticon.shape[0] <= frame.shape[0] and x_offset + emoticon.shape[1] <= frame.shape[1]:
                            frame[y_offset:y_offset + emoticon.shape[0], x_offset:x_offset + emoticon.shape[1], c] = \
                                emoticon[:, :, c] * (emoticon[:, :, 3] / 255.0) + \
                                frame[y_offset:y_offset + emoticon.shape[0], x_offset:x_offset + emoticon.shape[1], c] * \
                                (1.0 - emoticon[:, :, 3] / 255.0)

            cv2.imshow('Face Emotion Detection', frame)
            cv2.imshow('gray scale' , gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
