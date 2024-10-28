from keras.models import load_model
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array

# Load the Haar Cascade for face detection
face_classifier = cv2.CascadeClassifier(r'D:\Projects\Emotion Detector\haarcascade_frontalface_default (1).xml')

# Load the trained emotion detection model
classifier = load_model(r'D:\Projects\Emotion Detector\mobilenet_emotion_model.keras')

# Define class labels for the emotions
class_labels = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (96, 96), interpolation=cv2.INTER_AREA)

        if np.sum(roi_gray) != 0:
            # Convert grayscale image to RGB
            roi_rgb = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)

            roi = roi_rgb.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Make a prediction on the ROI
            preds = classifier.predict(roi)[0]
            label = class_labels[preds.argmax()]
            label_position = (x, y)

            # Display the label on the frame
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    # Display the video frame with detections
    cv2.imshow('Emotion Detector', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
