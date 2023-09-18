import cv2
import numpy as np
from keras.models import model_from_json

# Define your emotion classes (update to match the training code)
emotion_dict = {0: "Confused", 1: "Engaged", 2: "Frustrated", 3: "Bored", 4: "Drowsy", 5: "Looking Away"}

# Load the trained model from the JSON and weights files
json_file = open('emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# Load model weights
emotion_model.load_weights("emotion_model.h5")
print("Loaded model from disk")

# start the webcam feed
#cap = cv2.VideoCapture(0)
# Open the video capture (you can also use your webcam)
cap = cv2.VideoCapture("C:/Users/Shreyash Y Patil/Pictures/Azure/WIN_20230916_18_04_59_Pro.mp4")

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create a named window with a specific size
window_name = "Emotion Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Use WINDOW_NORMAL for resizable window
cv2.resizeWindow(window_name, 500, 500)  # Set the window size (width, height)

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract the face region
        face = frame[y:y + h, x:x + w]

        # Resize the face image to match the input size of the model (224x224) if needed
        face = cv2.resize(face, (224, 224))

        # Preprocess the face image
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(face, axis=0) / 255.0

        # Predict emotions for the face
        emotion_prediction = emotion_model.predict(img)
        maxindex = int(np.argmax(emotion_prediction))

        # Get the detected emotion text
        detected_emotion = emotion_dict[maxindex]

        # Draw a bounding box around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

        # Display the detected emotion on the frame
        cv2.putText(frame, detected_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the frame with emotion detection
    cv2.imshow(window_name, frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
