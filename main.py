import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start the webcam feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Analyze the frame for emotions
        result = DeepFace.analyze(frame, actions=['emotion'])
        dominant_emotion = result[0]['dominant_emotion'] if isinstance(result, list) else result['dominant_emotion']
    except Exception as e:
        print(f"Error analyzing frame: {e}")
        dominant_emotion = "N/A"
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around each detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Define the font and position for the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    color = (0, 0, 255)
    
    # Put the dominant emotion text on the frame
    cv2.putText(frame, dominant_emotion, org, font, 3, color, 2, cv2.LINE_AA)

    # Display the frame using matplotlib
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.draw()
    plt.pause(0.001)
    plt.clf()  # Clear the figure for the next frame

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all OpenCV windows
cap.release()
plt.close()
