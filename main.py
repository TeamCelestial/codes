import cv2
from PIL import Image
from transformers import pipeline

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the gender classification model
pipe = pipeline("image-classification", model="rizvandwiki/gender-classification")

# Open the camera (0 is the default camera)
cap = cv2.VideoCapture(0)

def classify_gender(face_image):
    try:
        # Resize the face image to 224x224 as required by the model
        face_resized = cv2.resize(face_image, (224, 224))
        
        # Convert the image to RGB (as the model expects RGB images)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # Convert the NumPy array (OpenCV image) to PIL Image
        pil_image = Image.fromarray(face_rgb)
        
        # Predict the gender using the pipeline
        predictions = pipe(pil_image)
        
        # Get the gender with the highest score
        predicted_gender = predictions[0]['label']
        
        return predicted_gender
    except Exception as e:
        print(f"Error in gender classification: {e}")
        return "Unknown"

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame using Haar Cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around detected faces and classify them
    for (x, y, w, h) in faces:
        # Define region of interest (ROI)
        roi = frame[y:y+h, x:x+w]
        
        # Classify gender using the pipeline model
        gender = classify_gender(roi)
        
        # Draw rectangle and label with the predicted gender
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, gender, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
    # Display the resulting frame
    cv2.imshow('Face Detection with Gender Classification', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()