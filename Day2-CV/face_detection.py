import cv2
import numpy as np

def detect_faces_webcam():
    """
    Real-time face detection using webcam.
    Press 'q' to quit, 's' to save current frame.
    """
    # Load face cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml'
    )

    smile_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_smile.xml'
    )
    
    # Open webcam (0 is default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print('Error: Could not open webcam')
        return
    
    # Set camera resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    
    print('Starting face detection... Press q to quit, s to save frame')
    
    while True:
        # Read frame
        ret, frame = cap.read()
        
        if not ret:
            print('Error: Failed to capture frame')
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=15,
            minSize=(5,5)
        )

        # Detect smiles
        smiles = smile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=15,
            minSize=(25, 25)
        )

        # Draw rectangles and labels
        for (x, y, w, h) in faces:
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add label
            cv2.putText(frame, 'Face', (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            face = gray[y:y + h, x:x + w]
            face_color = frame[y:y + h, x:x + w]
            
            for (ex, ey, ew, eh) in eyes:
                # Draw rectangle
                cv2.rectangle(face_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
                
                # Add label
                cv2.putText(face_color, 'Eye', (ex, ey - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
            for (sx, sy, sw, sh) in smiles:
                # Draw rectangle
                cv2.rectangle(face_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
                
                # Add label
                cv2.putText(face_color, 'Smile', (sx, sy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


        # Add info text
        info_text = f'Faces detected: {len(faces)} | Press q to quit, s to save'
        cv2.putText(frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow('Face Detection', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print('Quitting...')
            break
        elif key == ord('s'):
            filename = f'face_detection_{frame_count}.jpg'
            cv2.imwrite(filename, frame)
            print(f'Saved {filename}')
            frame_count += 1
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print('Face detection completed')


if __name__ == '__main__':
    detect_faces_webcam()