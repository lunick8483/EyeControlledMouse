import cv2
import mediapipe as mp
import pyautogui

# Initialize the camera
cam = cv2.VideoCapture(0)

# Initialize the FaceMesh model from Mediapipe
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Get the screen width and height using PyAutoGUI
screen_w, screen_h = pyautogui.size()

while True:
    # Read a frame from the camera
    _, frame = cam.read()

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert the BGR frame to RGB format (required by Mediapipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the RGB frame using the FaceMesh model to detect facial landmarks
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks

    # Get the height and width of the frame
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        # Get the facial landmarks of the first detected face
        landmarks = landmark_points[0].landmark

        # Draw green circles around the eye corners (landmarks 474 to 477)
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))

            # Move the mouse cursor to the position of the second landmark (index 1)
            if id == 1:
                screen_x = screen_w * landmark.x
                screen_y = screen_h * landmark.y
                pyautogui.moveTo(screen_x, screen_y)

        # Get the landmarks corresponding to the left eye (landmarks 145 and 159)
        left = [landmarks[145], landmarks[159]]

        # Draw cyan circles around the left eye landmarks
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))

        # Check if the user blinks their left eye (y-coordinate difference is small)
        if (left[0].y - left[1].y) < 0.004:
            pyautogui.click()
            pyautogui.sleep(1)

    # Show the frame with the drawn landmarks
    cv2.imshow('Eye Controlled Mouse', frame)

    # Wait for a keypress event with a delay of 1 millisecond
    cv2.waitKey(1)
