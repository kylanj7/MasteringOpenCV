import cv2

def play(prototxt_path, model_path):
    '''
    Run the main loop until cancelled.
    '''
    cap = cv2.VideoCapture(0)

    # Getting the Frame width and height.
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))

    # Co-ordinates of the bounding box on frame
    left_x, top_y = frame_width // 2 - 150, frame_height // 2 - 200
    right_x, bottom_y = frame_width // 2 + 150, frame_height // 2 + 200
    bbox = [left_x, right_x, bottom_y, top_y]

    while not cap.isOpened():
        cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            return 0

        frame = cv2.flip(frame, 1)
        # To be added: Detecting and drawing bounding box around faces

        # Drawing the control rectangle in the center of the frame.
        frame = cv2.rectangle(
            frame, (left_x, top_y), (right_x, bottom_y), (0, 0, 255), 5)
        # To be added: Checking for game-start position, and checking to run keyboard press.
        # Exit the loop on pressing the `esc` key.
        k = cv2.waitKey(5)
        if k == 27:
            return

