import cv2
import torch
from PIL import Image
from torchvision import transforms
from stages.utils.props import generate_bounding_box_after_pnet, resize_to_square

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the model and move it to the appropriate device
detector = torch.load('Face_Attribute_model.pth', map_location=device)
detector = detector.to(device)
detector.eval()

# Initialize the video capture object for the default camera
cap = cv2.VideoCapture(0)

while True:
    # Read the frame from the camera
    ret, frame = cap.read()

    # Get the dimensions of the frame
    frame_height, frame_width, _ = frame.shape

    pytorch_tensor = torch.from_numpy(frame).unsqueeze(0).to(device)

    # Detect faces using Facial Attribute Detection
    attributes, faces = detector(pytorch_tensor, device)

    # Draw the rectangle
    for face in faces:
        x1, y1, x2, y2 = face
        color = (0, 255, 0)  # Green color
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


    # Show the resulting frame
    cv2.imshow('Real-time Face Detection', frame)

    # Press 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
