import cv2
import torch
from PIL import Image
from torchvision import transforms

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the model and move it to the appropriate device
detector = torch.load('Face_Attribute_model.pth', map_location=device)
detector = detector.to(device)
detector.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((12, 12))
])

# Initialize the video capture object for the default camera
cap = cv2.VideoCapture(0)

while True:
    # read the frame from the camera
    ret, frame = cap.read()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the frame to PIL Image
    pil_frame = Image.fromarray(rgb_frame)

    # Transform the image to a PyTorch tensor
    tensor_frame = transform(pil_frame).unsqueeze(0).to(device)  # Add batch dimension

    # detect faces using Facial Attribute Detection
    attributes, face, prob = detector(tensor_frame)

    print(face)

    face = face[0]  # Extract the first bounding box
    frame_height, frame_width, _ = frame.shape

    # Assuming normalized coordinates [x1, y1, x2, y2]
    x1 = abs(int(face[0] * frame_width))
    y1 = abs(int(face[1] * frame_height))
    x2 = abs(int(face[2] * frame_width))
    y2 = abs(int(face[3] * frame_height))

    print(f"Face Coordinates: ({x1}, {y1}), ({x2}, {y2})")

    # Draw the rectangle
    color = (0, 255, 0)  # Green color
    thickness = 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # show the resulting frame
    cv2.imshow('Real-time Face Detection', frame)

    # press 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()