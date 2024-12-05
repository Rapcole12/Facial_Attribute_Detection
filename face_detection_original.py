import cv2
import torch
from PIL import Image
from torchvision import transforms
from stages.utils.props import generate_bounding_box_after_pnet, resize_to_square

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the model and move it to the appropriate device
detector = torch.load('Face_Attribute_model_original.pth', map_location=device)
detector = detector.to(device)
detector.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((12, 12)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  #normalizes
])

# Initialize the video capture object for the default camera
cap = cv2.VideoCapture(0)

while True:
    # Read the frame from the camera
    ret, frame = cap.read()

    if not ret:
        break  # Exit if frame is not read correctly

    # Get the dimensions of the frame
    frame_height, frame_width, _ = frame.shape

    # Determine the size of the square
    square_side = min(frame_width, frame_height)

    # Calculate cropping coordinates for a center crop
    x_center = frame_width // 2
    y_center = frame_height // 2
    x1 = x_center - square_side // 2
    y1 = y_center - square_side // 2
    x2 = x1 + square_side
    y2 = y1 + square_side

    # Crop the frame to a square
    square_frame = frame[y1:y2, x1:x2]

    # Convert the square frame to RGB
    rgb_frame = cv2.cvtColor(square_frame, cv2.COLOR_BGR2RGB)

    # Convert the frame to PIL Image
    pil_frame = Image.fromarray(rgb_frame)

    # Transform the image to a PyTorch tensor
    tensor_frame = transform(pil_frame).unsqueeze(0).to(device)  # Add batch dimension

    # Detect faces using Facial Attribute Detection
    attributes, face, prob = detector(tensor_frame)

    # Generate bounding box after detection
    bounding_box = generate_bounding_box_after_pnet(face, prob, .45, cell_size=(square_side // 2, square_side // 2))
    bounding_box = resize_to_square(bounding_box)
    bounding_box = bounding_box[0]

    # Draw the rectangle
    x1, y1, x2, y2 = bounding_box
    color = (0, 255, 0)  # Green color
    thickness = 2
    
    cv2.rectangle(square_frame, (x1, y1), (x2 - x1, y2 - y1), color, thickness)

    # Flip the frame for mirroring effect
    square_frame = cv2.flip(square_frame, 1)

    # Show the resulting frame
    cv2.imshow('Real-time Face Detection', square_frame)

    # Press 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()