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


# Define attribute labels
attribute_labels = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Bags_Under_Eyes', 'Bald',
    'Bangs', 'Big_Lips', 'Big_Nose', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
    'Eyeglasses', 'Goatee', 'High_Cheekbones', 'Mustache', 'Narrow_Eyes',
    'Oval_Face', 'Pointy_Nose', 'Receding_Hairline', 'Sideburns', 'Smiling'
]

# Initialize the video capture object for the default camera
cap = cv2.VideoCapture(0)

while True:
    # Read the frame from the camera
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)

    # Get the dimensions of the frame
    frame_height, frame_width, _ = frame.shape

    pytorch_tensor = torch.from_numpy(frame).unsqueeze(0).to(device)

    # Detect faces using Facial Attribute Detection
    with torch.no_grad():
        attributes, faces = detector(pytorch_tensor, device)

    # Draw the rectangle
    for face, attr in zip(faces, attributes):
        x1, y1, x2, y2 = face
        # color = (0, 255, 0)  # Green color
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Process attributes (sigmoid activation + thresholding)
        probabilities = torch.sigmoid(attr).cpu().numpy()
        detected_attributes = [label for label, prob in zip(attribute_labels, probabilities) if prob > 0.3]

        # print("Probabilities:", probabilities)

        # Print attributes in terminal
        # print(f"Face at ({x1}, {y1}, {x2}, {y2}): Detected Attributes: {detected_attributes}")

        # Display attributes on the frame
        # if detected_attributes:
        #     text = ", ".join(detected_attributes)
        #     cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                # Get top three attributes with confidence
        top_indices = probabilities.argsort()[-3:][::-1]
        top_attributes = [(attribute_labels[i], probabilities[i] * 100) for i in top_indices]

        # # Filter attributes by threshold
        # threshold = 0.3  # Set your confidence threshold
        # text_lines = [
        #     f"{label}: {prob * 100:.1f}%" 
        #     for label, prob in zip(attribute_labels, probabilities) 
        #     if prob > threshold
        # ]

        # # Format text for display
        # # text_lines = [f"{attr}: {conf:.1f}%" for attr, conf in top_attributes]

        # # Display text in the bottom-right corner
        # x_text = frame_width - 200  # Adjust for width
        # y_text = frame_height - 20  # Start from the bottom of the frame

        # for line in text_lines:
        #     cv2.putText(frame, line, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        #     y_text -= 20  # Move up for the next line
    
     # Filter attributes by threshold
        threshold = 0.3  # Set your confidence threshold
        detected_attributes = [
            (label, prob * 100) 
            for label, prob in zip(attribute_labels, probabilities) 
            if prob > threshold
        ]

        # Print attributes in the terminal
        print(f"Face at ({x1}, {y1}, {x2}, {y2}): Detected Attributes: {detected_attributes}")

        # Display percentage bars in the bottom-right corner
        bar_start_x = frame_width - 220  # Left edge of bars
        bar_height = 20  # Height of each bar
        bar_length = 200  # Maximum bar length
        y_offset = frame_height - 20  # Start near the bottom of the frame

        for label, confidence in detected_attributes:
            # Draw the background bar (gray)
            cv2.rectangle(frame, (bar_start_x, y_offset - bar_height), 
                          (bar_start_x + bar_length, y_offset), (200, 200, 200), -1)
            
            # Draw the confidence bar (green)
            cv2.rectangle(frame, (bar_start_x, y_offset - bar_height), 
                          (bar_start_x + int(bar_length * (confidence / 100)), y_offset), (0, 255, 0), -1)
            
            # Add the label and percentage
            text = f"{label}: {confidence:.1f}%"
            cv2.putText(frame, text, (bar_start_x, y_offset - bar_height - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Move to the next bar
            y_offset -= (bar_height + 20)

    # Show the resulting frame
    cv2.imshow('Real-time Face Detection', frame)

    # Press 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
