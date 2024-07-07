import os
from ultralytics import YOLO
import cv2

IMAGES_DIR = os.path.join('.', 'testdata')
image_path = os.path.join(IMAGES_DIR, '7.jpg')  # Path to your image

model_path = os.path.join('.', 'model', 'runs', 'detect', 'train', 'weights', 'best.pt')

# Load the YOLO model
model = YOLO(model_path)  # load a custom model

threshold = 0.4

# Read the image
image = cv2.imread(image_path)

# Perform object detection on the image
results = model(image)
print('model :', results)

# Draw bounding boxes and labels on the image
for i, box in enumerate(results[0].boxes.xyxy):
    x1, y1, x2, y2 = box[:4]  # Extracting the box coordinates
    score = results[0].boxes.conf[i]  # Extracting the score
    class_id = int(results[0].boxes.cls[i])  # Convert class ID to integer
    if score > threshold:
        # Change the color and thickness of the bounding box
        bounding_box_color = (255, 0, 0)  # Blue color
        bounding_box_thickness = 2
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), bounding_box_color, bounding_box_thickness)
        
        class_name = results[0].names[class_id].upper() if class_id in results[0].names else 'UNKNOWN'
        
        # Change the color and thickness of the text
        text_color = (255, 0, 0)  # Blue color
        text_thickness = 1
        font_scale = 0.5
        cv2.putText(image, class_name, (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, text_thickness, cv2.LINE_AA)
        
        # Display label information to terminal
        print(f"Detected: {class_name} (Confidence: {score:.2f})")

# Display the annotated image without resizing
cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)  # Create a resizable window
cv2.imshow('Object Detection', image)
cv2.resizeWindow('Object Detection', image.shape[1], image.shape[0])  # Resize window to fit image
cv2.waitKey(0)
cv2.destroyAllWindows()
