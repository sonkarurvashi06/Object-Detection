import cv2
from ultralytics import YOLO
from transformers import ViTImageProcessor, ViTForImageClassification

from PIL import Image
import numpy as np
import torch

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")  # Load YOLOv8 model

# Load ViT model for gender classification
processor = ViTImageProcessor.from_pretrained('rizvandwiki/gender-classification')
gender_model = ViTForImageClassification.from_pretrained('rizvandwiki/gender-classification')
gender_model.eval()  # Set model to evaluation mode

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Perform object detection
    results = yolo_model(frame)
    
    # Convert the frame (OpenCV) to PIL Image
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Process the results
    for result in results:
        boxes = result.boxes.xyxy.numpy()  # Get the bounding boxes
        scores = result.boxes.conf.numpy()  # Get the confidence scores
        class_ids = result.boxes.cls.numpy()  # Get the class IDs

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            conf = scores[i]
            cls = int(class_ids[i])

            if cls == 0 and conf > 0.5:  # Check if the detected class is 'person' (usually class ID 0 for 'person')
                # Draw bounding box around the detected person
                cv2.rectangle(frame, (int(x1)-10, int(y1)-10), (int(x2)-10, int(y2)-10), (0, 255, 0), 2)
                
                # Crop the detected person's face from the frame
                face_image = frame[int(y1):int(y2), int(x1):int(x2)]
                if face_image.size == 0:  # Check if face_image is empty
                    continue
                
                # Convert cropped face image to PIL Image
                face_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
                
                # Preprocess the image for gender classification
                inputs = processor(images=face_pil, return_tensors="pt")

                # Perform gender classification
                with torch.no_grad():
                    outputs = gender_model(**inputs)
                logits = outputs.logits
                predicted_class_idx = logits.argmax(-1).item()
                gender = gender_model.config.id2label[predicted_class_idx]

                # Put gender label on the frame
                cv2.putText(frame, f"Gender: {gender}", (int(x1), int(y1) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
exit
            # Put class label and confidence score on the frame
            label = f"{yolo_model.names[cls]}: {conf:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("YOLOv8 Object Detection & Gender Classification", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()