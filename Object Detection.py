import cv2
import torch

# Model
model = torch.hub.load('WongKinYiu/yolov7', 'yolov7')

# Image
img = cv2.imread('/content/a-group-of-runners-racing-through-the-park.webp')

# Check if image was loaded successfully
if img is None:
    print("Error: Could not load image. Please check the file path and ensure the image is not corrupted.")
else:
    # Inference
    results = model(img)

    # Results
    results.print()  # or .show(), .save()

    # Bounding Boxes
    results.xyxy[0]  # im predictions (tensor)
    print(results.pandas().xyxy[0])

    # Get model's class names
    class_names = model.names  # Assuming 'names' attribute contains class names

    # Loop through detections and draw bounding boxes with labels
    for *xyxy, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, xyxy)

        # Get class name and confidence
        class_id = int(cls)
        label = f'{class_names[class_id]} {conf:.2f}'

        # Draw bounding box and label
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the image with bounding boxes and labels
    cv2_imshow(img)  # Assuming cv2_imshow is defined in your environment (Colab)

    cv2.imwrite("output.jpg", img)
