import cv2
import numpy as np

def detect_objects(image_path):
    # Load YOLO model (Change the model files if you are using a different one)
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Load image
    image = cv2.imread(image_path)
    height, width, channels = image.shape

    # Prepare image for detection
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Run object detection
    outs = net.forward(output_layers)

    results = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                results.append({"class_id": class_id, "confidence": confidence})

    return results
