import numpy as np
import cv2

# Load model
net = cv2.dnn.readNet('weights/yolov3.weights', 'cfg/yolov3.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers().ravel()]

# name of classes on which the model was trained on
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f]

# set color for each individual classes
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Load image
img = cv2.imread('../Images/car-plate 4.jpg')
img = cv2.resize(img, (1600, 900))
height, width, channels = img.shape

# Feed image to network
blob = cv2.dnn.blobFromImage(image=img, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
net.setInput(blob)
outputs = net.forward(output_layers)

# Bounding boxes
boxes = []
confidences = []
class_ids = []

for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            
            # Rectangle co-ordinates
            x = int(center_x - (w / 2))
            y = int(center_y - (h / 2))
                        
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Non-max suppression
indices = cv2.dnn.NMSBoxes(bboxes=boxes, scores=confidences, score_threshold=0.5, nms_threshold=0.4)

# Draw most confident boxes in the image
for i, box in enumerate(boxes):
    if i in indices:
        x, y, w, h = box
        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]

        cv2.rectangle(img, (x, y), (x+w, y+h), color=color, thickness=2)
        text = "{}: {:.4f}".format(classes[class_ids[i]], confidences[i])
        cv2.putText(img=img, text=text, org=(x, y-5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color, thickness=2)
        
# Show output
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()