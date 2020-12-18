import cv2
import numpy as np

class Detector:
    def __init__(self, pth_weights: str, pth_cfg: str, pth_classes: str):
        self.net = cv2.dnn.readNet(pth_weights, pth_cfg)
        self.classes = []
        with open(pth_classes, 'r') as f:
            self.classes = f.read().splitlines()
        self.font = cv2.FONT_HERSHEY_PLAIN
        self.color = (255, 255, 0)
        self.coordinates = None
        self.img = None
        self.fig_image = None
        self.roi_image = None

    def detect(self, img_path: str):
        orig = cv2.imread(img_path)
        self.img = orig
        img = orig.copy()
        height, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0))
        self.net.setInput(blob)
        output_layer_names = self.net.getUnconnectedOutLayersNames()
        layer_outputs = self.net.forward(output_layer_names)
        boxes = []
        confidences = []
        class_ids = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                cv2.rectangle(img, (x, y), (x + w, y + h), self.color, 1)
                cv2.putText(img, label + ':' + confidence, (x, y-5), self.font, 1, (255, 255, 255), 1)
        self.image = img
        self.coordinates = (x, y, w, h)
        return


plane = Detector(
    pth_weights='Plane/yolov3_training_last.weights',
    pth_cfg='Plane/yolov3_testing.cfg',
    pth_classes='Plane/classes.txt'
)

plane.detect('Pathtotestimage')

cv2.imshow("Detection",cv2.cvtColor(plane.image, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("Detection.png",cv2.cvtColor(plane.image, cv2.COLOR_BGR2RGB))