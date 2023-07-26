import cv2
from ultralytics import YOLO
import numpy as np

class_names = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorbike",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18:"sheep",
    19:"cow",
    20:"elephant",
    21:"bear",
    22:"zebra",
    23:"giraffe",
    24:"backpack",
    25:"umbrella",
    26:"handbag",
    27:"tie",
    28:"suitcase",
    29:"frisbee",
    30:"skis",
    31:"snowboard",
    32:"sports ball",
    33:"kite",
    34:"baseball bat",
    35:"baseball glove",
    36:"skateboard",
    37:"surfboard",
    38:"tennis racket",
    39:"bottle",
    40:"wine glass",
    41:"cup",
    42:"fork",
    43:"knife",
    44:"spoon",
    45:"bowl",
    46:"banana",
    47:"apple",
    48:"sandwich",
    49:"orange",
    50:"broccoli",
    51:"carrot",
    52:"hot dog",
    53:"pizza",
    54:"donut",
    55:"cake",
    56:"chair",
    57:"sofa",
    58:"pottedplant",
    59:"bed",
    60:"diningtable",
    61:"toilet",
    62:"tvmonitor",
    63:"laptop",
    64:"mouse",
    65:"remote",
    66:"keyboard",
    67:"cell phone",
    68:"microwave",
    69:"oven",
    70:"toaster",
    71:"sink",
    72:"refrigerator",
    3:"book",
    74:"clock",
    75:"vase",
    76:"scissors",
    77:"teddy bear",
    78:"hair drier",
    79:"toothbrush"
}

cap = cv2.VideoCapture(0)

model = YOLO("yolov8m.pt")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, device="cpu")
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")

    for cls, bbox in zip(classes, bboxes):
        (x, y, x2, y2) = bbox

        class_name = class_names.get(cls, "Unknown")

        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, class_name, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    cv2.imshow("Img", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()