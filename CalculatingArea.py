import cv2
from time import time

from MaskiRCNN.MaskRCNNTool import MaskRCNNDetector

mrcnn_det = MaskRCNNDetector(
    model_path=r"dnn/frozen_inference_graph_coco.pb",
    config_path=r"dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt",
    coco_names_path=r"dnn/coco.names",
    confidence_threshold=0.5,
)
PIXEL_TO_SQUARE_CM = 48 * 48  # 48 pixels = 1cm in on the photo
classes = mrcnn_det.classes
video_mode = False

# Image example
img = cv2.imread("Images/test3.png")
detections = mrcnn_det.get_detections(img)

for det in detections:
    class_id, conf, x1, y1, x2, y2, object_contour = det
    conf *= 100
    roi = img[y1:y2, x1:x2]
    class_color = [int(c) for c in mrcnn_det.color_list[class_id]]

    img = mrcnn_det.draw_mask(img, roi, object_contour, class_color)

    area_px = cv2.contourArea(object_contour)
    area_cm = round(area_px / PIXEL_TO_SQUARE_CM, 2)

    cv2.rectangle(img, (x1, y1), (x2, y2), class_color, 2)
    cv2.rectangle(img, (x2, y1-20), (x1, y1), class_color, -1)
    cv2.putText(img, f"{area_cm} cm^2", (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 1.2,
                (255, 255, 255), 2)

cv2.imshow("res", img)
cv2.waitKey(0)