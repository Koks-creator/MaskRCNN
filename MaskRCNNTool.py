from dataclasses import dataclass
from typing import List, Union
import cv2
import numpy as np

np.random.seed(20)


@dataclass
class MaskRCNNDetector:
    model_path: str
    config_path: str
    coco_names_path: str
    confidence_threshold: float

    def __post_init__(self):
        with open(self.coco_names_path) as coco_f:
            self.classes = coco_f.read().splitlines()
            self.color_list = np.random.randint(low=0, high=255, size=(len(self.classes), 3))

        self.net = cv2.dnn.readNetFromTensorflow(
            model=self.model_path,
            config=self.config_path
        )

    def get_detections(self, img: np.array, allowed_classes=None) -> List[Union[list, np.array]]:
        """
        :param img: your image
        :param allowed_classes: list of classes you want to be detected
        :return: list of class_id, conf, x1, y1, x2, y2, contours of every detected object
        """

        h, w, _ = img.shape
        detections = []

        blob = cv2.dnn.blobFromImage(img, swapRB=True)
        self.net.setInput(blob)

        boxes, masks = self.net.forward(["detection_out_final", "detection_masks"])
        boxes = np.squeeze(boxes)

        if allowed_classes is None:
            allowed_classes = [i for i in range(len(self.classes))]

        for box, mask in zip(boxes, masks):
            class_id = int(box[1])
            conf = box[2]

            if conf < self.confidence_threshold:
                continue

            if class_id in allowed_classes:
                x1, y1, x2, y2 = int(box[3] * w), int(box[4] * h), int(box[5] * w), int(box[6] * h)

                roi = img[y1:y2, x1:x2]
                roi_h, roi_w, _ = roi.shape
                mask = mask[int(class_id)]

                mask = cv2.resize(mask, (roi_w, roi_h))
                _, mask = cv2.threshold(mask, self.confidence_threshold, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                detections.append((class_id, conf, x1, y1, x2, y2, contours[0]))

        return detections

    @staticmethod
    def draw_mask(img: np.array, roi: np.array, contours: np.array, color: List[Union[int, int, int]], alpha=0.5) -> np.array:
        """
        :param img: input image
        :param roi: region of interest (cropped object image)
        :param contours: contours of object took from get_detections method
        :param color:
        :param alpha:
        :return: image with object mask
        """
        overlay = img.copy()

        cv2.fillPoly(roi, [contours], color)
        cv2.polylines(roi, [contours], True, (255, 255, 255), 2)
        final_img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        return final_img
