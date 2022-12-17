import cv2
from time import time

from MaskiRCNN.MaskRCNNTool import MaskRCNNDetector

mrcnn_det = MaskRCNNDetector(
    model_path=r"dnn/frozen_inference_graph_coco.pb",
    config_path=r"dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt",
    coco_names_path=r"dnn/coco.names",
    confidence_threshold=0.5,
)
classes = mrcnn_det.classes
video_mode = False

if not video_mode:
    # Image example
    img = cv2.imread("Images/cars.jpg")
    detections = mrcnn_det.get_detections(img)

    for det in detections:
        class_id, conf, x1, y1, x2, y2, object_contour = det
        conf *= 100
        roi = img[y1:y2, x1:x2]
        class_color = [int(c) for c in mrcnn_det.color_list[class_id]]

        img = mrcnn_det.draw_mask(img, roi, object_contour, class_color)

        cv2.rectangle(img, (x1, y1), (x2, y2), class_color, 2)
        cv2.rectangle(img, (x2, y1-20), (x1, y1), class_color, -1)
        cv2.putText(img, f"{classes[class_id].capitalize()} {int(conf)}%", (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 1.2,
                    (255, 255, 255), 2)

    cv2.imshow("res", img)
    cv2.waitKey(0)

else:
    # Video Example
    cap = cv2.VideoCapture("Videos/street.mp4")
    # cap = cv2.VideoCapture(0)

    p_time = 0
    while cap.isOpened():
        success, frame = cap.read()

        # rois, masks = mrcnn_det.get_rois_and_masks(frame, allowed_classes=[0])
        detections = mrcnn_det.get_detections(frame)

        for det in detections:
            class_id, conf, x1, y1, x2, y2, object_contour = det
            conf *= 100
            roi = frame[y1:y2, x1:x2]
            class_color = [int(c) for c in mrcnn_det.color_list[class_id]]

            frame = mrcnn_det.draw_mask(frame, roi, object_contour, class_color)
            cv2.rectangle(frame, (x1, y1), (x2, y2), class_color, 2)
            cv2.putText(frame, f"{classes[class_id].capitalize()} {int(conf)}%", (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN,
                        1.2, class_color, 2)

        c_time = time()
        fps = int(1 / (c_time - p_time))
        p_time = c_time

        cv2.putText(frame, f"FPS: {fps}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("VidRes", frame)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    cap.release()
