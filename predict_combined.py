from ultralytics import YOLO
import cv2
import os

# Load mô hình phát hiện và phân loại
detect_model = YOLO("/home/nhung/Desktop/DLC2021/runs/detect/train/weights/best.pt")
classify_model = YOLO("/home/nhung/Desktop/DLC2021/runs/classify/train/weights/best.pt")

# Đường dẫn đến ảnh mới
image_path = "/home/nhung/Desktop/DLC2021/test5.jpg"
image = cv2.imread(image_path)
if image is None:
    exit()
# Bước 1: Phát hiện giấy tờ
detect_results = detect_model.predict(image_path, imgsz=640, conf=0.25)

# Duyệt qua các bounding box được phát hiện
for result in detect_results:
    for box in result.boxes:
        # Lấy tọa độ bounding box
        x, y, w, h = box.xywh[0].cpu().numpy()
        x, y, w, h = int(x - w/2), int(y - h/2), int(w), int(h)

        # Cắt vùng chứa giấy tờ
        cropped_image = image[y:y+h, x:x+w]
        if cropped_image.size == 0:
            continue

        # Lưu tạm vùng cắt để phân loại
        temp_path = "/home/nhung/Desktop/DLC2021/temp_cropped.jpg"
        cv2.imwrite(temp_path, cropped_image)

        # Bước 2: Phân loại vùng cắt
        classify_results = classify_model.predict(temp_path, imgsz=224)
        for cls_result in classify_results:
            predicted_class = cls_result.names[int(cls_result.probs.top1)]
            confidence = cls_result.probs.top1conf
            print(f"Giấy tờ tại ({x}, {y}, {w}, {h}): {predicted_class} (confidence: {confidence:.2f})")

        # Xóa file tạm
        os.remove(temp_path)