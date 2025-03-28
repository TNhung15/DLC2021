from ultralytics import YOLO
import os
import glob
import cv2
import shutil
import random

# Định nghĩa đường dẫn
BASE_DIR = "/home/nhung/Desktop/DLC2021"
or_path = os.path.join(BASE_DIR, "data", "or")
re_path = os.path.join(BASE_DIR, "data", "re")
splits = {"train": 0.7, "val": 0.2, "test": 0.1}
output_path = os.path.join(BASE_DIR, "re_or_dataset")

# Load mô hình yolov11 trong detection 10 loại giấy tờ
detection_model = YOLO("/home/nhung/Desktop/DLC2021/runs/detect/train/weights/best.pt")


if os.path.isdir(output_path):
    shutil.rmtree(output_path)
os.makedirs(os.path.join(output_path, "train/recap"))
os.makedirs(os.path.join(output_path, "train/original"))
os.makedirs(os.path.join(output_path, "val/recap"))
os.makedirs(os.path.join(output_path, "val/original"))
os.makedirs(os.path.join(output_path, "test/recap"))
os.makedirs(os.path.join(output_path, "test/original"))

# Xử lý dữ liệu
image_info = []
counter = 0

for path in [or_path, re_path]:
    label = "original" if "or" in path else "recap"
    img_base_path = os.path.join(path, "images")

    for class_path in glob.glob(os.path.join(img_base_path, "*")):
        class_name = os.path.basename(class_path)
        for sub_path in glob.glob(os.path.join(class_path, "*")):
            image_files = glob.glob(os.path.join(sub_path, "*.jpg"))

            for image_file in image_files:
                image = cv2.imread(image_file)
                if image is None:
                    continue

                # Dự đoán để phát hiện giấy tờ
                results = detection_model.predict(image_file, imgsz=640, conf=0.25)

                for i, result in enumerate(results):
                    for box in result.boxes:
                        x, y, w, h = box.xywh[0].cpu().numpy()
                        x, y, w, h = int(x - w/2), int(y - h/2), int(w), int(h)
                        class_id = int(box.cls)
                        doc_type = result.names[class_id]

                        # Crop vùng chứa giấy tờ
                        cropped_image = image[y:y+h, x:x+w]
                        cropped_name = f"{label}_{class_name}_{counter}.jpg"

                        # Lưu ảnh crop ngay lập tức vào thư mục tạm
                        temp_img_path = os.path.join(BASE_DIR, "temp", cropped_name)
                        os.makedirs(os.path.dirname(temp_img_path), exist_ok=True)
                        cv2.imwrite(temp_img_path, cropped_image)

                        image_info.append({
                            "path": temp_img_path,  # Lưu đường dẫn thay vì ảnh
                            "label": label,
                            "doc_type": doc_type,
                            "filename": cropped_name
                        })
                        counter += 1

                # Giải phóng bộ nhớ của ảnh gốc
                del image

# Chia dữ liệu thành train, val, test
random.shuffle(image_info)
total = len(image_info)
train_end = int(total * splits["train"])
val_end = train_end + int(total * splits["val"])

split_pairs = {
    "train": image_info[:train_end],
    "val": image_info[train_end:val_end],
    "test": image_info[val_end:]
}

# Sao chép file và lưu thông tin loại giấy tờ
image_info_file = os.path.join(BASE_DIR, "real_or_image_info.txt")
with open(image_info_file, "w") as f:
    for split, pairs in split_pairs.items():
        for info in pairs:
            label = info["label"]
            doc_type = info["doc_type"]
            temp_img_path = info["path"]
            filename = info["filename"]

            output_img_path = os.path.join(output_path, split, label, filename)
            shutil.move(temp_img_path, output_img_path)  # Di chuyển file từ thư mục tạm
            f.write(f"{output_img_path},{label},{doc_type}\n")

# Xóa thư mục tạm
shutil.rmtree(os.path.join(BASE_DIR, "temp"), ignore_errors=True)