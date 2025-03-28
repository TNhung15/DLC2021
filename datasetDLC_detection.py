import os
import glob
import cv2
import shutil
import json
import random
from pprint import pprint

# Định nghĩa đường dẫn dựa trên thư mục thực tế
BASE_DIR = "/home/nhung/Desktop/DLC2021"  # Thư mục chứa data
or_path = os.path.join(BASE_DIR, "data", "or")
re_path = os.path.join(BASE_DIR, "data", "re")
splits = {"train": 0.7, "val": 0.2, "test": 0.1}
output_path = os.path.join(BASE_DIR, "dlc_yolo_format", "images")

# Xóa và tạo lại thư mục đầu ra
if os.path.isdir(output_path):
    shutil.rmtree(output_path)
    shutil.rmtree(output_path.replace("images", "labels"))
os.makedirs(os.path.join(output_path, "train"))
os.makedirs(os.path.join(output_path, "val"))
os.makedirs(os.path.join(output_path, "test"))
os.makedirs(os.path.join(output_path.replace("images", "labels"), "train"))
os.makedirs(os.path.join(output_path.replace("images", "labels"), "val"))
os.makedirs(os.path.join(output_path.replace("images", "labels"), "test"))

# Danh sách các lớp
classes = [
    "alb_id", "aze_passport", "esp_id", "est_id", "fin_id",
    "grc_passport", "lva_passport", "rus_internalpassport",
    "srb_passport", "svk_id"
]
classes_idx = {cls: index for index, cls in enumerate(classes)}

# Hàm chuyển đổi polygon thành bounding box
def polygon_to_bbox(points_x, points_y):
    x_min = min(points_x)
    x_max = max(points_x)
    y_min = min(points_y)
    y_max = max(points_y)
    return [x_min, y_min, x_max, y_max]

# Hàm chuẩn hóa bounding box sang định dạng YOLO
def convert_to_yolo_format(bbox, img_width, img_height):
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return x_center, y_center, width, height

# Xử lý dữ liệu
image_anno_pairs = []
counter = 0  # counter global

for path in [or_path, re_path]:
    anno_path = os.path.join(path, "annotations")  # Thư mục annotations
    img_base_path = os.path.join(path, "images")  # Thư mục gốc của images

    # Duyệt qua các thư mục con trong annotations
    for class_path in glob.glob(os.path.join(anno_path, "*")):
        class_name = os.path.basename(class_path)
        class_id = classes_idx[class_name]

        # Tìm tất cả file JSON trực tiếp trong class_path
        json_files = glob.glob(os.path.join(class_path, "*.json"))

        # Duyệt qua từng file JSON
        for json_file in json_files:
            # Trích xuất tên thư mục từ tên file JSON (bỏ đuôi .json)
            sub_path_name = os.path.basename(json_file).replace(".json", "")  # "07.or0002"

            # Tạo đường dẫn đến thư mục ảnh tương ứng
            img_dir = os.path.join(img_base_path, class_name, sub_path_name)  # data/or/images/fin_id/07.or0002

            # Đọc file JSON
            with open(json_file, "r") as f:
                json_data = json.load(f)

            # Duyệt qua từng ảnh trong _via_img_metadata
            for img_key, img_data in json_data["_via_img_metadata"].items():
                img_name = img_data["filename"]
                img_file = os.path.join(img_dir, img_name)

                # Đọc ảnh để lấy kích thước
                frame = cv2.imread(img_file)
                if frame is None:
                    continue
                height, width = frame.shape[:2]

                # Tìm vùng doc_quad đầu tiên và tạo label
                label_line = ""
                for region in img_data.get("regions", []):
                    if region.get("region_attributes", {}).get("field_name") == "doc_quad":
                        shape = region.get("shape_attributes", {})
                        points_x, points_y = shape.get("all_points_x"), shape.get("all_points_y")
                        if points_x and points_y:
                            bbox = polygon_to_bbox(points_x, points_y)
                            x_center, y_center, norm_width, norm_height = convert_to_yolo_format(bbox, width, height)
                            label_line = f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n"
                        break

                image_anno_pairs.append((img_file, label_line, f"{counter}.jpg"))
                counter += 1


# Chia dữ liệu thành train, val, test
random.shuffle(image_anno_pairs)
total = len(image_anno_pairs)
train_end = int(total * splits["train"])
val_end = train_end + int(total * splits["val"])

split_pairs = {
    "train": image_anno_pairs[:train_end],
    "val": image_anno_pairs[train_end:val_end],
    "test": image_anno_pairs[val_end:]
}

# Sao chép file và tạo label
for split, pairs in split_pairs.items():
    for img_path, label_line, img_name in pairs:
        # Đọc lại ảnh từ đường dẫn
        try:
            frame = cv2.imread(img_path)
            if frame is None:
                continue
        except Exception:
            continue

        # Sao chép ảnh
        output_img_path = os.path.join(output_path, split, img_name)
        cv2.imwrite(output_img_path, frame)

        # Tạo file label
        label_path = os.path.join(output_path.replace("images", "labels"), split, img_name.replace(".jpg", ".txt"))
        with open(label_path, "w") as f:
            f.write(label_line)