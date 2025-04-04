import os
import glob
import cv2
import shutil
import json
import random
from pprint import pprint

# Định nghĩa đường dẫn
BASE_DIR = "/home/nhung/Desktop/DLC2021"
or_path = os.path.join(BASE_DIR, "data", "or")
re_path = os.path.join(BASE_DIR, "data", "re")
splits = {"train": 0.7, "val": 0.2, "test": 0.1}
output_path = os.path.join(BASE_DIR, "re_or_dataset")

# Xóa và tạo lại thư mục đầu ra
if os.path.isdir(output_path):
    shutil.rmtree(output_path)
os.makedirs(os.path.join(output_path, "train/recap"))
os.makedirs(os.path.join(output_path, "train/original"))
os.makedirs(os.path.join(output_path, "val/recap"))
os.makedirs(os.path.join(output_path, "val/original"))
os.makedirs(os.path.join(output_path, "test/recap"))
os.makedirs(os.path.join(output_path, "test/original"))

# Hàm chuyển đổi polygon thành bounding box
def polygon_to_bbox(points_x, points_y):
    x_min = min(points_x)
    x_max = max(points_x)
    y_min = min(points_y)
    y_max = max(points_y)
    return [x_min, y_min, x_max, y_max]

# Xử lý dữ liệu
image_info = []
counter = 0

for path in [or_path, re_path]:
    label = "original" if "or" in path else "recap"
    anno_path = os.path.join(path, "annotations")  # Thư mục annotations
    img_base_path = os.path.join(path, "images")  # Thư mục gốc của images

    # Duyệt qua các thư mục con trong annotations
    for class_path in glob.glob(os.path.join(anno_path, "*")):
        class_name = os.path.basename(class_path)

        # Tìm tất cả file JSON trực tiếp trong class_path
        json_files = glob.glob(os.path.join(class_path, "*.json"))

        # Duyệt qua từng file JSON
        for json_file in json_files:
            # Trích xuất tên thư mục từ tên file JSON
            sub_path_name = os.path.basename(json_file).replace(".json", "")  # "00.re0001"

            # Tạo đường dẫn đến thư mục ảnh tương ứng
            img_dir = os.path.join(img_base_path, class_name, sub_path_name)  # data/re/images/alb_id/00.re0001

            # Đọc file JSON
            with open(json_file, "r") as f:
                json_data = json.load(f)

            # Duyệt qua từng ảnh trong _via_img_metadata
            for img_key, img_data in json_data["_via_img_metadata"].items():
                img_name = img_data["filename"]
                img_file = os.path.join(img_dir, img_name)

                # Đọc ảnh
                image = cv2.imread(img_file)
                if image is None:
                    continue

                # Tìm vùng doc_quad và crop vùng tài liệu
                for region in img_data.get("regions", []):
                    if region.get("region_attributes", {}).get("field_name") == "doc_quad":
                        shape = region.get("shape_attributes", {})
                        points_x, points_y = shape.get("all_points_x"), shape.get("all_points_y")
                        if points_x and points_y:
                            bbox = polygon_to_bbox(points_x, points_y)
                            x_min, y_min, x_max, y_max = bbox

                            # Cắt vùng tài liệu
                            cropped_image = image[y_min:y_max, x_min:x_max]
                            if cropped_image.size == 0:
                                continue

                            # Tính tỷ lệ w/h và h/w
                            height, width = cropped_image.shape[:2]
                            aspect_ratio_wh = width / height if height > 0 else float('inf')
                            aspect_ratio_hw = height / width if width > 0 else float('inf')

                            # Kiểm tra điều kiện tỷ lệ > 4
                            if aspect_ratio_wh > 4 or aspect_ratio_hw > 4:
                                print(f"Bỏ qua ảnh {img_file} do tỷ lệ w/h hoặc h/w > 4")
                                continue

                            cropped_name = f"{label}_{class_name}_{counter}.jpg"

                            # Lưu ảnh crop vào thư mục tạm
                            temp_img_path = os.path.join(BASE_DIR, "temp", cropped_name)
                            os.makedirs(os.path.dirname(temp_img_path), exist_ok=True)
                            cv2.imwrite(temp_img_path, cropped_image)

                            image_info.append({
                                "path": temp_img_path,  # Lưu đường dẫn thay vì ảnh
                                "label": label,
                                "doc_type": class_name,
                                "filename": cropped_name
                            })
                            counter += 1
                        break

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