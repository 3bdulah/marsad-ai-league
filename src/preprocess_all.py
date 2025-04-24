import os
import json
import scipy.io
import shutil
import glob
from tqdm import tqdm

# Dataset directories
FDST_PATH = "dataset/FDST"
JHU_PATH = "dataset/JHU-CROWD++"
UCF_PATH = "dataset/UCF-QNRF"
OUTPUT_PATH = "dataset/processed"

# Create output folders
os.makedirs(f"{OUTPUT_PATH}/images", exist_ok=True)
os.makedirs(f"{OUTPUT_PATH}/labels", exist_ok=True)

def process_fdst():
    """ Process FDST JSON annotations to YOLO format """
    for subset in ["train_data", "test_data"]:
        subset_path = os.path.join(FDST_PATH, subset)

        for folder in tqdm(os.listdir(subset_path), desc=f"Processing FDST {subset}"):
            folder_path = os.path.join(subset_path, folder)
            img_path = os.path.join(folder_path, "001.jpg")
            json_path = os.path.join(folder_path, "001.json")

            if os.path.exists(img_path) and os.path.exists(json_path):
                new_img_name = f"FDST_{folder}.jpg"
                shutil.copy(img_path, f"{OUTPUT_PATH}/images/{new_img_name}")

                with open(json_path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        if "objects" not in data:
                            continue
                    except:
                        continue

                label_txt = []
                for obj in data["objects"]:
                    x, y, w, h = obj["bbox"]
                    x_center = x + w / 2
                    y_center = y + h / 2
                    label_txt.append(f"0 {x_center} {y_center} {w} {h}")

                with open(f"{OUTPUT_PATH}/labels/{new_img_name.replace('.jpg', '.txt')}", "w") as f:
                    f.write("\n".join(label_txt))

def process_jhu():
    """ Copy images and YOLO labels from JHU-CROWD++ """
    for subset in ["train", "valid", "test"]:
        img_dir = os.path.join(JHU_PATH, subset, "images")
        label_dir = os.path.join(JHU_PATH, subset, "labels")

        for img_file in tqdm(glob.glob(f"{img_dir}/*.jpg"), desc=f"Processing JHU {subset}"):
            shutil.copy(img_file, os.path.join(OUTPUT_PATH, "images"))

        for label_file in glob.glob(f"{label_dir}/*.txt"):
            shutil.copy(label_file, os.path.join(OUTPUT_PATH, "labels"))

def process_ucf():
    """ Convert UCF-QNRF .mat annotations to YOLO format """
    for subset in ["Train", "Test"]:
        subset_path = os.path.join(UCF_PATH, subset)

        for img_file in tqdm(glob.glob(f"{subset_path}/*.jpg"), desc=f"Processing UCF {subset}"):
            img_name = os.path.basename(img_file)
            mat_file = img_file.replace(".jpg", "_ann.mat")

            shutil.copy(img_file, os.path.join(OUTPUT_PATH, "images", img_name))

            if os.path.exists(mat_file):
                mat = scipy.io.loadmat(mat_file)
                annotations = mat["annPoints"]

                label_txt = []
                for point in annotations:
                    x, y = point
                    label_txt.append(f"0 {x} {y} 10 10")  # Approximate bbox

                with open(f"{OUTPUT_PATH}/labels/{img_name.replace('.jpg', '.txt')}", "w") as f:
                    f.write("\n".join(label_txt))

# Run all
if __name__ == "__main__":
    print("📦 Starting preprocessing...")
    process_fdst()
    process_jhu()
    process_ucf()
    print("✅ All datasets processed into YOLO format.")