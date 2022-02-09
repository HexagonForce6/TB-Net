
import os
import shutil
import cv2
from tqdm import tqdm
import pandas as pd
import preprocess_data


DATASET_PATH = "Dataset"

DATASET_FLAT_PATH = "Dataset_Flat"
DATASET_SPLIT_PATH = "Dataset_Split"

TRAIN_SPLIT_PATH = os.path.join(DATASET_SPLIT_PATH, "train")
VAL_SPLIT_PATH = os.path.join(DATASET_SPLIT_PATH, "val")
TEST_SPLIT_PATH = os.path.join(DATASET_SPLIT_PATH, "test")

TRAIN_SPLIT_CSV = "train_split.csv"
VAL_SPLIT_CSV = "val_split.csv"
TEST_SPLIT_CSV = "test_split.csv"

PREPROCESSED_PATH = "Preprocessed_Images"
TRAIN_IMG_PATH = os.path.join(PREPROCESSED_PATH, "train")
VAL_IMG_PATH = os.path.join(PREPROCESSED_PATH, "val")
TEST_IMG_PATH = os.path.join(PREPROCESSED_PATH, "test")

CLASSES = ['Normal', 'Tuberculosis']


# make directories
os.makedirs(DATASET_FLAT_PATH, exist_ok = True)
os.makedirs(DATASET_SPLIT_PATH, exist_ok = True)
os.makedirs(TRAIN_SPLIT_PATH, exist_ok=True)
os.makedirs(VAL_SPLIT_PATH, exist_ok=True)
os.makedirs(TEST_SPLIT_PATH, exist_ok=True)

for class_id in CLASSES:
    os.makedirs(os.path.join(TRAIN_SPLIT_PATH, str(class_id)), exist_ok=True)
    os.makedirs(os.path.join(VAL_SPLIT_PATH, str(class_id)), exist_ok=True)
    os.makedirs(os.path.join(TEST_SPLIT_PATH, str(class_id)), exist_ok=True)
    os.makedirs(os.path.join(TRAIN_IMG_PATH, str(class_id)), exist_ok=True)
    os.makedirs(os.path.join(VAL_IMG_PATH, str(class_id)), exist_ok=True)
    os.makedirs(os.path.join(TEST_IMG_PATH, str(class_id)), exist_ok=True)


# flatten dataset
for class_id in CLASSES:
    subdir = os.path.join(DATASET_PATH, class_id)
    for img_path in os.listdir(subdir):
        shutil.copy(os.path.join(subdir, img_path), DATASET_FLAT_PATH)

print("Dataset flattened")


# split dataset
def classify_data(df, img_path):
    for i, row in tqdm(df.iterrows(), total=len(df)):
        class_id = row[1]
        class_id = "Normal" if class_id == 0 else "Tuberculosis"
        shutil.copy(os.path.join(DATASET_FLAT_PATH, row[0]),
                    os.path.join(img_path, str(class_id)))

df_train = pd.read_csv(TRAIN_SPLIT_CSV, header=None)
df_val = pd.read_csv(VAL_SPLIT_CSV, header=None)
df_test = pd.read_csv(TEST_SPLIT_CSV, header=None)

classify_data(df_train, TRAIN_SPLIT_PATH)
classify_data(df_val, VAL_SPLIT_PATH)
classify_data(df_test, TEST_SPLIT_PATH)

print("Dataset splitted")


# preprocess data for each phase
def do_preprocess(phase):
    for class_id in CLASSES:
        src = os.path.join(DATASET_SPLIT_PATH, phase, class_id)
        dest = os.path.join(PREPROCESSED_PATH, phase, class_id)

        if phase == "train":
            for img_name in os.listdir(src):       
                img = preprocess_data.preprocess_image_train(os.path.join(src, img_name))
                save_path = os.path.join(dest, img_name)
                cv2.imwrite(save_path, img)

        if phase == "val":
            for img_name in os.listdir(src):       
                img = preprocess_data.preprocess_image_inference(os.path.join(src, img_name))
                save_path = os.path.join(dest, img_name)
                cv2.imwrite(save_path, img)

        if phase == "test":
            for img_name in os.listdir(src):       
                img = preprocess_data.preprocess_image_inference(os.path.join(src, img_name))
                save_path = os.path.join(dest, img_name)
                cv2.imwrite(save_path, img)

do_preprocess("train")
print("Training data preprocessed")

do_preprocess("val")
print("Validation data preprocessed")

do_preprocess("test")
print("Testing data preprocessed")
