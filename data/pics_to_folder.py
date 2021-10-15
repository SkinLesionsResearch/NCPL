import os
import shutil
from pathlib import Path

import pandas as pd

root_proj_path = "/home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer/"
ham_path = os.path.join(root_proj_path, "data_ham10000")
descriptor_path = os.path.join(ham_path, "descriptors")
dataset_path = os.path.join(ham_path, "datasets")

total_descriptor_path = os.path.join(descriptor_path, "HAM10000_metadata.tab")

df = pd.read_csv(total_descriptor_path, sep="\t")
df = df.iloc[:, 0:3]
df.columns = ["id", "fname", "type"]
df["fname"] = (df["fname"] + ".jpg")
df_grp = df.groupby(by=["type"])
type_set = set(df["type"])
print(type_set)
classes_path = os.path.join(ham_path, "dataset_classes")
if not os.path.exists(classes_path):
    os.mkdir(classes_path)
for t in type_set:
    if not os.path.exists(os.path.join(classes_path, t)):
        os.mkdir(os.path.join(classes_path, t))

i = 0
for img in Path(dataset_path).iterdir():
    # print(i, "-th pic")
    img_fname = str(img).split("/")[-1]
    img_type_ser = df[df["fname"] == img_fname]["type"]
    if img_type_ser.size > 1 or img_type_ser.size <= 0:
        print(i, "-th pic")
        print("fname", img_fname)
        print(img_type_ser)
        print(df[df["fname"] == img_fname])
        print()
        continue
    img_type = img_type_ser.iloc[0]
    img_dir_path = os.path.join(classes_path, img_type)
    img_target_path = os.path.join(img_dir_path, img_fname)
    # print(t)
    # print(img)
    shutil.copy(img, img_target_path)
    i += 1
    # break
