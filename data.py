import os

import pandas as pd

from config import label_mapping

from PIL import Image
from tqdm.auto import tqdm


def resize(array, size, keep_ratio=False, resample=Image.LANCZOS):
    img = Image.fromarray(array)

    if keep_ratio:
        img.thumbnail((size, size), resample)
    else:
        img = img.resize((size, size), resample)

    return img


def assign_label(row):
    row = row[
        [
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Enlarged Cardiomediastinum",
            "Fracture",
            "Lung Lesion",
            "Lung Opacity",
            "No Finding",
            "Pleural Effusion",
            "Pleural Other",
            "Pneumonia",
            "Pneumothorax",
        ]
    ]
    disease = row.idxmax()
    return label_mapping[disease]


def preprocess():
    # label file load
    df = pd.read_csv("./data/mimic-cxr-label-skt.csv")
    df = df[
        [
            "subject_id",
            "study_id",
            "dicom_id",
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Enlarged Cardiomediastinum",
            "Fracture",
            "Lung Lesion",
            "Lung Opacity",
            "No Finding",
            "Pleural Effusion",
            "Pleural Other",
            "Pneumonia",
            "Pneumothorax",
        ]
    ]

    # Processing path
    image_pathes = []
    root_path = "./data/img"
    for _, row in tqdm(df.iterrows()):
        subject_id = str(row["subject_id"])
        study_id = str(row["study_id"])
        dicom_id = str(row["dicom_id"])

        to_path = os.path.join(
            root_path,
            f"p{subject_id[:2]}",
            f"p{subject_id}",
            f"s{study_id}",
            f"{dicom_id}.jpg",
        )

        image_pathes.append(to_path)
    df["image_path"] = image_pathes
    df["label"] = df.apply(assign_label, axis=1)
    df = df[["image_path", "label"]]

    sampled_df = df.groupby("label", group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), 25), random_state=42)
    )

    return sampled_df


def load_data(data_path):
    df = pd.read_csv(data_path)
    return df


if __name__ == "__main__":
    train_data_df = preprocess()
    train_data_df.to_csv("sampled_data.csv", index=False, encoding="utf-8")
