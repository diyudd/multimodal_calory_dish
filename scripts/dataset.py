import torch
from torch.utils.data import Dataset

from PIL import Image
import timm
import numpy as np
import pandas as pd

from transformers import AutoTokenizer

import albumentations as A


class MultimodalDataset(Dataset):

    def __init__(self, config, transforms, ds_type="train"):
         
        df = pd.read_csv(config.DISH_CSV_PATH)
        self.ingr_df = pd.read_csv(config.INGR_CSV_PATH)
        
        self.df = df[df["split"] == ds_type].reset_index(drop=True)

        self.image_cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
        self.transforms = transforms
        self.img_dir = config.IMG_DIR

        
        self.ingr_dict = {f"ingr_{int(row['id']):09d}": row['ingr'] for _, row in self.ingr_df.iterrows()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        ingr_ids = self.df.loc[idx, "ingredients"].split(";") # type: ignore
        text = " ".join([self.ingr_dict.get(i, "") for i in ingr_ids]) # type: ignore
        
        label = self.df.loc[idx, "total_calories"]

        dish_id = self.df.loc[idx, "dish_id"]
        img_path = f"{self.img_dir}/{dish_id}/rgb.png"
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = torch.randint(0, 255, (*self.image_cfg.input_size[1:], # type: ignore
                                           self.image_cfg.input_size[0])).to( # pyright: ignore[reportOptionalMemberAccess]
                                               torch.float32)

        image = self.transforms(image=np.array(image))["image"]

        return {"label": torch.tensor(label, dtype=torch.float32), 
                "image": image, 
                "text": text
                }


def collate_fn(batch, tokenizer):
    texts = [item["text"] for item in batch]
    images = torch.stack([item["image"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)


    tokenized_input = tokenizer(texts,
                                return_tensors="pt",
                                padding="max_length",
                                truncation=True)

    return {
        "label": labels,
        "image": images,
        "input_ids": tokenized_input["input_ids"],
        "attention_mask": tokenized_input["attention_mask"]
    }


def get_transforms(config, ds_type="train"):
    cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)  # cfg.input_size = (C,H,W)

    H, W = cfg.input_size[1], cfg.input_size[2]  # type: ignore # целевой размер

    if ds_type == "train":
        transforms = A.Compose([
            A.Resize(height=H, width=W),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02, p=0.3),
            A.Normalize(mean=cfg.mean, std=cfg.std), # type: ignore
            A.ToTensorV2()
        ], seed=42)

        # transforms = A.Compose([
        #     A.RandomResizedCrop(height=H, width=W, scale=(0.9, 1.1), ratio=(0.95, 1.05), p=1.0),
        #     A.HorizontalFlip(p=0.5),
        #     A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
        #     A.Normalize(mean=cfg.mean, std=cfg.std),
        #     A.ToTensorV2()
        # ], seed=42)

    else:
        transforms = A.Compose([
            A.Resize(height=H, width=W),
            A.CenterCrop(height=H, width=W),
            A.Normalize(mean=cfg.mean, std=cfg.std), # type: ignore
            A.ToTensorV2()
        ], seed=42)

    return transforms

# def get_transforms(config, ds_type="train"):
#     cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)

#     if ds_type == "train":
#         transforms = A.Compose(
#             [
#                 A.SmallestMaxSize(
#                     max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0), # type: ignore
#                 A.RandomCrop(
#                     height=cfg.input_size[1], width=cfg.input_size[2], p=1.0), # type: ignore
#                 A.Affine(scale=(0.9, 1.1),
#                          rotate=(-10, 10),
#                          translate_percent=(-0.05, 0.05),
#                          shear=(-8, 8),
#                          fill=0,
#                          p=0.8),
#                 A.CoarseDropout(
#                     num_holes_range=(2, 8),
#                     hole_height_range=(int(0.07 * cfg.input_size[1]), # type: ignore
#                                        int(0.15 * cfg.input_size[1])), # type: ignore
#                     hole_width_range=(int(0.1 * cfg.input_size[2]), # type: ignore
#                                       int(0.15 * cfg.input_size[2])), # type: ignore
#                     fill=0,
#                     p=0.5),
#                 A.ColorJitter(brightness=0.15,
#                               contrast=0.15,
#                               saturation=0.15,
#                               hue=0.1,
#                               p=0.7),
#                 A.Normalize(mean=cfg.mean, std=cfg.std), # type: ignore
#                 A.ToTensorV2(p=1.0)
#             ],
#             seed=42,
#         )
#     else:
#         transforms = A.Compose(
#             [
#                 A.SmallestMaxSize(
#                     max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0), # type: ignore
#                 A.CenterCrop(
#                     height=cfg.input_size[1], width=cfg.input_size[2], p=1.0), # type: ignore
#                 A.Normalize(mean=cfg.mean, std=cfg.std), # type: ignore
#                 A.ToTensorV2(p=1.0)
#             ],
#             seed=42,
#         )

#     return transforms

class ImageOnlyDataset(Dataset):
    def __init__(self, config, transforms, ds_type: str = "train"):
        # читаем общий dish.csv и фильтруем по split
        df = pd.read_csv(config.DISH_CSV_PATH)
        self.df = df[df["split"] == ds_type].reset_index(drop=True)

        self.image_cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)
        self.transforms = transforms
        self.img_dir = config.IMG_DIR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.loc[idx]
        label = self.df.loc[idx, "total_calories"]
        dish_id = row["dish_id"]

        img_path = f"{self.img_dir}/{dish_id}/rgb.png"

        try:
            image = Image.open(img_path).convert("RGB")
            image = np.array(image)
        except Exception:
            # fallback: синтетика подходящего размера (редко понадобится)
            H, W = self.image_cfg.input_size[1], self.image_cfg.input_size[2]  # type: ignore
            image = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)

        # применяем аугментации/нормализацию
        image = self.transforms(image=image)["image"] 

        return {
            "label": torch.tensor(label, dtype=torch.float32),
            "image": image,
            "dish_id": dish_id,
        }


def image_only_collate_fn(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
    out = {
        "image": images,
        "label": labels, 
    }
    return out