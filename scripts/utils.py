import os
import random
from functools import partial
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torchmetrics
import timm
from transformers import AutoModel, AutoTokenizer
from dataset import MultimodalDataset, collate_fn, get_transforms, ImageOnlyDataset, image_only_collate_fn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from transformers import get_cosine_schedule_with_warmup

from clearml import Task

task = Task.init(
    project_name="Multimodal_Calory_Dish",
    task_name="Эксперимент_2_image",
    task_type=Task.TaskTypes.training
)

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

def set_requires_grad(module: nn.Module, unfreeze_pattern="", verbose=False):
    if len(unfreeze_pattern) == 0:
        for _, param in module.named_parameters():
            param.requires_grad = False
        return

    pattern = unfreeze_pattern.split("|")

    for name, param in module.named_parameters():
        if any([name.startswith(p) for p in pattern]):
            param.requires_grad = True
            if verbose:
                print(f"Разморожен слой: {name}")
        else:
            param.requires_grad = False


class MultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.image_model = timm.create_model(
            config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0
        )
        self.emb_dim = config.HIDDEN_DIM 

        self.text_proj = nn.Linear(self.text_model.config.hidden_size, config.HIDDEN_DIM)
        self.image_proj = nn.Linear(self.image_model.num_features, config.HIDDEN_DIM) # type: ignore

    def forward(self, input_ids, attention_mask, image):
        text_features = self.text_model(input_ids, attention_mask).last_hidden_state[:, 0, :]
        image_features = self.image_model(image)

        text_emb = self.text_proj(text_features)
        image_emb = self.image_proj(image_features)

        return text_emb, image_emb
    
class CrossAttentionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config 
        # Инициализация базовых моделей
        self.base_model = MultimodalModel(config)

        # Механизм внимания
        self.cross_attn1 = nn.MultiheadAttention(embed_dim=config.HIDDEN_DIM, 
                                                 num_heads=2, 
                                                 #dropout=config.DROPOUT
                                                 )
        self.cross_attn2 = nn.MultiheadAttention(embed_dim=config.HIDDEN_DIM, 
                                                 num_heads=2, 
                                                 #dropout=config.DROPOUT
                                                 )
        
        # LayerNorm и Dropout после attention
        self.post_attn_norm1 = nn.LayerNorm(config.HIDDEN_DIM)
        self.post_attn_norm2 = nn.LayerNorm(config.HIDDEN_DIM)
        self.post_attn_dropout = nn.Dropout(config.DROPOUT)
        in_dim = config.HIDDEN_DIM
       
        # Эксперимент_4
        self.regressor = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(128, 1),
        )

        # self.regressor = nn.Linear(config.HIDDEN_DIM, 1)
        
    def forward(self, input_ids, attention_mask, image):
       
        text_emb, image_emb = self.base_model(input_ids, attention_mask, image)

        text_emb = text_emb.unsqueeze(0)   # [1, batch, dim]
        image_emb = image_emb.unsqueeze(0) # [1, batch, dim]

        attended_emb1, _ = self.cross_attn1(
            query=text_emb,
            key=image_emb,
            value=image_emb
        )
        # Residual connection
        fused_emb1 = attended_emb1 + text_emb

        attended_emb2, _ = self.cross_attn2(
            query=fused_emb1,
            key=image_emb,
            value=image_emb
        )
        fused_emb2 = (attended_emb2 + fused_emb1).squeeze(0)

        output = self.regressor(fused_emb2).squeeze(-1) 
        return output

class ConcatFusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.base_model = MultimodalModel(config)
        in_dim = config.HIDDEN_DIM*2

        # Эксперимент 1
        # self.regressor = nn.Sequential(
        #     nn.Linear(in_dim, 1),
        # )

        # Эксперимент 2
        self.regressor = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(config.DROPOUT),
            nn.Linear(in_dim, 1)
        )

        # Эксперимент 3
        # self.regressor = nn.Sequential(
        #     nn.Linear(in_dim, 128),
        #     nn.LayerNorm(128),
        #     nn.ReLU(),
        #     nn.Dropout(config.DROPOUT),
        #     nn.Linear(128, 1),
        # )

        # Эксперимент 4
        # self.regressor = nn.Sequential(
        #     nn.LayerNorm(in_dim),
        #     nn.Linear(in_dim, 128),
        #     nn.ReLU(),
        #     nn.Dropout(config.DROPOUT),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Dropout(config.DROPOUT),
        #     nn.Linear(64, 1),
        # )

        # Эксперимент_5
        self.regressor = nn.Sequential(
            nn.Linear(in_dim, 256), 
            nn.LayerNorm(256), 
            nn.ReLU(), 
            nn.Dropout(config.DROPOUT), 
            nn.Linear(256, 1), 
        )


    def forward(self, input_ids, attention_mask, image):
        t, v = self.base_model(input_ids, attention_mask, image)
        x = torch.cat([t, v], dim=1) 
        return self.regressor(x).squeeze(-1)

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, x, y):
        return torch.sqrt(self.mse(x, y) + self.eps)

class ImageRegressor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_model = timm.create_model(
            config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0
        )
        in_dim = self.image_model.num_features  
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),  # type: ignore
            nn.Dropout(config.DROPOUT),
            nn.Linear(in_dim, 256),# type: ignore
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(256, 1),
        )

    def forward(self, image):
        feats = self.image_model(image)
        out = self.head(feats).squeeze(-1)
        return out

def train(config, device):
    seed_everything(config.SEED)

    # --- ВЕТВЛЕНИЕ ПО РЕЖИМУ ---
    if getattr(config, "USE_IMAGE_ONLY", False):
        # ===== Image-only =====
        model = ImageRegressor(config).to(device)

        # Размораживание только изображения
        set_requires_grad(model.image_model,
                          unfreeze_pattern=config.IMAGE_MODEL_UNFREEZE,
                          verbose=True)

        optimizer = AdamW([
            {'params': model.image_model.parameters(), 'lr': config.IMAGE_LR},
            {'params': model.head.parameters(),        'lr': config.REGRESSOR_LR},
        ], weight_decay=5e-5)

        criterion = nn.HuberLoss(delta=15.0) 

        # Трансформы и датасеты только для изображений
        train_tfms = get_transforms(config, ds_type="train")
        val_tfms   = get_transforms(config, ds_type="test")

        train_ds = ImageOnlyDataset(config, train_tfms, ds_type="train")  # type: ignore
        val_ds   = ImageOnlyDataset(config, val_tfms,   ds_type="test")   # type: ignore

        train_loader = DataLoader(
            train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
            num_workers=4, pin_memory=True, collate_fn=image_only_collate_fn
        )
        val_loader = DataLoader(
            val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
            num_workers=4, pin_memory=True, collate_fn=image_only_collate_fn
        )

        steps_per_epoch    = len(train_loader)
        num_training_steps = config.EPOCHS * steps_per_epoch
        num_warmup_steps   = int(0.1 * num_training_steps)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        plateau = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

        best_mae = float("inf")
        print("training started (IMAGE-ONLY)")
        for epoch in range(config.EPOCHS):
            model.train()
            total_loss = 0.0

            for batch in train_loader:
                optimizer.zero_grad()
                preds  = model(image=batch['image'].to(device))
                labels = batch['label'].to(device)

                loss = criterion(preds, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            train_mae = validate(model, train_loader, device, use_image_only=True)
            val_mae   = validate(model, val_loader,   device, use_image_only=True)
            plateau.step(val_mae)

            task.get_logger().report_scalar("Loss",      "train_loss", iteration=epoch, value=avg_train_loss)
            task.get_logger().report_scalar("MAE_train", "train_mae",  iteration=epoch, value=train_mae)
            task.get_logger().report_scalar("MAE_val",   "val_mae",    iteration=epoch, value=val_mae)

            print(f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {avg_train_loss:.4f} | Train MAE: {train_mae:.2f} | Val MAE: {val_mae:.2f}")

            if val_mae < best_mae:
                best_mae = val_mae
                torch.save(model.state_dict(), config.SAVE_PATH)
                print(f"New best (IMAGE-ONLY) model saved with MAE: {best_mae:.2f}")

    else:
        # ===== Multimodal (твой текущий) =====
        model = ConcatFusionModel(config).to(device)
        tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

        set_requires_grad(model.base_model.text_model,
                          unfreeze_pattern=config.TEXT_MODEL_UNFREEZE, verbose=True)
        set_requires_grad(model.base_model.image_model,
                          unfreeze_pattern=config.IMAGE_MODEL_UNFREEZE, verbose=True)

        optimizer = AdamW([
            {'params': model.base_model.text_model.parameters(), 'lr': config.TEXT_LR},
            {'params': model.base_model.image_model.parameters(), 'lr': config.IMAGE_LR},
            {'params': model.regressor.parameters(),              'lr': config.REGRESSOR_LR}
        ], weight_decay=1e-4)

        criterion = RMSELoss()

        transforms     = get_transforms(config)
        val_transforms = get_transforms(config, ds_type="test")
        train_dataset  = MultimodalDataset(config, transforms)
        val_dataset    = MultimodalDataset(config, val_transforms, ds_type="test")

        train_loader = DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
            collate_fn=partial(collate_fn, tokenizer=tokenizer),
            persistent_workers=True, num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
            collate_fn=partial(collate_fn, tokenizer=tokenizer),
            persistent_workers=True, num_workers=4, pin_memory=True
        )

        steps_per_epoch    = len(train_loader)
        num_training_steps = config.EPOCHS * steps_per_epoch
        num_warmup_steps   = int(0.1 * num_training_steps)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        best_mae = float("inf")
        print("training started (MULTIMODAL)")
        for epoch in range(config.EPOCHS):
            model.train()
            total_loss = 0.0

            for batch in train_loader:
                optimizer.zero_grad()
                preds = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    image=batch['image'].to(device),
                )
                labels = batch['label'].to(device)

                loss = criterion(preds, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            train_mae = validate(model, train_loader, device, use_image_only=False)
            val_mae   = validate(model, val_loader,   device, use_image_only=False)

            task.get_logger().report_scalar("Loss",      "train_loss", iteration=epoch, value=avg_train_loss)
            task.get_logger().report_scalar("MAE_train", "train_mae",  iteration=epoch, value=train_mae)
            task.get_logger().report_scalar("MAE_val",   "val_mae",    iteration=epoch, value=val_mae)

            print(f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {avg_train_loss:.4f} | Train MAE: {train_mae:.2f} | Val MAE: {val_mae:.2f}")

            if val_mae < best_mae:
                best_mae = val_mae
                torch.save(model.state_dict(), config.SAVE_PATH)
                print(f"New best (MULTIMODAL) model saved with MAE: {best_mae:.2f}")


@torch.no_grad()
def validate(model, val_loader, device, use_image_only: bool):
    model.eval()
    mae_metric = torchmetrics.MeanAbsoluteError().to(device)
    mae_metric.reset()

    with torch.no_grad():
        for batch in val_loader:
            if use_image_only:
                preds = model(image=batch['image'].to(device))
            else:
                preds = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    image=batch['image'].to(device),
                )

            labels = batch['label'].to(device)
            mae_metric.update(preds, labels)

    return mae_metric.compute().item()