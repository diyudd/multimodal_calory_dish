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
from dataset import MultimodalDataset, collate_fn, get_transforms

from clearml import Task

task = Task.init(
    project_name="Multimodal_Calory_Dish",
    task_name="CrossAttention_Training",
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

        # Нормализация перед вниманием
        #self.text_norm = nn.LayerNorm(config.HIDDEN_DIM)
        #self.image_norm = nn.LayerNorm(config.HIDDEN_DIM)
        #self.dropout = nn.Dropout(config.DROPOUT)


    def forward(self, input_ids, attention_mask, image):
        text_features = self.text_model(input_ids, attention_mask).last_hidden_state[:,  0, :]
        image_features = self.image_model(image)

        text_emb = self.text_proj(text_features)
        image_emb = self.image_proj(image_features)

        # Без нормализаци
        # text_emb = self.dropout(text_emb)
        # image_emb = self.dropout(image_emb)

        # С нормализацией
        #text_emb = self.dropout(self.text_norm(text_emb))
        #image_emb = self.dropout(self.image_norm(image_emb))

        return text_emb, image_emb
    
class CrossAttentionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Инициализация базовых моделей
        self.base_model = MultimodalModel(config)
        
        # Нормализация перед вниманием
        #self.text_norm = nn.LayerNorm(config.HIDDEN_DIM)
        #self.image_norm = nn.LayerNorm(config.HIDDEN_DIM)

        # Механизм внимания
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.HIDDEN_DIM, 
            num_heads=4,
            dropout=config.DROPOUT)
        
        # LayerNorm и Dropout после attention
        #self.post_attn_norm = nn.LayerNorm(config.HIDDEN_DIM)
        #self.post_attn_dropout = nn.Dropout(config.DROPOUT)
        
        # Регрессор с нормализацией и дропаут
        self.regressor = nn.Sequential(
            nn.LayerNorm(config.HIDDEN_DIM),  # Дополнительная нормализация после attention
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2), 
            nn.ReLU(),                               
            nn.Linear(config.HIDDEN_DIM // 2, 1)
        )
        # self.regressor = nn.Linear(config.HIDDEN_DIM, 1)
        
    def forward(self, input_ids, attention_mask, image):
        # Получаем эмбеддинги модальностей
        text_emb, image_emb = self.base_model(input_ids, attention_mask, image)

        # Нормализация
        #text_emb = self.text_norm(text_emb)
        #image_emb = self.image_norm(image_emb)

        text_emb = text_emb.unsqueeze(0)   # [1, batch, dim]
        image_emb = image_emb.unsqueeze(0) # [1, batch, dim]

        attended_emb, _ = self.cross_attn(
            query=text_emb,
            key=image_emb,
            value=image_emb
        )
        
        fused_emb = attended_emb.squeeze(0)  # [batch, embed_dim]

        # Residual connection + LayerNorm + Dropout
        # fused_emb = self.post_attn_dropout(self.post_attn_norm(fused_emb + text_emb.squeeze(0)))

        output = self.regressor(fused_emb).squeeze(1) 
        return output


def train(config, device):
    seed_everything(config.SEED)

    # Инициализация модели
    model = CrossAttentionModel(config).to(device) 
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

    set_requires_grad(model.base_model.text_model,
                      unfreeze_pattern=config.TEXT_MODEL_UNFREEZE, verbose=True)
    set_requires_grad(model.base_model.image_model,
                      unfreeze_pattern=config.IMAGE_MODEL_UNFREEZE, verbose=True)

    #optimizer = AdamW(model.parameters(), lr=config.LR)
    optimizer = AdamW([
    {'params': model.base_model.text_model.parameters(), 'lr': config.TEXT_LR},
    {'params': model.base_model.image_model.parameters(), 'lr': config.IMAGE_LR},
    {'params': model.cross_attn.parameters(), 'lr': config.ATTENTION_LR},
    {'params': model.regressor.parameters(), 'lr': config.REGRESSOR_LR}
    ], weight_decay=1e-4)

    criterion = nn.L1Loss()

    # Загрузка данных
    transforms = get_transforms(config)
    val_transforms = get_transforms(config, ds_type="test")

    train_dataset = MultimodalDataset(config, transforms)
    val_dataset = MultimodalDataset(config, val_transforms, ds_type="test")

    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE,
                              shuffle=True,
                              collate_fn=partial(collate_fn,
                                                 tokenizer=tokenizer),
                              num_workers=4,
                              pin_memory=True)
    
    val_loader = DataLoader(val_dataset,
                            batch_size=config.BATCH_SIZE,
                            shuffle=False,
                            collate_fn=partial(collate_fn,
                                               tokenizer=tokenizer),
                            num_workers=4,
                            pin_memory=True)

    best_mae = float("inf")

    print("training started")
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            preds = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                image=batch['image'].to(device)
            )
            labels = batch['label'].to(device)

            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        # Валидация
        val_mae = validate(model, val_loader, device)

        task.get_logger().report_scalar("Loss", "train_loss", iteration=epoch, value=avg_train_loss)
        task.get_logger().report_scalar("MAE", "val_mae", iteration=epoch, value=val_mae)

        print(f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val MAE: {val_mae:.2f}")

        # Сохраняем лучшую модель по MAE
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), config.SAVE_PATH)
            print(f"New best model saved with MAE: {best_mae:.2f}")


def validate(model, val_loader, device):
    model.eval()
    mae_metric = torchmetrics.MeanAbsoluteError().to(device)

    with torch.no_grad():
        for batch in val_loader:
            preds = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                image=batch['image'].to(device)
            )
            labels = batch['label'].to(device)

            labels = batch['label'].to(device)
            mae_metric.update(preds, labels)

    return mae_metric.compute().item()