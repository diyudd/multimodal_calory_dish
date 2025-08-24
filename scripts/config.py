print("start")
import torch
from utils import train
from clearml import Task

task = Task.init(
    project_name="Multimodal_Calory_Dish",
    task_name="CrossAttention_Training",
    task_type=Task.TaskTypes.training
)

class Config:

    SEED = 42
    # Модели
    TEXT_MODEL_NAME = "bert-base-uncased"
    IMAGE_MODEL_NAME = "tf_efficientnet_b0"
    
    # Какие слои размораживаем - совпадают с нэймингом в моделях
    TEXT_MODEL_UNFREEZE = "encoder.layer.11|pooler"
    IMAGE_MODEL_UNFREEZE = "blocks.6|conv_head|bn2"
    
    # Гиперпараметры
    BATCH_SIZE = 256
    LR = 1e-3
    TEXT_LR = 3e-5
    IMAGE_LR = 1e-4
    ATTENTION_LR = 1e-3
    REGRESSOR_LR = 1e-3
    EPOCHS = 30
    DROPOUT = 0.3
    HIDDEN_DIM = 256
    
    # Пути
    DISH_CSV_PATH = "data/dish.csv"
    INGR_CSV_PATH = "data/ingredients.csv"
    IMG_DIR = "data/images"
    SAVE_PATH = "best_model.pth"
    



device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

cfg = Config()
task.connect(cfg)

train(cfg, device)
