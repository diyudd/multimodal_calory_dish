import torch
from utils import train
from clearml import Task

class Config:

    SEED: int = 42
    # Модели
    TEXT_MODEL_NAME: str = "bert-base-uncased"
    IMAGE_MODEL_NAME: str = "tf_efficientnet_b4"
    
    
    TEXT_MODEL_UNFREEZE: str = "encoder.layer.11|pooler"
    IMAGE_MODEL_UNFREEZE: str = "blocks.6|conv_head|bn2"
    
    # Гиперпараметры
    BATCH_SIZE: int = 128
    LR: float = 1e-3
    TEXT_LR: float = 3e-5
    IMAGE_LR: float = 1e-4
    ATTENTION_LR: float = 1e-3
    REGRESSOR_LR: float = 1e-3
    EPOCHS: int = 50
    DROPOUT: float = 0.3
    HIDDEN_DIM: int = 256
    
    MASS_MEAN: float = 214.98   
    MASS_STD:  float = 161.50  

    # Пути
    DISH_CSV_PATH: str = "data/dish.csv"
    INGR_CSV_PATH: str = "data/ingredients.csv"
    IMG_DIR: str = "data/images"
    SAVE_PATH: str = "best_model.pth"
    

def main():
    print("start")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    cfg = Config()

    # ClearML
    task = Task.init(
        project_name="Multimodal_Calory_Dish",
        task_name="CrossAttention_Training",
        task_type=Task.TaskTypes.training
    )
    task.connect(cfg)

    train(cfg, device)

if __name__ == "__main__":
    main()
