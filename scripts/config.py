import torch
from utils import train
from clearml import Task
from argparse import Namespace

class Config:

    SEED: int = 42

    USE_IMAGE_ONLY: bool = True

    # Модели
    TEXT_MODEL_NAME: str = "bert-base-uncased"
    IMAGE_MODEL_NAME: str = "tf_efficientnet_b4"
    
    
    TEXT_MODEL_UNFREEZE: str = "encoder.layer.11|pooler"
    IMAGE_MODEL_UNFREEZE: str = "blocks.5|blocks.6|conv_head|bn2"
    
    # Гиперпараметры
    BATCH_SIZE: int = 4
    LR: float = 1e-3
    TEXT_LR: float = 3e-4
    IMAGE_LR: float = 1e-3
    ATTENTION_LR: float = 1e-3
    REGRESSOR_LR: float = 1e-3
    EPOCHS: int = 40
    DROPOUT: float = 0.25
    HIDDEN_DIM: int = 256

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
        task_name="Эксперимент_2_image",
        task_type=Task.TaskTypes.training
    )
    parameters = {
    'seed': 42,
    'use_image_only': True,

    'text_model_name': 'bert-base-uncased',
    'image_model_name': 'tf_efficientnet_b4',

    'text_model_unfreeze': 'encoder.layer.11|pooler',
    'image_model_unfreeze': 'blocks.5|blocks.6|conv_head|bn2',

    'batch_size': 4,
    'lr': 1e-3,
    'text_lr': 3e-4,
    'image_lr': 1e-3,
    'attention_lr': 1e-3,
    'regressor_lr': 1e-3,

    'epochs': 40,
    'dropout': 0.25,
    'hidden_dim': 256,

    'dish_csv_path': 'data/dish.csv',
    'ingr_csv_path': 'data/ingredients.csv',
    'img_dir': 'data/images',
    'save_path': 'best_model.pth',
}
    parameters = task.connect(parameters)

    train(cfg, device)

if __name__ == "__main__":
    main()
