from src import (
    CSTA, get_config, get_video_dataset, set_all_seeds, train_epoch, evaluate
)
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, WeightedRandomSampler
from accelerate import Accelerator
import torch, os, random, accelerate, logging, datetime, argparse, copy
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pandas as pd
from torchvision.io import read_video
import torch.optim as optim
from warnings import filterwarnings
import torchvision.transforms as transforms
import math

def main():
    parser = argparse.ArgumentParser(description="Train CSTA for task 0.")
    parser.add_argument('--config', type=str, default="config/train_configs/UCF101/train_task0.yml", help="Config File Path for Training Task 0.")
    parser.add_argument('--save_folder', type=str, default="model_save/test", help="Folder to save everything.")
    args = parser.parse_args()
    config_path = args.config
    config = get_config(config_path)
    
    # Create workdirectory
    work_dir = os.path.join(args.save_folder, config.data.dataset_name, "task_0")
    
    # setup directory. work_dir -) a. config, b. logs, c. model
    save_model_config_dir = os.path.join(work_dir, "config")
    log_dir = os.path.joing(work_dir, "logs")
    model_save_dir = os.path.join(work_dir, "model")
    os.makedirs(save_model_config_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    
    # setup logger
    logging.basicConfig(
        filename=os.path.join(log_dir, "train_task0.log"),
        filemode="a",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    logging.info(f"\n\nTraining for task 0 starting on: {datetime.datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}\n\n")
    
    # create dataset
    dataset = get_video_dataset(config)
    train_dataset, test_dataset, valid_dataset, id2label, label2id = dataset["train"], dataset["test"], dataset["valid"], dataset["id2label"], dataset["label2id"]
    TrainingConfigs = config.train
        
    # fix randomness
    set_all_seeds(TrainingConfigs.random_seed)
    
    # get dataloaders
    train_dataloader = DataLoader(train_dataset, 
                                batch_size=TrainingConfigs.training_batch_size, 
                                shuffle=True, 
                                pin_memory=TrainingConfigs.dataloader_pin_memory, 
                                persistent_workers=TrainingConfigs.dataloader_persistent_workers,
                                num_workers=TrainingConfigs.dataloader_num_workers,
                                )
    valid_dataloader = DataLoader(valid_dataset, 
                                batch_size=TrainingConfigs.evaluation_batch_size, 
                                shuffle=False,
                                pin_memory=TrainingConfigs.dataloader_pin_memory, 
                                persistent_workers=TrainingConfigs.dataloader_persistent_workers,
                                num_workers=TrainingConfigs.dataloader_num_workers,
                                )
    test_dataloader = DataLoader(test_dataset, 
                                batch_size=TrainingConfigs.evaluation_batch_size, 
                                shuffle=False,
                                pin_memory=TrainingConfigs.dataloader_pin_memory, 
                                persistent_workers=TrainingConfigs.dataloader_persistent_workers,
                                num_workers=TrainingConfigs.dataloader_num_workers,
                                )
    
    # now do model
    model = CSTA(config_file=config_path)
    model.prepare_architecture_for_current_task()
    
    # optimizer = optim.SGD(model.parameters(), lr = TrainingConfigs.learning_rate, weight_decay=TrainingConfigs.weight_decay)
    optimizer = optim.AdamW(model.parameters(), lr = TrainingConfigs.learning_rate, betas = TrainingConfigs.adamw_betas, weight_decay=TrainingConfigs.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=TrainingConfigs.T_max, eta_min=TrainingConfigs.eta_min)
    
    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader, scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, valid_dataloader, test_dataloader, scheduler
        )

    best_loss = float('inf')
    best_acc = 0.0
    for epoch in range(TrainingConfigs.num_training_epochs):
        train_loss, _ = train_epoch(model, train_dataloader, optimizer, accelerator, epoch)
        eval_loss, eval_acc = evaluate(model, eval_dataloader, accelerator, epoch)
        
        if eval_loss < best_loss:
            best_loss = eval_loss
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), os.path.join(model_save_dir, "checkpoints", f'best_model_epoch_{epoch}.pth'))
        
        elif eval_acc > best_acc:
            best_acc = eval_acc
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), os.path.join(model_save_dir, "checkpoints", f'best_model_epoch_{epoch}.pth'))

        if epoch > TrainingConfigs.warmup_epochs:
            scheduler.step()

    accelerator.wait_for_everyone()
    accelerator.end_training()
    unwrapped_model = accelerator.unwrap_model(model)
    torch.save(unwrapped_model.state_dict(), os.path.join(model_save_dir, "checkpoints", f'final_model.pth'))
    
    eval_loss, eval_acc = evaluate(model, test_dataloader, accelerator, epoch)
    logging.info(
        f"Test Performance: "
        f"Average Loss: {eval_loss:.4f}, "
        f"Average Accuracy: {eval_acc:.4f}, "
    )

if __name__ == "__main__":
    filterwarnings("ignore")
    main()