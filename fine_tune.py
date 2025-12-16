from warnings import filterwarnings
filterwarnings("ignore")
from src import (
    CSTA, get_config, get_video_dataset, set_all_seeds, train_epoch, evaluate, get_video_dataset_for_ft
)
from torch.utils.data import DataLoader
from accelerate import Accelerator
import torch, os, logging, datetime, argparse, shutil
import torch.optim as optim

def main():
    parser = argparse.ArgumentParser(description="Fine-tuning CSTA.")
    parser.add_argument('--config', type=str, default="config/train_configs/dummy/train_task1.yml", help="Config File Path for Fine tuning.")
    parser.add_argument('--save_folder', type=str, default="model_save/test", help="Folder to save everything.")
    args = parser.parse_args()
    config_path = args.config
    config = get_config(config_path)

    # fix randomness
    TrainingConfigs = config.train
    set_all_seeds(TrainingConfigs.random_seed)
    
    # Create workdirectory
    work_dir = os.path.join(args.save_folder, config.data.dataset_name, f"task_{config.task.task_n}")
    
    # setup directory
    save_model_config_dir = os.path.join(work_dir, "config")
    log_dir = os.path.join(work_dir, "logs")
    model_save_dir = os.path.join(work_dir, "model")
    os.makedirs(save_model_config_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(model_save_dir, "checkpoints"), exist_ok=True)

    # setup logger
    logging.basicConfig(
        filename=os.path.join(log_dir, f"train_task{config.task.task_n}.log"),
        filemode="w",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    logging.info(f"\n\nTraining for task {config.task.task_n} starting on: {datetime.datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}\n")
    logging.info(f"The config is being used from file: {config_path}")

    # Check memory bank
    logging.info(f"\nChecking for pre-calculated features and relations...\n")
    assert os.path.exists(config.loss.temporal_features_path) and os.path.exists(config.loss.spatial_features_path) and os.path.exists(config.loss.temporal_relations_path) and os.path.exists(config.loss.spatial_relations_path), \
    "CRITICAL ERROR: Memory bank files not found. Please run the feature extraction script on Task 0 data first."
    logging.info(f"Memory bank found. Starting current task: {config.task.task_n}.\n\n")
    
    # Create dataset
    dataset = get_video_dataset(config)
    train_dataset, test_dataset, valid_dataset = dataset["train"], dataset["test"], dataset["valid"]

    # Get dataloaders
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
                                num_workers=TrainingConfigs.dataloader_num_workers,
                                )
    test_dataloader = DataLoader(test_dataset, 
                                batch_size=TrainingConfigs.evaluation_batch_size, 
                                shuffle=False, 
                                pin_memory=TrainingConfigs.dataloader_pin_memory, 
                                num_workers=TrainingConfigs.dataloader_num_workers,
                                )

    shutil.copyfile(config_path, f"{save_model_config_dir}/train_config.yml")

    # Model Setup
    model = CSTA(config_file=config_path)
    model.prepare_architecture_for_current_task()
    logging.info(f"Model config: model_name: {config.model.model_name}, dim: {config.model.dim}, num_heads: {config.model.num_heads}, num_layers: {config.model.num_layers}\n\n")

    optimizer = optim.AdamW(model.parameters(), lr=float(TrainingConfigs.learning_rate), weight_decay=float(TrainingConfigs.weight_decay))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=TrainingConfigs.num_training_epochs - TrainingConfigs.warmup_epochs, eta_min=float(TrainingConfigs.eta_min))

    accelerator = Accelerator()
    
    # Prepare Phase 1
    model, optimizer, train_dataloader, valid_dataloader, test_dataloader, scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, valid_dataloader, test_dataloader, scheduler
        )

    best_loss = float('inf')
    best_acc = 0.0
    early_stop_count = 0

    for epoch in range(TrainingConfigs.num_training_epochs):
        train_loss, _ = train_epoch(model, train_dataloader, optimizer, accelerator, epoch, grad_acc=getattr(TrainingConfigs, 'grad_acc', 1), max_grad=TrainingConfigs.max_grad)
        eval_loss, eval_acc = evaluate(model, valid_dataloader, accelerator, epoch)

        if eval_loss < best_loss:
            best_loss = eval_loss
            early_stop_count = 0
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), os.path.join(model_save_dir, "checkpoints", f'best_model_epoch_{epoch}.pth'))
        elif eval_acc > best_acc:
            best_acc = eval_acc
            early_stop_count = 0
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), os.path.join(model_save_dir, "checkpoints", f'best_model_epoch_{epoch}.pth'))
        else:
            early_stop_count += 1
            if early_stop_count > TrainingConfigs.early_stop_patience:
                logging.info(f"Stopping early after {early_stop_count} non-improvement.")
                break

        if epoch > TrainingConfigs.warmup_epochs:
            scheduler.step()
        if eval_acc*100 > 93:
            break

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    torch.save(unwrapped_model.state_dict(), os.path.join(model_save_dir, "checkpoints", f'final_model.pth'))

    eval_loss, eval_acc = evaluate(unwrapped_model, test_dataloader, accelerator, epoch)
    logging.info(f"Phase 1 Test Performance: Loss: {eval_loss:.4f}, Accuracy: {eval_acc:.4f}")

    if config.fine_tune.fine_tune:
        task_n = config.fine_tune.task_n_ft
        logging.info(f"\n\nStarting Fine-tuning for task {task_n}.\n")

        del valid_dataloader 
        torch.cuda.empty_cache()
        model = accelerator.unwrap_model(model)

        for param in model.parameters():
            param.requires_grad = False
        model.calculate_distil_loss = False
        model.calculate_lt_ls_loss = False

        for classifier in model.classifiers:
            for param in classifier.parameters():
                param.requires_grad = True
                
        ft_optimizer = optim.AdamW(model.parameters(), lr=float(config.fine_tune.ft_learning_rate), weight_decay=float(config.fine_tune.ft_weight_decay))
        dataset = get_video_dataset_for_ft(config)
        ft_train_dataset = dataset["train"]
        
        ft_dataloader = DataLoader(ft_train_dataset, 
                            batch_size=5,
                            shuffle=True, 
                            pin_memory=TrainingConfigs.dataloader_pin_memory, 
                            num_workers=TrainingConfigs.dataloader_num_workers
                            )

        model, ft_optimizer, ft_dataloader, test_dataloader = accelerator.prepare(model, ft_optimizer, ft_dataloader, test_dataloader)

        for epoch in range(config.fine_tune.ft_epochs):
            train_loss, _ = train_epoch(model, ft_dataloader, ft_optimizer, accelerator, epoch, grad_acc=1, max_grad=TrainingConfigs.max_grad)
            eval_loss, eval_acc = evaluate(model, test_dataloader, accelerator, epoch)

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), os.path.join(model_save_dir, "checkpoints", f'final_model_after_ft.pth'))

        logging.info(f"Final Test Performance (After FT): Loss: {eval_loss:.4f}, Accuracy: {eval_acc:.4f}")
        
    unwrapped_model.save_feature_banks(train_dataloader, accelerator, config.loss.memory_bandk_dir)

if __name__ == "__main__":
    main()