from warnings import filterwarnings
filterwarnings("ignore")
from src import (
    CSTA, get_config, set_all_seeds, test, get_eval_dataset
)
from torch.utils.data import DataLoader
from accelerate import Accelerator
import torch, os, logging, datetime, argparse, shutil
import torch.optim as optim

def main():
    parser = argparse.ArgumentParser(description="Fine-tuning CSTA.")
    parser.add_argument('--config', type=str, default="config/eval_configs/UCF101/eval_task2.yml", help="Config File Path for Fine tuning.")
    parser.add_argument('--save_results', type=str, default="model_save/test", help="Folder to save everything.")
    args = parser.parse_args()
    config_path = args.config
    config = get_config(config_path)
    
    # fix randomness
    EvalConfig = config.eval
    set_all_seeds(EvalConfig.random_seed)
    
    # Create workdirectory
    work_dir = os.path.join(args.save_results, config.data.dataset_name, f"task_{config.task.task_n}")
    
    # setup directory. work_dir -) a. config, b. results
    save_eval_config_dir = os.path.join(work_dir, "config")
    log_dir = os.path.join(work_dir, "results")
    os.makedirs(save_eval_config_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    shutil.copyfile(config_path, f"{save_eval_config_dir}/eval_config.yml")
    
    # setup logger
    logging.basicConfig(
        filename=os.path.join(log_dir, f"results_{config.task.task_n}.log"),
        filemode="w",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    logging.info(f"\n\nEvaluation for model for task {config.task.task_n}: {datetime.datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}\n")
    logging.info(f"The config is being used from file: {config_path}")

    # load old model first
    model= CSTA(config_file=config.checkpoints.current_config)
    model.prepare_architecture_for_current_task()
    logging.info(f"\n\nDO NOT WORRY ABOUT THE CHECKPOINT LOADING. \nWe're now loading the actual checkpoint from {config.checkpoints.current_checkpoint}")
    model.load_weights(config.checkpoints.current_checkpoint)
    
    if config.task.task_n == 2:
        # load test datasets (hopefully i didn't mess up the labels)
        dataset=get_eval_dataset(config)
        task_0_test, task_1_test, task_2_test, id2label, label2id = dataset['task_0_test'], dataset['task_1_test'], dataset['task_2_test'], dataset['id2label'], dataset['label2id']
        
        task_0_test_datloader = DataLoader(task_0_test, 
                                    batch_size=EvalConfig.test_batch_size, 
                                    shuffle=False,
                                    pin_memory=EvalConfig.dataloader_pin_memory, 
                                    persistent_workers=EvalConfig.dataloader_persistent_workers,
                                    num_workers=EvalConfig.dataloader_num_workers,
                                    )
        task_1_test_datloader = DataLoader(task_1_test, 
                                    batch_size=EvalConfig.test_batch_size, 
                                    shuffle=False,
                                    pin_memory=EvalConfig.dataloader_pin_memory, 
                                    persistent_workers=EvalConfig.dataloader_persistent_workers,
                                    num_workers=EvalConfig.dataloader_num_workers,
                                    )
        task_2_test_datloader = DataLoader(task_2_test, 
                                    batch_size=EvalConfig.test_batch_size, 
                                    shuffle=False,
                                    pin_memory=EvalConfig.dataloader_pin_memory, 
                                    persistent_workers=EvalConfig.dataloader_persistent_workers,
                                    num_workers=EvalConfig.dataloader_num_workers,
                                    )

        accelerator = Accelerator()
        model, task_0_test_datloader, task_1_test_datloader, task_2_test_datloader = accelerator.prepare(
                model, task_0_test_datloader, task_1_test_datloader, task_2_test_datloader
            )
        
        _, task_0_acc = test(model, task_0_test_datloader, accelerator, task_n=0)
        _, task_1_acc = test(model, task_1_test_datloader, accelerator, task_n=1)
        _, task_2_acc = test(model, task_2_test_datloader, accelerator, task_n=2)
        
        logging.info(f"\n\n===========================================================================================================")
        logging.info(f"Model Used: {config.checkpoints.current_checkpoint}")
        logging.info(f"Final Accuracy: task_0: {task_0_acc:.4f}, task_1: {task_1_acc:.4f}, task_2: {task_2_acc:.4f}")
        logging.info(f"Average Accuracy: {((task_0_acc+task_1_acc+task_2_acc) / 3):.4f}")
        logging.info(f"\n\n===========================================================================================================")
    
    elif config.task.task_n == 1:
        # load test datasets (hopefully i didn't mess up the labels)
        dataset=get_eval_dataset(config)
        task_0_test, task_1_test, id2label, label2id = dataset['task_0_test'], dataset['task_1_test'], dataset['task_2_test'], dataset['id2label'], dataset['label2id']
        
        task_0_test_datloader = DataLoader(task_0_test, 
                                    batch_size=EvalConfig.test_batch_size, 
                                    shuffle=False,
                                    pin_memory=EvalConfig.dataloader_pin_memory, 
                                    persistent_workers=EvalConfig.dataloader_persistent_workers,
                                    num_workers=EvalConfig.dataloader_num_workers,
                                    )
        task_1_test_datloader = DataLoader(task_1_test, 
                                    batch_size=EvalConfig.test_batch_size, 
                                    shuffle=False,
                                    pin_memory=EvalConfig.dataloader_pin_memory, 
                                    persistent_workers=EvalConfig.dataloader_persistent_workers,
                                    num_workers=EvalConfig.dataloader_num_workers,
                                    )

        accelerator = Accelerator()
        model, task_0_test_datloader, task_1_test_datloader = accelerator.prepare(
                model, task_0_test_datloader, task_1_test_datloader
            )
        
        _, task_0_acc = test(model, task_0_test_datloader, accelerator, task_n=0)
        _, task_1_acc = test(model, task_1_test_datloader, accelerator, task_n=1)
        
        logging.info(f"\n\n=======================================================================================")
        logging.info(f"Model Used: {config.checkpoints.current_checkpoint}")
        logging.info(f"Final Accuracy: task_0: {task_0_acc:.4f}, task_1: {task_1_acc:.4f}")
        logging.info(f"Average Accuracy: {((task_0_acc+task_1_acc) / 2):.4f}")
        logging.info(f"\n\n=======================================================================================")
        
    elif config.task.task_n == 0:
        # load test datasets (hopefully i didn't mess up the labels)
        dataset=get_eval_dataset(config)
        task_0_test, id2label, label2id = dataset['task_0_test'], dataset['task_1_test'], dataset['task_2_test'], dataset['id2label'], dataset['label2id']
        
        task_0_test_datloader = DataLoader(task_0_test, 
                                    batch_size=EvalConfig.test_batch_size, 
                                    shuffle=False,
                                    pin_memory=EvalConfig.dataloader_pin_memory, 
                                    persistent_workers=EvalConfig.dataloader_persistent_workers,
                                    num_workers=EvalConfig.dataloader_num_workers,
                                    )

        accelerator = Accelerator()
        model, task_0_test_datloader = accelerator.prepare(
                model, task_0_test_datloader
            )
        
        _, task_0_acc = test(model, task_0_test_datloader, accelerator, task_n=0)
        
        logging.info(f"\n\n=======================================================================================")
        logging.info(f"Model Used: {config.checkpoints.current_checkpoint}")
        logging.info(f"Final Accuracy: task_0: {task_0_acc:.4f}")
        logging.info(f"\n\n=======================================================================================")
        
    else:
        raise NotImplementedError
    
if __name__ == "__main__":
    main()