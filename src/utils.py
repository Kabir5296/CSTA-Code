import yaml
from collections import namedtuple

def dict_to_object(dictionary):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = dict_to_object(value)
    return namedtuple('CSTAConfig', dictionary.keys())(**dictionary)

def get_config(file_name):
    with open(file_name, "r") as f:
        config=yaml.safe_load(f)
    return dict_to_object(config)

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import torch, os, random, logging
import numpy as np
from tqdm import tqdm
from torchvision.io import read_video

def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_video_dataset(config):
    training_csv_path = config.data.train_csv
    valid_csv_path = config.data.valid_csv
    test_csv_path = config.data.test_csv
    train = pd.read_csv(training_csv_path)

    all_labels = sorted(train['label'].unique().tolist())
    id2label = {}
    label2id = {}
    if config.task.task_n == 0:
        for index, label in enumerate(all_labels):
            id2label[index] = label
            label2id[label] = index
    elif config.task.task_n == 1:
        for index, label in enumerate(all_labels):
            id2label[index + config.task.num_classes_t0] = label
            label2id[label] = index + config.task.num_classes_t0
    elif config.task.task_n == 2:
        for index, label in enumerate(all_labels):
            id2label[index + config.task.num_classes_t0 + 10] = label
            label2id[label] = index + config.task.num_classes_t0 + 10
    
    return {
        "train" : VideoDataset(config=config, csv_path=training_csv_path, label2id=label2id, split="train"),
        "test" : VideoDataset(config=config, csv_path=test_csv_path, label2id=label2id, split="test"),
        "valid" : VideoDataset(config=config, csv_path=valid_csv_path, label2id=label2id, split="valid"),
        "id2label" : id2label,
        "label2id" : label2id,
    }

class VideoDataset(Dataset):
    def __init__(self, config, csv_path, label2id ,split="train"):
        self.dataset_name = config.data.dataset_name
        self.split=split
        
        self.num_frames = config.data.num_frames
        self.img_size = config.data.img_size
        self.num_channels = config.data.num_channels
        self.patch_size = config.data.patch_size
        self.root_path = config.data.root_dir
        self.label2id = label2id
        
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
            # transforms.RandomGrayscale(0.1),
            # transforms.RandomRotation((-15,15)),
            # transforms.RandomErasing(0.1),
            ])
        self.df = pd.read_csv(csv_path)
        # logging.info(f"Dataset loaded for {self.dataset_name}/{split}.")

    def __len__(self):
        return len(self.df)
    
    @staticmethod
    def sample_frames(video, num_frames):
        total_frames = video.shape[0]
        indices = torch.linspace(0, total_frames - 1, num_frames)
        indices = torch.clamp(indices, 0, total_frames - 1).long()
        frames = video[indices]
        frames = frames.float() / 255.0
        return frames

    @staticmethod
    def load_video(path):
        video, _, _ = read_video(filename=path, output_format="TCHW", pts_unit='sec')
        return video.float()
    
    def __getitem__(self, index):
        video_path = os.path.join(self.root_path, self.df['clip_path'][index][1:])
        label = self.df['label'][index]

        frames = self.sample_frames(self.load_video(video_path), num_frames=self.num_frames)
        
        if frames.shape[-1] == 3:   # if the channel dimension is at last
            frames = frames.permute(0, 3, 1, 2)
        
        processed_frames = torch.stack([
            self.transform(frame) for frame in frames
        ])

        return {
            "input_frames": processed_frames,
            "label": self.label2id[label],
        }
        
def train_epoch(model, train_dataloader, optimizer, accelerator, epoch, max_grad = 3):
    model.train()

    running_acc = 0.0
    total_samples = num_steps_with_grad = 0
    running_grad_norm = current_grad_norm = 0.0 
    running_loss = running_ce_loss = running_distil_loss = running_lt_loss = running_ls_loss = 0.0
    
    progress_bar = tqdm(
        total=len(train_dataloader),
        disable=not accelerator.is_local_main_process,
        desc=f"Training epoch {epoch}"
    )
    for batch_idx, batch in enumerate(train_dataloader):
        input_frames = batch["input_frames"]
        labels = batch["label"]
        batch_size = labels.size(0)
        
        with accelerator.accumulate(model):
            outputs = model(input_frames, labels)
            loss = outputs.loss
            
            ce_loss = outputs.ce_loss
            distil_loss = outputs.distil_loss
            lt_loss = outputs.lt_loss
            ls_loss = outputs.ls_loss
            
            predictions = outputs.predictions
            correct = (predictions == labels).sum().item()
            accuracy = correct / batch_size

            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), float(max_grad))
                if grad_norm is not None:
                    current_grad_norm = grad_norm.item()
                    running_grad_norm += current_grad_norm
                    num_steps_with_grad += 1

            optimizer.step()
            optimizer.zero_grad()
        
        running_loss += loss.item() * batch_size
        
        running_ce_loss += ce_loss.item() * batch_size
        running_distil_loss += distil_loss.item() * batch_size if distil_loss is not None else 0.0
        running_lt_loss += lt_loss.item() * batch_size if lt_loss is not None else 0.0
        running_ls_loss += ls_loss.item() * batch_size if ls_loss is not None else 0.0
        
        running_acc += correct
        total_samples += batch_size
        
        progress_bar.update(1)
        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "batch_acc": f"{accuracy:.4f}",
            "running_acc": f"{running_acc/total_samples:.4f}",
            "grad": f"{current_grad_norm:.4f}",
        })
    
    avg_loss = running_loss / total_samples
    avg_acc = running_acc / total_samples
    avg_grad = running_grad_norm / num_steps_with_grad if num_steps_with_grad > 0 else 0.0
    
    avg_ce_loss = running_ce_loss / total_samples
    avg_distil_loss = running_distil_loss / total_samples
    avg_lt_loss = running_lt_loss / total_samples
    avg_ls_loss = running_ls_loss / total_samples
    
    progress_bar.close()
    logging.info(
        f"| Training Epoch {epoch}:: | "
        f"| Average Loss: {avg_loss:.4f}, | "
        f"| Average Accuracy: {avg_acc:.4f}, | "
        f"| Avg Grad Norm: {avg_grad:.4f}, | "
        f"| Samples: {total_samples} | "
        f"| CE Loss: {avg_ce_loss:.4f} | "
        f"| Distill Loss: {avg_distil_loss:.4f} | "
        f"| Lt, Ls Loss: {avg_lt_loss:.4f}, {avg_ls_loss:.4f} | "
        f"| Max Grad Limit: {max_grad} | " 
    )
    
    return avg_loss, avg_acc

def evaluate(model, eval_dataloader, accelerator, epoch):
    model.eval()
    
    running_acc = 0.0
    total_samples = 0
    running_loss = running_ce_loss = running_distil_loss = running_lt_loss = running_ls_loss = 0.0
    
    progress_bar = tqdm(
        total=len(eval_dataloader),
        disable=not accelerator.is_local_main_process,
        desc=f"Evaluation epoch {epoch}"
    )
    
    with torch.no_grad():
        for batch in eval_dataloader:
            input_frames = batch["input_frames"]
            labels = batch["label"]
            batch_size = labels.size(0)

            outputs = model(input_frames, labels)
            loss = outputs.loss
            
            ce_loss = outputs.ce_loss
            distil_loss = outputs.distil_loss
            lt_loss = outputs.lt_loss
            ls_loss = outputs.ls_loss
            
            predictions = outputs.predictions
            correct = (predictions == labels).sum().item()
            accuracy = correct / batch_size
            
            running_loss += loss.item() * batch_size
            
            running_ce_loss += ce_loss.item() * batch_size
            running_distil_loss += distil_loss.item() * batch_size if distil_loss is not None else 0.0
            running_lt_loss += lt_loss.item() * batch_size if lt_loss is not None else 0.0
            running_ls_loss += ls_loss.item() * batch_size if ls_loss is not None else 0.0
            
            running_acc += correct
            total_samples += batch_size
            
            progress_bar.update(1)
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "batch_acc": f"{accuracy:.4f}",
                "running_acc": f"{running_acc/total_samples:.4f}"
            })
    
    avg_loss = running_loss / total_samples
    avg_acc = running_acc / total_samples
    avg_ce_loss = running_ce_loss / total_samples
    avg_distil_loss = running_distil_loss / total_samples
    avg_lt_loss = running_lt_loss / total_samples
    avg_ls_loss = running_ls_loss / total_samples
    progress_bar.close()
    logging.info(
        f"| Evaluation Epoch {epoch}:: | "
        f"| Average Loss: {avg_loss:.4f}, | "
        f"| Average Accuracy: {avg_acc:.4f}, | "
        f"| Samples: {total_samples} | "
        f"| CE Loss: {avg_ce_loss:.4f} | "
        f"| Distill Loss: {avg_distil_loss:.4f} | "
        f"| Lt, Ls Loss: {avg_lt_loss:.4f}, {avg_ls_loss:.4f} | "
    )
    
    return avg_loss, avg_acc

def get_model_info(model):
    # 1. Model Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 2. Model Size (Theoretical)
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024**2
    
    print(f"Total Params:     {total_params / 1e6:.2f}M")
    print(f"Trainable Params: {trainable_params / 1e6:.2f}M")
    print(f"Model Size (MB):  {size_mb:.2f} MB")