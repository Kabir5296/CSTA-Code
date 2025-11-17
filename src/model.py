import torch, pdb, json, warnings
import torch.nn as nn
import torch.nn.functional as F
from .model_utils import (
    Adapter, TimesFormerBlock, save_current_model, load_model_weights, ConfigurationError
)
from .utils import get_config
from dataclasses import dataclass
from typing import Optional
from einops import rearrange

@dataclass
class CSTAOutput:
    logits: torch.FloatTensor = None
    predictions: torch.FloatTensor = None
    ce_loss: Optional[torch.FloatTensor] = None
    distil_loss: Optional[torch.FloatTensor] = None
    lt_loss: Optional[torch.FloatTensor] = None
    ls_loss: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    accuracy: Optional[torch.FloatTensor] = None
    spatial_relations: Optional[torch.FloatTensor] = None
    temporal_relations: Optional[torch.FloatTensor] = None

class CSTA(nn.Module):
    def __init__(self,
                 config_file="config/train_task0.yml",
                 **kwargs,
                 ):
        super().__init__()
        """
        CSTA model class.
        Args:
            num_frames: number of frames in video
            img_size: size of input image, must be square shape
            patch_size: size of patch to create from frames
            dim: dimension of the model
            num_classes: number of classes to initialize first classifier
            num_layers: number of TimesFormer blocks
            num_channels: number of channels in the input image, typically 3
            num_heads: number of heads in attention, typically 8
            init_with_adapters: whether to initialize the model with adapters. if true, default one adapter is added in the blocks.
            calculate_distil_loss: whether to calculate the distillation loss. if true, distil calculation is done on the model without one last adapter. time complexity increases.
            calculate_lt_ls_loss: whether to calculate the lt/ls losses. currently not implemented.
            miu_d: weight of the distillation loss (hyperparameter)
            miu_t: weight of the temporal loss (hyperparameter)
            miu_s: weight of the spatial loss (hyperparameter)

        Methods:
            add_one_adapter_per_block: adds one adapter to each block in the model
            add_one_new_classifier: adds one new classifier to the model
            add_new_task_components: adds one adapter and one classifier to the model, sets calculate_distil_loss to True.
            freeze_all_but_last: freezes all blocks, all adapters except the last, all classifiers except the last
            get_distil_loss: calculates the distillation loss
            forward: forward pass of the model
        """
        # load self.config
        self.config = get_config(config_file)
        
        # load model self.configs
        self.dim = self.config.model.dim
        self.num_layers = self.config.model.num_layers
        self.num_heads = self.config.model.num_heads
        
        # load data self.configs (also belongs to the model self.configs)
        self.img_size = self.config.data.img_size
        self.num_channels = self.config.data.num_channels
        self.patch_size = self.config.data.patch_size
        self.num_frames = self.config.data.num_frames

        # load loss self.configs
        self.calculate_distil_loss = self.config.loss.calculate_distil_loss
        self.calculate_lt_ls_loss = self.config.loss.calculate_lt_ls_loss
        self.miu_d = self.config.loss.miu_d
        self.miu_t = self.config.loss.miu_t
        self.miu_s = self.config.loss.miu_s
        self.lambda_1 = self.config.loss.lambda_1
        self.K = self.config.loss.K
        self.count_relations = self.config.loss.count_relations
        temporal_relations_path = self.config.loss.temporal_relations_path
        spatial_relations_path = self.config.loss.spatial_relations_path

        # load tasks self.configs
        self.task_n = self.config.task.task_n
        init_with_adapters = self.config.task.init_with_adapters
        self.num_classes_t0 = self.config.task.num_classes_t0
        
        # convert video frames to patches with conv2d, initialize cls_token, positional embeddings
        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.patch_embed = nn.Conv2d(self.num_channels, self.dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.norm = nn.LayerNorm(self.dim)
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, self.num_frames, 1, self.dim))
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, self.dim))

        # keeping a list of adapters. Each transformer block has a list of adapters
        self.temporal_adapters = nn.ModuleList([nn.ModuleList() for _ in range(self.num_layers)])
        self.spatial_adapters = nn.ModuleList([nn.ModuleList() for _ in range(self.num_layers)])

        # keeping a list of transformer blocks and inter task cross attention blocks
        self.blocks = nn.ModuleList([TimesFormerBlock(dim = self.dim, num_heads=self.num_heads) for _ in range(self.num_layers)])

        # keeping a list of classifiers
        self.classifiers = nn.ModuleList([nn.Linear(self.dim, self.num_classes_t0)])

        # add the first adapter to all the timesformer blocks
        if init_with_adapters:
            self.add_one_adapter_per_block()

        # keep the numbers stored somewhere
        self.model_attributes = self.get_model_attributes()
        
        if temporal_relations_path is not None and spatial_relations_path is not None:
            with open(temporal_relations_path, 'r') as f:
                self.temporal_relations = json.load(f)
            with open(spatial_relations_path, 'r') as f:
                self.spatial_relations = json.load(f)
        else:
            self.temporal_relations = None
            self.spatial_relations = None

    def get_model_attributes(self):
        total_blocks = len(self.blocks)
        total_temporal_adapters = total_spatial_adapters = 0

        for block_idx in range(len(self.blocks)):
            total_temporal_adapters += len(self.temporal_adapters[block_idx])
            total_spatial_adapters += len(self.spatial_adapters[block_idx])
        adapters_per_block = int(total_spatial_adapters/len(self.blocks))
        total_classifiers = len(self.classifiers)
        current_task = adapters_per_block - 1
        return {
            "total_blocks": total_blocks,
            "adapters_per_block": adapters_per_block,
            "current_task": current_task,
            "total_temporal_adapters": total_temporal_adapters,
            "total_spatial_adapters": total_spatial_adapters,
            "total_classifiers": total_classifiers,
            "total_adapters": total_temporal_adapters + total_spatial_adapters,
            "total_params": sum(p.numel() for p in self.parameters()),
        }

    def add_one_adapter_per_block(self):
        for block_idx in range(len(self.blocks)):
            self.temporal_adapters[block_idx].append(Adapter(self.dim))
            self.spatial_adapters[block_idx].append(Adapter(self.dim))
        self.model_attributes = self.get_model_attributes()

    def add_one_new_classifier(self, num_new_classes):
        new_classifier = nn.Linear(self.dim, num_new_classes)
        self.classifiers.append(new_classifier)
        self.model_attributes = self.get_model_attributes()

    def add_new_task_components(self, num_new_classes):
        self.add_one_adapter_per_block()
        self.add_one_new_classifier(num_new_classes)
        self.calculate_distil_loss = True
        self.calculate_lt_ls_loss = True
        self.model_attributes = self.get_model_attributes()

    def freeze_all_but_last(self):
        for classifier in self.classifiers[:-1]:
            for param in classifier.parameters():
                param.requires_grad = False

        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False

        for block_idx in range(len(self.blocks)):
            for adapter in self.temporal_adapters[block_idx][:-1]:
                for param in adapter.parameters():
                    param.requires_grad = False

            for adapter in self.spatial_adapters[block_idx][:-1]:
                for param in adapter.parameters():
                    param.requires_grad = False

    def run_classifier_head(self, last_hidden_layer, B, T):
        x = last_hidden_layer
        x = x[:, :1, :].squeeze(1).reshape([B,T,self.dim]).mean(dim=1)
        outputs = []
        for classifier in self.classifiers:
            outputs.append(classifier(x))
        return torch.cat(outputs, dim=-1)
    
    def get_relations(self, s_feat, t_feat, f_feat, B, T):
        """R = (Clf(S)/Clf(F), Cos(Clf(S), Clf(T)))"""
        
        # Run classifier on separate components
        # We use the current (newest) classifier for the relation extraction

        # Helper to pool and classify
        def pool_and_classify(feat):
            if feat is None: return None
            # Pool: [B*T, P, D] -> [B, T, D] -> [B, D]
            pooled = feat.view(B, T, -1, self.dim)[:, :, 0, :].mean(dim=1) 
            # Using the LAST classifier (current task) for relation extraction
            return self.classifiers[-1](pooled)

        # Get Logits for components
        logits_s = pool_and_classify(s_feat)
        logits_t = pool_and_classify(t_feat)
        logits_f = pool_and_classify(f_feat)

        # Avoid division by zero
        eps = 1e-6
        cos_st = F.cosine_similarity(logits_s, logits_t, dim=1).unsqueeze(1)
        
        # Rs = Concat( Ls/Lf, Cos(Ls, Lt) )
        ratio_s = logits_s / (logits_f + eps)
        ratio_t = logits_t / (logits_f + eps)

        return torch.cat([ratio_s, cos_st], dim=1), torch.cat([ratio_t, cos_st], dim=1)

    def calculate_causal_loss(self, R_curr, R_old, stored_relations):
        """Calculates Lt or Ls based on Eq (7)"""
        if stored_relations is None or len(stored_relations) == 0:
            return torch.tensor(0.0, device=R_curr.device)

        # Shape: [M, Relation_Dim]
        memory_bank = torch.tensor(stored_relations, device=R_curr.device, dtype=torch.float32)
        
        # Normalize for Cosine Search
        R_old_norm = F.normalize(R_old, p=2, dim=1)
        mem_norm = F.normalize(memory_bank, p=2, dim=1)

        # KNN Search (Algorithm 1, Line 5-6)
        sim_matrix = torch.mm(R_old_norm, mem_norm.t())
        
        # Get Top-K similar relations
        top_k_val, top_k_idx = torch.topk(sim_matrix, k=min(self.K, len(stored_relations)), dim=1)
        
        # Calculate Hybrid Relation R^K (Target)
        weights = F.softmax(top_k_val, dim=1).unsqueeze(-1) # [B, K, 1]
        retrieved_rels = memory_bank[top_k_idx]             # [B, K, Rel_Dim]
        
        # R_target [B, Rel_Dim]
        R_target = torch.sum(retrieved_rels * weights, dim=1)

        # Calculate Loss Eq (7): 1 - Cos(R^K, R_n)
        return 1 - F.cosine_similarity(R_target, R_curr, dim=1).mean()

    def vanilla_forward(self, x, B, T):
        if self.task_n == 0 and self.calculate_distil_loss:
            raise ConfigurationError("You can not run distillation loss during the first task. Initialize with calculate_distil_loss=False")

        distil_loss = None
        if self.calculate_distil_loss:
            x_old = x

        t_feat_curr = None
        s_feat_curr = None
        t_feat_old = None
        s_feat_old = None

        for block_idx, block in enumerate(self.blocks):
            is_last_layer = (block_idx == len(self.blocks) - 1)

            # Temporal Branch
            block_t_msa = block.temporal_msa(x, B, T, self.num_patches)
            temporal_adapter_features = [block_t_msa]

            if self.task_n != 0:
                for temporal_adapters in self.temporal_adapters[block_idx]:
                    adapter_output = temporal_adapters(block_t_msa)
                    temporal_adapter_features.append(adapter_output)
                
                q = temporal_adapter_features[-1]
                if self.task_n == 1:
                    kv = torch.tensor(temporal_adapter_features[-2])
                else:
                    reshaped_feats = [f.view(B, T, -1, self.dim) for f in temporal_adapter_features[:-1]]
                    kv = torch.cat(reshaped_feats, dim=1)
                    kv = kv.view(-1, kv.shape[2], self.dim)
                x = block.norm_t(x + block_t_msa + block.temporal_cross_attention(q, kv, kv, B, T, self.num_patches))
            else:
                x = block.norm_t(x + block_t_msa)
            
            # Capture Temporal Feature (Tn) at last layer
            if is_last_layer:
                t_feat_curr = x.clone()

            # Spatial Branch
            block_s_msa = block.spatial_msa(x)
            spatial_adapter_features = [block_s_msa]

            if self.task_n != 0:
                for spatial_adapters in self.spatial_adapters[block_idx]:
                    adapter_output = spatial_adapters(block_s_msa)
                    spatial_adapter_features.append(adapter_output)

                q = spatial_adapter_features[-1]
                if self.task_n == 1:
                    kv = spatial_adapter_features[-2]
                else:
                    kv = torch.cat(spatial_adapter_features[:-1], dim=1)
                x = block.norm_s(x + block_s_msa + block.spatial_cross_attention(q, kv, kv))
            else:
                x = block.norm_s(x + block_s_msa)

            # Capture Spatial Feature (Sn) at last layer
            if is_last_layer:
                s_feat_curr = x.clone()

            # Final MLP
            x = block.norm_mlp(x + block.mlp(x))
            
            if self.calculate_distil_loss:
                with torch.no_grad():
                    block_t_msa_old = block.temporal_msa(x_old, B, T, self.num_patches)
                    temporal_adapter_features_old = [block_t_msa_old]

                    if (self.task_n - 1) != 0:
                        for temporal_adapters in self.temporal_adapters[block_idx][:-1]:
                            adapter_output = temporal_adapters(block_t_msa_old)
                            temporal_adapter_features_old.append(adapter_output)

                        q = temporal_adapter_features_old[-1]
                        if self.task_n == 1:
                            kv = torch.tensor(temporal_adapter_features_old[-2])
                        else:
                            reshaped_feats = [f.view(B, T, -1, self.dim) for f in temporal_adapter_features_old[:-1]]
                            kv = torch.cat(reshaped_feats, dim=1)
                            kv = kv.view(-1, kv.shape[2], self.dim)
                        x_old = block.norm_t(x_old + block_t_msa_old + self.temporal_cross_attention_old[block_idx](q, kv, kv, B, T, self.num_patches))
                    else:
                        x_old = block.norm_t(x_old + block_t_msa_old)
                    
                    if is_last_layer:
                        t_feat_old = x_old.clone()

                    # same for spatial
                    block_s_msa_old = block.spatial_msa(x_old)
                    spatial_adapter_features_old = [block_s_msa_old]

                    if (self.task_n - 1) != 0:
                        for spatial_adapters in self.spatial_adapters[block_idx][:-1]:
                            adapter_output = spatial_adapters(block_s_msa_old)
                            spatial_adapter_features_old.append(adapter_output)

                        q = spatial_adapter_features_old[-1]
                        if self.task_n == 1:
                            kv = spatial_adapter_features_old[-2]
                        else:
                            kv = torch.cat(spatial_adapter_features_old[:-1], dim=1)
                        x_old = block.norm_s(x_old + block_s_msa_old + self.spatial_cross_attention_old[block_idx](q, kv, kv))
                    else:
                        x_old = block.norm_s(x_old + block_s_msa_old)

                    if is_last_layer:
                        s_feat_old = x_old.clone()

                    # final mlp
                    x_old = block.norm_mlp(x_old + block.mlp(x_old))
        
        # Run classifiers
        logits = self.run_classifier_head(x, B, T)
        
        if self.calculate_distil_loss:
            with torch.no_grad():
                logits_old = self.run_classifier_head(x_old, B, T)
            distil_loss = F.kl_div(F.log_softmax(logits, dim=1), F.softmax(logits_old, dim=1), reduction="batchmean")

        return logits, distil_loss, x, t_feat_curr, s_feat_curr, t_feat_old, s_feat_old

    def process_x_for_forward(self, x):
        B, T, C, H, W = x.shape
        if H != W:
            raise ValueError('Input tensor must have equal height and width')
        elif H != self.img_size:
            raise ValueError('Input tensor has incorrect height and width')

        x = x.reshape(B * T, C, H, W)                       # reshape to (B * T, C, H, W) for patch embedding
        x = self.patch_embed(x)                             # shape: B*T, dim, H//patch_size, W//patch_size
        x = x.flatten(2).transpose(1, 2)                    # shape: B*T, num_patches, dim : num_patches = (H/patch_size)*(W/patch_size)

        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)           # shape: B*T, 1, dim
        x = torch.cat((cls_tokens, x), dim=1)                           # shape: B*T, num_patches+1, dim
        
        x = x.reshape([B,T,self.num_patches+1,self.dim])                # reshape for adding positional embeddings
        x = x + self.spatial_pos_embed + self.temporal_pos_embed
        x = x.reshape([B*T,self.num_patches+1,self.dim])                # reshape back to: B*T, num_patches+1, dim
        return x

    def forward(self, x, targets=None):
        B, T, C, H, W = x.shape
        x = self.process_x_for_forward(x)
        loss = ce_loss = distil_loss = lt_loss = ls_loss = accuracy = None

        if self.calculate_distil_loss:
            import copy
            self.temporal_cross_attention_old = nn.ModuleList()
            self.spatial_cross_attention_old = nn.ModuleList()
            
            for block in self.blocks:
                frozen_copy = copy.deepcopy(block.temporal_cross_attention)
                for param in frozen_copy.parameters():
                    param.requires_grad = False
                frozen_copy.eval() 
                self.temporal_cross_attention_old.append(frozen_copy)
                del frozen_copy

                frozen_copy = copy.deepcopy(block.spatial_cross_attention)
                for param in frozen_copy.parameters():
                    param.requires_grad = False
                frozen_copy.eval() 
                self.spatial_cross_attention_old.append(frozen_copy)
                del frozen_copy

        logits, distil_loss, full_feat, t_feat_curr, s_feat_curr, t_feat_old, s_feat_old = self.vanilla_forward(x, B, T)
        predictions = torch.argmax(logits, dim=-1)

        R_s_curr, R_t_curr = self.get_relations(s_feat_curr, t_feat_curr, full_feat, B, T)
                
        total_loss = []
        if targets is not None:
            accuracy = (predictions == targets).float().mean()
            ce_loss = F.cross_entropy(logits, targets)
            total_loss.append(ce_loss)
            
            if self.calculate_distil_loss:
                total_loss.append(self.miu_d * distil_loss) if distil_loss is not None else None
            
            if self.calculate_lt_ls_loss and self.calculate_distil_loss:
                with torch.no_grad():
                    R_s_old, R_t_old = self.get_relations(s_feat_old, t_feat_old, full_feat, B, T)
                lt_loss = self.calculate_causal_loss(R_t_curr, R_t_old, self.temporal_relations)
                ls_loss = self.calculate_causal_loss(R_s_curr, R_s_old, self.spatial_relations)
                total_loss.append(self.miu_t * lt_loss) if lt_loss is not None else None
                total_loss.append(self.miu_s * ls_loss) if ls_loss is not None else None

            loss = sum(total_loss)

        return CSTAOutput(
            logits = logits,
            loss = loss,
            ce_loss = ce_loss,
            distil_loss = distil_loss,
            lt_loss = lt_loss,
            ls_loss = ls_loss,
            predictions = predictions,
            last_hidden_state = x,
            accuracy = accuracy,
            spatial_relations=R_s_curr,
            temporal_relations=R_t_curr
        )
