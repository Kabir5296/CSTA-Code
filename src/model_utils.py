import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class Adapter(nn.Module):
    def __init__(self, dim, reduction_factor=4):
        super().__init__()
        """
        Adapter module from the CSTA paper.
        SkipConnection( FullyConnectedUpSample -> GELU -> FullyConnectedDownSample )
        """
        self.fc_down = nn.Linear(dim, dim // reduction_factor)
        self.fc_up = nn.Linear(dim // reduction_factor, dim)
        self.gelu = nn.GELU()
        for p in self.fc_down.parameters():
            if p.dim() > 1: nn.init.kaiming_uniform_(p, mode='fan_in', nonlinearity='relu')
        for p in self.fc_up.parameters():
            if p.dim() > 1: nn.init.kaiming_uniform_(p, mode='fan_in', nonlinearity='relu')
    def forward(self, x):
        return self.fc_up(self.gelu(self.fc_down(x)))

# class TemporalMultiheadAttention(nn.Module):
#     def __init__(self, dim, num_heads=8):
#         super().__init__()
#         """
#         Temporal multihead attention. Applies attention on the temporal axis, reshapes and converts back the data as necessary.
#         """
#         self.dim = dim
#         # self.layer_norm = nn.LayerNorm(dim)
#         self.msa = nn.MultiheadAttention(dim, num_heads, batch_first=True)
#         # self.proj = nn.Linear(dim,dim)
    
    # def temporal_preprocess(self, x, B, T, num_patches):
    #     # take input x in B*T, num_patches+1, dim format. convert to B*num_patches, T, dim and returns
    #     x = x.reshape(B, T, num_patches+1, self.dim)        # shape: B, T, num_patches+1, dim
    #     cls_token, patches = x[:, :, :1, :], x[:, :, 1:, :]
    #     patches = patches.permute(0, 2, 1, 3)
    #     patches = patches.reshape(-1, T, self.dim)          #shape: B*num_patches, T, sim
    #     return patches, cls_token
    
    # def forward(self, x, B, T, num_patches):
    #     """Forward for temporal multihead attention, residual connection and normalization are not used because it will be handled in the CSTA model itself"""
    #     # x is the patches added with cls token, shape: B*T, num_patches+1, dim
    #     x, cls_token = self.temporal_preprocess(x,B,T,num_patches)
    #     # res = x
    #     # x = self.layer_norm(x)      # shape: B*num_patches, T, dim
    #     x, _= self.msa(x,x,x)       # shape: B*num_patches, T, dim
    #     # x = self.proj(x)            # shape: B*num_patches, T, dim
    #     # x = x + res
    #     x = x.reshape([B, num_patches, T, self.dim])    # shape: B, num_patches, T, dim
    #     x = x.permute(0,2,1,3)                          # shape: B, T, num_patches, dim
    #     x = torch.cat((cls_token, x), dim=2)            # shape: B, T, num_patches+1, dim 
    #     return x.reshape(B*T, num_patches +1, self.dim) # shape: B * T, num_patches+1, dim
    
class TemporalMultiheadAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.msa = nn.MultiheadAttention(dim, num_heads, batch_first=True)
    
    def forward(self, x, B, T, num_patches):
        x = x.reshape(B, T, num_patches + 1, self.dim) # [B, T, N+1, D]
        patches = x[:, :, 1:, :] 
        patches = patches.permute(0, 2, 1, 3).reshape(-1, T, self.dim)
        patches, _ = self.msa(patches, patches, patches)
        patches = patches.reshape(B, num_patches, T, self.dim).permute(0, 2, 1, 3)
        cls_residual = torch.zeros((B, T, 1, self.dim), device=x.device, dtype=x.dtype)
        
        output = torch.cat([cls_residual, patches], dim=2)
        return output.reshape(B * T, num_patches + 1, self.dim)
    
class SpatialMultiheadAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        """
        Spatial multihead attention. Same usual attention because spatial is the usual one.
        """
        self.msa = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x):
        """Forward for spatial multihead attention, residual connection and normalization are not used because it will be handled in the CSTA model itself"""
        x, _= self.msa(x,x,x)
        return x
    
class SpatialMultiheadCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.msa = nn.MultiheadAttention(dim, num_heads, batch_first=True)
    
    def forward(self, q, k, v):
        x, _ = self.msa(q, k, v)
        return x
    
class TemporalMultiheadCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.msa = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, query, key, value, B, T, num_patches):
        """
        Applies Cross Attention on the Temporal Dimension.
        Query: [B*T, N+1, D] (Usually current task adapter output)
        Key/Val: [B*T, N+1, D] (Usually prev task adapter output)
        """        
        q = query.reshape(B, T, num_patches+1, self.dim)[:, :, 1:, :] # [B, T, N, D]
        k = key.reshape(B, T, num_patches+1, self.dim)[:, :, 1:, :]
        v = value.reshape(B, T, num_patches+1, self.dim)[:, :, 1:, :]
        
        # 2. Reshape for Temporal: [B*N, T, D]
        q = q.permute(0, 2, 1, 3).reshape(-1, T, self.dim)
        k = k.permute(0, 2, 1, 3).reshape(-1, T, self.dim)
        v = v.permute(0, 2, 1, 3).reshape(-1, T, self.dim)
        
        # 3. Cross Attention
        attn_output, _ = self.msa(query=q, key=k, value=v)
        
        # 4. Reshape Back
        attn_output = attn_output.reshape(B, num_patches, T, self.dim).permute(0, 2, 1, 3)
        
        # 5. Add Zero CLS Residual
        cls_residual = torch.zeros((B, T, 1, self.dim), device=query.device, dtype=query.dtype)
        output = torch.cat([cls_residual, attn_output], dim=2)
        
        return output.reshape(B * T, num_patches + 1, self.dim)

class TimesFormerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, factor=4):
        super().__init__()
        """
        x_temporal = T_MSA(x)
        x = NORM(x_temporal + x)

        x_spatial = S_MSA(x)
        x = NORM(x + x_spatial)

        x = NORM(MLP(x) + x)
        """

        self.temporal_msa = TemporalMultiheadAttention(dim, num_heads)
        self.spatial_msa = SpatialMultiheadAttention(dim, num_heads)
        self.temporal_cross_attention = TemporalMultiheadCrossAttention(dim, num_heads)
        self.spatial_cross_attention = SpatialMultiheadCrossAttention(dim, num_heads)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * factor),
            nn.GELU(),
            nn.Linear(dim * factor, dim)
        )
        
        self.norm_t = nn.LayerNorm(dim)
        self.norm_s = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)

    def forward(self, x):
        """This forward method is not used in the actual implementation, hence not implemented"""
        raise NotImplementedError("This forward method is not meant to be used. Please define forward in the CSTA forward method")
    
def save_current_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Current model successfully saved to: {path}")
    
def load_model_weights(weights_path, device="cpu"):
    return torch.load(weights_path, weights_only=True, map_location=device)

class ConfigurationError(Exception):
    pass