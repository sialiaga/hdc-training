# mm_project/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

class CrossAttentionFusion(nn.Module):
    """Mecanismo de Atención Cruzada."""
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super(CrossAttentionFusion, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, text_emb: torch.Tensor, tabular_emb: torch.Tensor) -> torch.Tensor:
        out, _ = self.attention(
            query=tabular_emb.unsqueeze(1),
            key=text_emb.unsqueeze(1),
            value=text_emb.unsqueeze(1)
        )
        fused_emb = self.norm(out.squeeze(1) + tabular_emb)
        return torch.cat((text_emb, fused_emb), dim=1)

class FusionModel(nn.Module):
    """Modelo de Fusión unificado."""
    def __init__(self, text_model: PreTrainedModel, num_classes: int, tabular_net: nn.Sequential,
                 tabular_out_features: int, fusion_head: nn.Sequential, fusion_head_out_features: int,
                 fusion_strategy: str = 'concat', model_type: str = 'cls'):
        super(FusionModel, self).__init__()
        
        self.text_model = text_model
        self.tabular_net = tabular_net
        self.fusion_strategy = fusion_strategy
        self.model_type = model_type
        text_hidden_size = self.text_model.config.hidden_size

        if fusion_strategy in ['add', 'multiply', 'cross_attention', 'weighted_add']:
            if tabular_out_features != text_hidden_size:
                raise ValueError(f"Para {fusion_strategy}, tabular_out_features ({tabular_out_features}) debe ser igual a text_hidden_size ({text_hidden_size}).")

        self.w_text = nn.Parameter(torch.tensor(0.5, dtype=torch.float))
        self.w_tab = nn.Parameter(torch.tensor(0.5, dtype=torch.float))
        self.cross_attn = CrossAttentionFusion(embed_dim=text_hidden_size)
        self.fusion_head = nn.Sequential(
            fusion_head,
            nn.Linear(fusion_head_out_features, num_classes)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, tabular_data: torch.Tensor) -> torch.Tensor:
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)

        if self.model_type == 'mean_pooling':
            text_embedding = text_output.last_hidden_state.mean(dim=1)
        elif self.model_type == 'cls':
            if hasattr(text_output, 'pooler_output') and text_output.pooler_output is not None:
                text_embedding = text_output.pooler_output
            else:
                text_embedding = text_output.last_hidden_state[:, 0, :]
        else:
             raise ValueError(f"model_type '{self.model_type}' no reconocido.")

        tabular_embedding = self.tabular_net(tabular_data)

        if self.fusion_strategy == 'add':
            combined_embedding = text_embedding + tabular_embedding
        elif self.fusion_strategy == 'multiply':
            combined_embedding = text_embedding * tabular_embedding
        elif self.fusion_strategy == 'weighted_add':
            w_text = F.relu(self.w_text)
            w_tab = F.relu(self.w_tab)
            combined_embedding = w_text * text_embedding + w_tab * tabular_embedding
        elif self.fusion_strategy == 'cross_attention':
            combined_embedding = self.cross_attn(text_embedding, tabular_embedding)
        elif self.fusion_strategy == 'hyper':
             fusion_add = text_embedding + tabular_embedding
             fusion_multiply = text_embedding * tabular_embedding
             fusion_concat = torch.cat((text_embedding, tabular_embedding), dim=1)
             combined_embedding = torch.cat((fusion_concat, fusion_add, fusion_multiply), dim=1)
        else:
            combined_embedding = torch.cat((text_embedding, tabular_embedding), dim=1)

        return self.fusion_head(combined_embedding)