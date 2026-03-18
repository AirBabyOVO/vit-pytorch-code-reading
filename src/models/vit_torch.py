"""
Vision Transformer PyTorch 内置版（vit_torch.py）
功能：使用 PyTorch 内置的 Transformer 组件实现 ViT，简化代码
对应原项目 model.py 中的 VisionTransformer_pytorch 类
"""
# 先导入必要的包（必须加，否则会报错）
import torch
import torch.nn as nn
import torch.nn.functional as F



class VisionTransformer_pytorch(nn.Module):
    """
    Vision Transformer Class with Pytorch transformer layers (TransformerEncoder and TransformerEncoderLayer) instead of scratch implementation.
    These layer replace the encoder layers (including self-attention operation). Hence, SelfAttention and Encoder classes can be removed.
    Embed layer cannot be replaced as Image to PatchEmbedding Block not available in PyTorch yet.
    Classifier is a simple MLP (and not replaced/not available in PyTorch).

    Parameters:
        n_channels (int)        : Number of channels of the input image
        embed_dim  (int)        : Embedding dimension
        n_layers   (int)        : Number of encoder blocks to use
        n_attention_heads (int) : Number of attention heads to use for performing MultiHeadAttention
        forward_mul (float)     : Used to calculate dimension of the hidden fc layer = embed_dim * forward_mul
        image_size (int)        : Image size
        patch_size (int)        : Patch size
        n_classes (int)         : Number of classes
        dropout  (float)        : dropout value

    Input:
        x (tensor): Image Tensor of shape B, C, IW, IH

    Returns:
        Tensor: Logits of shape B, CL
    """

    def __init__(self, n_channels, embed_dim, n_layers, n_attention_heads, forward_mul, image_size, patch_size,
                 n_classes, dropout=0.1):
        super().__init__()
        self.embedding = EmbedLayer(n_channels, embed_dim, image_size, patch_size, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=n_attention_heads,
                                                   dim_feedforward=forward_mul * embed_dim,
                                                   dropout=dropout,
                                                   activation=nn.GELU(),
                                                   batch_first=True,
                                                   norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers, norm=nn.LayerNorm(embed_dim))
        self.classifier = Classifier(embed_dim, n_classes)

        self.apply(vit_init_weights)  # Weight initalization

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.classifier(x)
        return x