import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F



class IntermediateSequential(nn.Sequential):
    def __init__(self, *args, return_intermediate=False):
        super().__init__(*args)
        self.return_intermediate = return_intermediate

    def forward(self, input):
        if not self.return_intermediate:
            return super().forward(input)

        intermediate_outputs = {}
        output = input
        for name, module in self.named_children():
            output = intermediate_outputs[name] = module(output)

        return output, intermediate_outputs


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.gelu(input)

class SelfAttention(nn.Module):
    def __init__(
        self, dim, heads=4, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads  #自注意力头数
        head_dim = dim // heads   #每个注意力头部分配的特征维度
        self.scale = qk_scale or head_dim ** -0.5  #缩放因子

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        dim,##输入特征维度
        depth,##Transformer层数
        heads,##自注意力头数
        mlp_dim,##前馈网络的隐藏层维度
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
    ):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    Residual(
                        PreNormDrop(
                            dim,
                            dropout_rate,
                            SelfAttention(dim, heads=heads, dropout_rate=attn_dropout_rate),
                        )
                    ),
                    Residual(
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate))
                    ),
                ]
            )
            # dim = dim / 2
        self.net = IntermediateSequential(*layers)


    def forward(self, x):
        return self.net(x)

if __name__ == '__main__':
    input_dim = 32 # 输入特征维度（例如，对于文本任务可能是嵌入维度）

    seq_length = 256**2  # 序列长度

    depth = 3  # Transformer层数

    heads = 3  # 自注意力头数

    mlp_dim = 256  # 前馈网络的隐藏层维度

    dropout_rate = 0.1

    attn_dropout_rate = 0.1
    x = torch.randn(1, seq_length, input_dim)
    layer =TransformerModel(
        dim=input_dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        dropout_rate=dropout_rate,
        attn_dropout_rate=attn_dropout_rate,

    )
    # output_tensor, intermediate_outputs= layer(x)
    output_tensor = layer(x)
    print("Output tensor shape:", output_tensor.shape)
    # for layer_name, layer_output in intermediate_outputs.items():
    #     print(f"Intermediate output from layer '{layer_name}' shape:", layer_output.shape)