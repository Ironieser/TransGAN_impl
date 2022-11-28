import numpy as np
import torch
import torch.nn as nn

from models_search.vit.vit_pytorch import ViT
from einops import rearrange

class matmul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        x = x1 @ x2
        return x


def count_matmul(m, x, y):
    num_mul = x[0].numel() * x[1].size(-1)
    # m.total_ops += torch.DoubleTensor([int(num_mul)])
    m.total_ops += torch.DoubleTensor([int(0)])



def build_transformer_encoder(image_size=(6, 10),patch_size=(6, 10), dim=1024,
                              depth=2,num_heads=8,mlp_ratio=4.0,pool='all', channels=128):
    return ViT(
        image_size=image_size,
        patch_size=patch_size,
        dim=dim,
        depth=depth,
        heads=num_heads,
        mlp_dim=dim*mlp_ratio,
        pool=pool,
        channels=channels,
    )


def pixel_upsample(x, H, W):
    B, N, C = x.size()
    assert N == H * W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.PixelShuffle(2)(x)
    B, C, H, W = x.size()
    x = x.view(-1, C, H * W)
    x = x.permute(0, 2, 1).contiguous()
    return x, H, W


class Generator(nn.Module):
    def __init__(self, args=None, img_size=224, patch_size=16, in_chans=3, num_classes=10, embed_dim=384, depth=5,
                 num_heads=16, mlp_ratio=4., qkv_bias=True, qk_scale=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, debug=False, use_rpe=False):
        super(Generator, self).__init__()
        self.debug = debug
        self.use_rpe = use_rpe
        self.args = None if debug else args
        self.latent_norm = False if debug else args.latent_norm
        self.ch = embed_dim
        self.bottom_width = 8 if debug else args.bottom_width
        if not debug:
            self.embed_dim = embed_dim = args.gf_dim
        else:
            self.embed_dim = embed_dim = 1024
        self.l1_embed_dim = self.embed_dim //4
        norm_layer = 'ln' if debug else args.g_norm
        mlp_ratio = 2 if debug else args.g_mlp
        depth = [5, 4, 2] if debug else [int(i) for i in args.g_depth.split(",")]
        self.latent_dim = 256 if debug else args.latent_dim
        self.l1 = nn.Linear(self.latent_dim, (self.bottom_width ** 2) * self.embed_dim)
        self.blocks = build_transformer_encoder(  # [-1,c,8,8] -> []
                        image_size=(8, 8),
                        patch_size=(1, 1),
                        depth=depth[0],  # 5
                        dim=self.embed_dim,  # 1024
                        num_heads=self.embed_dim//64,  # 4
                        mlp_ratio=mlp_ratio,  # 4
                        channels=embed_dim//4,
                        )
        self.upsample_blocks = nn.ModuleList([
            build_transformer_encoder(
                image_size=(16, 16),
                patch_size=(1, 1),
                depth=depth[1],  # 4
                dim=self.embed_dim//4,  # 256
                num_heads=(self.embed_dim//4//64),  # 4
                mlp_ratio=mlp_ratio,  # 4
                channels=self.embed_dim//16,
            ),
            build_transformer_encoder(
                image_size=(32, 32),
                patch_size=(1, 1),
                depth=depth[2],  # 2
                dim=self.embed_dim//16,  # 64
                num_heads=(self.embed_dim//16//64),  # 4
                mlp_ratio=mlp_ratio,  # 4
                channels=self.embed_dim//16,
            ),
        ])
        self.deconv = nn.Sequential(
            nn.Conv2d(self.embed_dim//16, 3, 1, 1, 0)
        )

        self.initialize_weights()

    def initialize_weights(self):
        self._init_conv(self.deconv)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_conv(self,m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, z, epoch):
        H, W = self.bottom_width, self.bottom_width
        if self.latent_norm:
            latent_size = z.size(-1)
            z = (z / z.norm(dim=-1, keepdim=True) * (latent_size ** 0.5))

        x = self.l1(z)  # [4,256] -> [4,8*8*256]
        x = rearrange(x, 'b (h w c) -> b (h w) c', h=H, w=W, c=self.embed_dim).contiguous()  # [4,8*8*256] -> [4,64,1024]
        x = self.blocks(x, embedded=True)  # [b,64,8,8] -> [b,4*4,1024]
        x, H, W = pixel_upsample(x, H, W)  # [4,4*4,1024] -> [4, 16*16, 256] -> [4,1024,256]
        x = self.upsample_blocks[0](x, embedded=True)   # [4, 16*16, 256]  -> [4, 16*16, 256]
        x, H, W = pixel_upsample(x, H, W)  # [4, 8*8, 1024] -> [4, 32*32, 64]
        x = self.upsample_blocks[1](x, embedded=True)  # [4, 32*32, 64] -> [4, 32*32, 64]
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W).contiguous()
        output = self.deconv(x)  # reshape -> [4,64,32,32] -> [4,3,32,32]
        return output


def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return nn.AvgPool2d(kernel_size=2)(x)




if __name__ == '__main__':
    net = Generator(debug=True)
    z = torch.FloatTensor(np.random.normal(0, 1, (4, 256)))
    _ = net(z, epoch=1)
