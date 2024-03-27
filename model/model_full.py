import functools

import torch
import torch.nn as nn
from .networks import Encoder, Denoiser, Decoder, SEBlock, get_norm_layer
from base.base_model import BaseModel
from utils.util import torch_laplacian

def normalize_to_0_1(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor
# class DeblurModel(nn.Module):
#     """
#     去模糊模型,包括编码器、去噪器和解码器  
#     输入:
#         - blurred_image (N, 1, 256, 320)
#         - events (N, 13, 256, 320)
#     输出: 
#         - log_diff (N, 1, 256, 320)  
#         - sharp_image (N, 1, 256, 320)
#     """
class DefaultModel(BaseModel):
    """
    去模糊模型,包括编码器、去噪器和解码器,使用skip connection
    输入:
        - blurred_image (N, 1, 256, 320)
        - events (N, 13, 256, 320)
    输出:
        - log_diff (N, 1, 256, 320)
        - sharp_image (N, 1, 256, 320)
    """
    def __init__(self, init_dim=64, n_ev=13, grid=(8, 10), norm_type='instance', use_dropout=False, rgb=False):
        super(DefaultModel, self).__init__()

        norm_layer = get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        self.rgb = rgb
        self.encoder_b = nn.ModuleList([
            Encoder(1, 32),   # (N, 1, 256, 320) -> down(N, 32, 128, 160), self(N, 32, 256, 320)
            Encoder(32, 64),  # (N, 32, 128, 160) -> (N, 64, 64, 80), (N, 64, 128, 160)
            Encoder(64, 128), # (N, 64, 64, 80) -> (N, 128, 32, 40), (N, 128, 64, 80)
            Encoder(128, 256), # (N, 128, 32, 40) -> (N, 256, 16, 20), (N, 256, 32, 40)

        ])
        self.encoder_e = nn.ModuleList([
            Encoder(13, 32),  # (N, 13, 256, 320) -> (N, 32, 128, 160), (N, 32, 256, 320)
            Encoder(32, 64),  # (N, 32, 128, 160) -> (N, 64, 64, 80), (N, 64, 128, 160)  
            Encoder(64, 128), # (N, 64, 64, 80) -> (N, 128, 32, 40), (N, 128, 64, 80)
            Encoder(128, 256), # (N, 128, 32, 40) -> (N, 256, 16, 20), (N, 256, 32, 40)
        ])
        self.seb1 = SEBlock(512, 8)
        self.seb2 = SEBlock(1, 1)
        self.denoiser = Denoiser(512, 256) # (N, 512, 16, 20) -> (N, 256, 16, 20)
        self.decoder = nn.ModuleList([
            Decoder(256, 128), # (N, 256, 16, 20) -> (N, 128, 32, 40)
            Decoder(128, 64),  # (N, 128, 32, 40) -> (N, 64, 64, 80)
            Decoder(64, 32),   # (N, 64, 64, 80) -> (N, 32, 128, 160)
            Decoder(32, 1),   # (N, 64, 64, 80) -> (N, 32, 128, 160)
        ])
        self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # (N, 1, 128, 160) -> (N, 1, 256, 320)
            )
        #optical flow, bidirectional
        # self.flow_block = nn.Sequential(
        #     nn.Conv2d(1, init_dim // 8, kernel_size=3, stride=1, padding=1),
        #     norm_layer(init_dim // 8),
        #     nn.ReLU(True),
        #     nn.Conv2d(init_dim // 8, init_dim // 4, kernel_size=1, stride=1),
        #     norm_layer(init_dim // 4),
        #     nn.ReLU(True),
        #     nn.Conv2d(init_dim // 4, init_dim // 2, kernel_size=1, stride=1),
        #     norm_layer(init_dim // 2),
        #     nn.ReLU(True),
        #     nn.Conv2d(init_dim // 2, init_dim // 4, kernel_size=1, stride=1),
        #     norm_layer(init_dim // 4),
        #     nn.ReLU(True),
        #     nn.Conv2d(init_dim // 4, init_dim // 8, kernel_size=1, stride=1),
        #     norm_layer(init_dim // 8),
        #     nn.ReLU(True),
        #     nn.Conv2d(init_dim // 8, 4, kernel_size=1, stride=1),
        #     nn.ReLU(True)
        # )
        self.tanh = nn.Tanh()

    def forward(self, blurred_image, events):
        b_code = torch_laplacian(blurred_image)
        e_code = events
        b_codes = []
        e_codes = []

        for b_enc, e_enc in zip(self.encoder_b, self.encoder_e):
            b_code, b_skip = b_enc(b_code)
            e_code, e_skip = e_enc(e_code)
            b_codes.append(b_skip)
            e_codes.append(e_skip)

        code = torch.cat([b_code, e_code], 1)
        print("code.shape after cat", code.shape) # ([2, 512, 16, 20])
        code = self.seb1(code)
        code = self.denoiser(code)
        print("code.shape", code.shape) # [1, 256, 16, 20])

        for dec, b_skip, e_skip in zip(self.decoder, reversed(b_codes), reversed(e_codes)):
            print("++++++++++++++++++++++++++++++++++++++++++++++++++")
            skips = torch.cat([b_skip, e_skip], 1)
            print("skips", skips.shape)
            code = dec(code, skips)
            print("code.shape", code.shape)
        print("code.shape after decode",code.shape)
        # code = self.seb2(code)

        # flow = self.flow_block(code)
        # code = self.up(code)
        # log_diff = code
        log_diff = torch.neg(code)
        log_diff = self.tanh(log_diff)
        sharp_image = log_diff + blurred_image
        # sharp_image = normalize_to_0_1(sharp_image)
        print(sharp_image.shape)
        # return flow, log_diff, sharp_image, log_diff
        return log_diff, sharp_image, log_diff