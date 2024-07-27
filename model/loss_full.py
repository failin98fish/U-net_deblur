import torch
import torch.nn.functional as F

from .loss_utils.total_variation_loss import TVLoss
from .networks import CONV3_3_IN_VGG_19
from utils.util import torch_laplacian

tv = TVLoss()

def denoise_loss(Bi_clean_pred, Bi_clean_gt, **kwargs):
    l1_loss_lambda = kwargs.get('l1_loss_lambda', 1)
    l1_loss = F.l1_loss(Bi_clean_pred, Bi_clean_gt) * l1_loss_lambda
    print('denoise_loss: l1_loss:', l1_loss.item())

    l2_loss_lambda = kwargs.get('l2_loss_lambda', 1)
    l2_loss = F.mse_loss(Bi_clean_pred, Bi_clean_gt) * l2_loss_lambda
    print('denoise_loss: l2_loss:', l2_loss.item())

    return l1_loss + l2_loss


def reconstruction_loss(S_pred, S_gt, **kwargs):
    l2_loss_lambda = kwargs.get('l2_loss_lambda', 1)
    l2_loss = F.mse_loss(S_pred, S_gt) * l2_loss_lambda
    print('reconstruction_loss: l2_loss:', l2_loss.item())

    rgb = kwargs.get('rgb', False)
    model = CONV3_3_IN_VGG_19
    if rgb:
        S_pred_feature_map = model(S_pred)
        S_feature_map = model(S_gt).detach()  # we do not need the gradient of it
    else:
        S_pred_feature_map = model(torch.cat([S_pred] * 3, dim=1))
        S_feature_map = model(torch.cat([S_gt] * 3, dim=1)).detach()  # we do not need the gradient of it

    perceptual_loss_lambda = kwargs.get('perceptual_loss_lambda', 1)
    perceptual_loss = F.mse_loss(S_pred_feature_map, S_feature_map) * perceptual_loss_lambda
    print('reconstruction_loss: perceptual_loss:', perceptual_loss.item())

    return l2_loss + perceptual_loss


def edge_loss(pred, target):
    pred_edge = torch.abs(torch_laplacian(pred))
    target_edge = torch.abs(torch_laplacian(target))
    return F.l1_loss(pred_edge, target_edge)



def loss_full(Bi_clean_pred, Bi_clean_gt, S_pred, S_gt, code, **kwargs):
    # Lf_lambda = kwargs.get('Lf_lambda', 1)
    # Lf = flow_loss(F_pred, F_gt, **kwargs['flow_loss']) * Lf_lambda
    # print('Lf:', Lf.item())

    Lr_lambda = kwargs.get('Lr_lambda', 1)
    Lr = reconstruction_loss(S_pred, S_gt, **kwargs['reconstruction_loss']) * Lr_lambda
    print('Lr:', Lr.item())

    Ld_lambda = kwargs.get('Ld_lambda', 1)
    Ld = denoise_loss(Bi_clean_pred, Bi_clean_gt, **kwargs['denoise_loss']) * Ld_lambda
    print('Ld:', Ld.item())

<<<<<<< HEAD
    loss_log_diff = torch.mean(torch.abs(code))  # log difference的L1正则化项
    print('loss_log_diff:', 0.1*loss_log_diff)
    
    
    # tv_loss_lambda = kwargs.get('tv_loss_lambda', 1)
    # tv_loss = tv(S_pred) * tv_loss_lambda
    # print(' tv_loss:', tv_loss)
=======
    loss_log_diff = torch.mean(torch.abs(code))  # log difference的L1正则化项日志差异正则化项 (Log Difference Regularization)：
#    - 这个正则化项鼓励编码（code）的值接近于零，可以看作是一种稀疏性约束。
#    - 在事件相机图像去模糊中，这种正则化可以帮助网络学习更紧凑和信息丰富的编码表示。
#    - 但是，需要注意权衡这个正则化项的权重，以免过度约束编码并影响重建质量。

    print('loss_log_diff:', 0.1*loss_log_diff)
    
    
    tv_loss_lambda = kwargs.get('tv_loss_lambda', 1)
    tv_loss = tv(S_pred) * tv_loss_lambda
    print(' tv_loss:', tv_loss)
#     TV损失鼓励生成的图像具有平滑性，减少噪声和伪影。
#    - 在事件相机图像去模糊中，TV损失可以帮助生成更干净和视觉上更令人满意的结果。
#    - 但是，需要注意权衡TV损失的权重，以免过度平滑图像并损失细节。
>>>>>>> 2e3ab4e04d86ce1a97b7fa56110e2f3affdde98d

    return Ld + Lr + 0.1*loss_log_diff
