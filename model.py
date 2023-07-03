# author:xxy,time:2022/2/23
import numpy as np
from PIL import Image
# import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
# import scipy.stats as st
# from skimage import io, data, color
# from functools import reduce
import cv2

import losses

############ Constants ############
batch_size = 5
patch_size_x = 224
patch_size_y = 224


############ Encoder ############
# 输入img为concat红外可见光图像的结果，通道数为2
# 输出为256个feature—map
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Weight and Bias initialization
        init.normal_(self.conv1.weight, mean=0, std=1e-3)
        init.normal_(self.conv2.weight, mean=0, std=1e-3)
        init.normal_(self.conv3.weight, mean=0, std=1e-3)
        init.normal_(self.conv4.weight, mean=0, std=1e-3)
        init.constant_(self.conv1.bias, 0.0)
        init.constant_(self.conv2.bias, 0.0)
        init.constant_(self.conv3.bias, 0.0)
        init.constant_(self.conv4.bias, 0.0)

    def forward(self, img):
        conv1_output = lrelu(self.bn1(self.conv1(img)))
        conv2_output = lrelu(self.bn2(self.conv2(conv1_output)))
        conv3_output = lrelu(self.bn3(self.conv3(conv2_output)))
        feature = lrelu(self.bn4(self.conv4(conv3_output)))
        return feature


############ Decoder ############
class DecoderIr(nn.Module):
    def __init__(self):
        super(DecoderIr, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(1)

        # Weight and Bias initialization
        init.normal_(self.conv1.weight, mean=0, std=1e-3)
        init.normal_(self.conv2.weight, mean=0, std=1e-3)
        init.normal_(self.conv3.weight, mean=0, std=1e-3)
        init.normal_(self.conv4.weight, mean=0, std=1e-3)
        init.constant_(self.conv1.bias, 0.0)
        init.constant_(self.conv2.bias, 0.0)
        init.constant_(self.conv3.bias, 0.0)
        init.constant_(self.conv4.bias, 0.0)

    def forward(self, feature_ir):
        conv1_output = lrelu(self.bn1(self.conv1(feature_ir)))
        conv2_output = lrelu(self.bn2(self.conv2(conv1_output)))
        conv3_output = lrelu(self.bn3(self.conv3(conv2_output)))
        ir_r = torch.sigmoid(self.bn4(self.conv4(conv3_output)))
        return ir_r


class DecoderViL(nn.Module):
    def __init__(self):
        super(DecoderViL, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(1)

        self.l_conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.l_bn1 = nn.BatchNorm2d(128)
        self.l_conv2 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        self.l_bn2 = nn.BatchNorm2d(64)
        self.l_conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.l_bn3 = nn.BatchNorm2d(32)
        self.l_conv4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.l_bn4 = nn.BatchNorm2d(1)

        # Weight initialization
        init.normal_(self.conv1.weight, mean=0, std=1e-3)
        init.normal_(self.conv2.weight, mean=0, std=1e-3)
        init.normal_(self.conv3.weight, mean=0, std=1e-3)
        init.normal_(self.conv4.weight, mean=0, std=1e-3)
        init.constant_(self.conv1.bias, 0.0)
        init.constant_(self.conv2.bias, 0.0)
        init.constant_(self.conv3.bias, 0.0)
        init.constant_(self.conv4.bias, 0.0)

        init.normal_(self.l_conv1.weight, mean=0, std=1e-3)
        init.normal_(self.l_conv2.weight, mean=0, std=1e-3)
        init.normal_(self.l_conv3.weight, mean=0, std=1e-3)
        init.normal_(self.l_conv4.weight, mean=0, std=1e-3)
        init.constant_(self.l_conv1.bias, 0.0)
        init.constant_(self.l_conv2.bias, 0.0)
        init.constant_(self.l_conv3.bias, 0.0)
        init.constant_(self.l_conv4.bias, 0.0)

    def forward(self, feature_vi_e, feature_l):
        conv1_output = lrelu(self.bn1(self.conv1(feature_vi_e)))
        conv2_output = lrelu(self.bn2(self.conv2(conv1_output)))
        conv3_output = lrelu(self.bn3(self.conv3(conv2_output)))
        vi_e_r = torch.sigmoid(self.bn4(self.conv4(conv3_output)))

        l_conv1_output = lrelu(self.l_bn1(self.l_conv1(feature_l)))
        l_conv1_output = torch.cat([l_conv1_output, conv1_output], dim=1)
        l_conv2_output = lrelu(self.l_bn2(self.l_conv2(l_conv1_output)))
        l_conv3_output = lrelu(self.l_bn3(self.l_conv3(l_conv2_output)))
        l_conv3_output = torch.cat([l_conv3_output, conv3_output], dim=1)
        l_r = torch.sigmoid(self.l_bn4(self.l_conv4(l_conv3_output)))

        return vi_e_r, l_r


############ CAM #############
# def CAM_IR(input_feature):
# def CAM_VI_E(input_feature):
# def CAM_L(input_feature):


############ Special Feature ############
# def get_sf_ir(vector_ir, feature):
# def get_sf_l(vector_l, feature):
# def get_sf_vi_e(vector_vi_e, feature):


############ Squeeze & Excitation Block ############
class SEBlock(nn.Module):
    def __init__(self):
        super(SEBlock, self).__init__()
        # CAM
        self.conv1 = nn.Conv2d(256, 32, kernel_size=3, stride=1, padding=1)
        init.normal_(self.conv1.weight, mean=0, std=1e-3)
        init.constant_(self.conv1.bias, 0.0)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 256, kernel_size=3, stride=1, padding=1)
        init.normal_(self.conv2.weight, mean=0, std=1e-3)
        init.constant_(self.conv2.bias, 0.0)
        self.bn2 = nn.BatchNorm2d(256)

    def forward(self, input_feature):
        x = lrelu(self.bn1(self.conv1(input_feature)))
        x = self.bn2(self.conv2(x))
        vector_feature = torch.mean(x, dim=[2, 3], keepdim=True)
        vector_feature = F.softmax(vector_feature, dim=1)

        # Special Feature
        output_feature = torch.mul(vector_feature, input_feature)
        return output_feature


############ All_model ############
class SIDNet(nn.Module):
    def __init__(self):
        super(SIDNet, self).__init__()

        # forward functions
        self.encoder = Encoder()
        self.CAM_IR = SEBlock()
        self.CAM_VI_E = SEBlock()
        self.CAM_L = SEBlock()
        self.decoder_ir = DecoderIr()
        self.decoder_vi_l = DecoderViL()

        # loss functions
        self.recon_loss_vi = losses.recon_loss_vi
        self.recon_loss_ir = losses.recon_loss_ir
        self.mutual_i_loss = losses.mutual_i_loss
        self.perceptual_loss = losses.perceptual_loss
        self.mutual_i_input_loss = losses.mutual_i_input_loss

    def forward(self, vi, ir):
        img = torch.cat([vi, ir], dim=1)
        feature = self.encoder(img)
        feature_ir = self.CAM_IR(feature)
        feature_vi_e = self.CAM_VI_E(feature)
        feature_l = self.CAM_L(feature)

        ir_r = self.decoder_ir(feature_ir)
        [vi_e_r, l_r] = self.decoder_vi_l(feature_vi_e, feature_l)

        return ir_r, vi_e_r, l_r


############ Tool ############
def lrelu(x, leak=0.2):
    return F.leaky_relu(x, negative_slope=leak)


def laplacian(input_tensor):
    # kernel = tf.constant([[0, 1, 0], [1, -4, 1], [0, 1, 0]], tf.float32)
    kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
    # gradient_orig = tf.abs(tf.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))
    gradient_orig = torch.abs(F.conv2d(input=input_tensor, weight=kernel, stride=1, padding='same'))
    grad_min = torch.min(gradient_orig)
    grad_max = torch.max(gradient_orig)
    grad_norm = (gradient_orig - grad_min) / (grad_max - grad_min + 0.0001)
    return grad_norm


def load_images(file):
    im = Image.open(file)
    img = np.array(im, dtype="float32") / 255.0
    # img_max = np.max(img)
    # img_min = np.min(img)
    # img_norm = np.float32((img - img_min) / np.maximum((img_max - img_min), 0.001))
    img_norm = np.float32(img)
    return img_norm


def hist(img):
    input_int = np.uint8((img * 255.0))
    input_hist = cv2.equalizeHist(input_int)
    input_hist = (input_hist / 255.0).astype(np.float32)
    return input_hist


def save_images(filepath, result_1, result_2=None, result_3=None):
    result_1 = np.squeeze(result_1)
    # result_1 = np.expand_dims(result_1, axis=-1)
    result_2 = np.squeeze(result_2)
    result_3 = np.squeeze(result_3)

    if not result_2.any():
        cat_image = result_1
    else:
        # result_2 = np.expand_dims(result_2, axis=-1)
        cat_image = np.concatenate([result_1, result_2], axis=1)
    if not result_3.any():
        cat_image = cat_image
    else:
        # result_3 = np.expand_dims(result_3, axis=-1)
        cat_image = np.concatenate([cat_image, result_3], axis=1)

    im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
    im.save(filepath, 'png')


def save_colored_images(filepath, y, cb, cr):
    vi_y = np.expand_dims(y, axis=-1)
    vi_cb = np.expand_dims(cb, axis=-1)
    vi_cr = np.expand_dims(cr, axis=-1)
    img_ycbcr = np.concatenate([vi_y, vi_cb, vi_cr], axis=-1)
    img_rgb = ycbcr_rgb_np(img_ycbcr)
    img = Image.fromarray(np.clip(img_rgb * 255.0, 0, 255.0).astype('uint8'))
    img.save(filepath, 'png')


def rgb_ycbcr(img_rgb):
    # R = tf.expand_dims(img_rgb[:, :, 0], axis=-1)
    # G = tf.expand_dims(img_rgb[:, :, 1], axis=-1)
    # B = tf.expand_dims(img_rgb[:, :, 2], axis=-1)
    R = torch.unsqueeze(img_rgb[:, :, 0], -1)
    G = torch.unsqueeze(img_rgb[:, :, 1], -1)
    B = torch.unsqueeze(img_rgb[:, :, 2], -1)
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128 / 255
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128 / 255
    img_ycbcr = torch.cat([Y, Cb, Cr], dim=-1)
    return img_ycbcr


def rgb_ycbcr_np(img_rgb):
    R = np.expand_dims(img_rgb[:, :, 0], axis=-1)
    G = np.expand_dims(img_rgb[:, :, 1], axis=-1)
    B = np.expand_dims(img_rgb[:, :, 2], axis=-1)
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128 / 255.0
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128 / 255.0
    img_ycbcr = np.concatenate([Y, Cb, Cr], axis=-1)
    return img_ycbcr


def ycbcr_rgb_np(img_ycbcr):
    Y = np.expand_dims(img_ycbcr[:, :, 0], axis=-1)
    Cb = np.expand_dims(img_ycbcr[:, :, 1], axis=-1)
    Cr = np.expand_dims(img_ycbcr[:, :, 2], axis=-1)
    R = Y + 1.402 * (Cr - 128 / 255)
    G = Y - 0.34414 * (Cb - 128 / 255) - 0.71414 * (Cr - 128 / 255)
    B = Y + 1.772 * (Cb - 128 / 255)
    img_rgb = np.concatenate([R, G, B], axis=-1)
    return img_rgb

# def shuffle_unit(x, groups):
#     with tf.variable_scope('shuffle_unit'):
#         n, h, w, c = x.get_shape().as_list()
#         x = tf.reshape(x, shape=tf.convert_to_tensor([tf.shape(x)[0], h, w, groups, c // groups]))
#         x = tf.transpose(x, tf.convert_to_tensor([0, 1, 2, 4, 3]))
#         x = tf.reshape(x, shape=tf.convert_to_tensor([tf.shape(x)[0], h, w, c]))
#     return x
