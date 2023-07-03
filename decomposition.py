import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional
import numpy as np
import glob
import cv2
import losses
from math import sqrt
import os
from model import *
from losses import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Constants
batch_size = 8
patch_size_x = 224
patch_size_y = 224

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model
model = SIDNet().to(device)

# Loss functions
criterion_recon_vi = model.recon_loss_vi  # 重构损失可见光
criterion_recon_ir = model.recon_loss_ir  # 重构损失红外光
criterion_mutual = model.mutual_i_loss  # 相互一致性损失
criterion_perceptual = model.perceptual_loss  # 感知损失
criterion_mutual_i_input_loss = model.mutual_i_input_loss  # 照度平滑度损失

# Optimizer
learning_rate = sqrt(1.6)*1e-4
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Prepare data
train_ir_data = []
train_vi_data = []
train_vi_3_data = []
eval_ir_data = []
eval_vi_data = []
eval_vi_3_data = []

train_ir_data_names = glob.glob('./ours_dataset_240/train/infrared/*.jpg')
train_vi_data_names = glob.glob('./ours_dataset_240/train/visible/*.jpg')
train_ir_data_names.sort()
train_vi_data_names.sort()
print('[*] Number of training data_ir/vi: %d' % len(train_ir_data_names))
for idx in range(len(train_ir_data_names)):
    im_before_ir = load_images(train_ir_data_names[idx])
    ir_gray = cv2.cvtColor(im_before_ir, cv2.COLOR_RGB2GRAY)
    train_ir_data.append(ir_gray)
    im_before_vi = load_images(train_vi_data_names[idx])
    vi_gray = cv2.cvtColor(im_before_vi, cv2.COLOR_RGB2GRAY)
    vi_y = rgb_ycbcr_np(im_before_vi)[:, :, 0]
    train_vi_data.append(vi_y)  # 归一化之后的图像形成一个list组
    vi_rgb = np.zeros_like(im_before_vi)  # 和vgg匹配用
    vi_rgb[:, :, 0] = vi_y
    vi_rgb[:, :, 1] = vi_y
    vi_rgb[:, :, 2] = vi_y
    train_vi_3_data.append(vi_rgb)
    print(f'[{idx}/{len(train_ir_data_names)}] Train Image Loaded!')

eval_ir_data_names = glob.glob('./ours_dataset_240/test_50/infrared/*.jpg')
eval_vi_data_names = glob.glob('./ours_dataset_240/test_50/visible/*.jpg')
eval_ir_data_names.sort()
eval_vi_data_names.sort()
for idx in range(len(eval_ir_data_names)):
    eval_im_before_ir = load_images(eval_ir_data_names[idx])
    eval_ir_gray = cv2.cvtColor(eval_im_before_ir, cv2.COLOR_RGB2GRAY)
    eval_ir_data.append(eval_ir_gray)
    eval_im_before_vi = load_images(eval_vi_data_names[idx])
    eval_vi_gray = cv2.cvtColor(eval_im_before_vi, cv2.COLOR_RGB2GRAY)
    eval_vi_y = rgb_ycbcr_np(eval_im_before_vi)[:, :, 0]
    eval_vi_data.append(eval_vi_y)
    eval_vi_3 = np.zeros_like(eval_im_before_vi)  # 和vgg匹配
    eval_vi_3[:, :, 0] = eval_vi_y
    eval_vi_3[:, :, 1] = eval_vi_y
    eval_vi_3[:, :, 2] = eval_vi_y
    eval_vi_3_data.append(eval_vi_3)
    print(f'[{idx}/{len(eval_ir_data_names)}] Test Image Loaded!')

# Training settings
start_epoch = 0
iter_num = 0
end_epoch = 2000  # default value: 2000
eval_every_epoch = 200  # default value: 200
save_every_epoch = 100  # default value: 100
print_every_iter = 100  # default value: 100
image_id = 0
ckpt_dir = './checkpoint/decom_net_train/'
if not os.path.isdir(ckpt_dir):
    os.makedirs(ckpt_dir)
eval_dir = './decom_eval_results/'
if not os.path.isdir(eval_dir):
    os.makedirs(eval_dir)
print("[*] Start training...")

for epoch in range(start_epoch, end_epoch):
    for batch_id in range(len(train_ir_data) // batch_size):
        batch_input_ir = torch.zeros((batch_size, 1, patch_size_y, patch_size_x), dtype=torch.float32).to(device)
        batch_input_vi = torch.zeros((batch_size, 1, patch_size_y, patch_size_x), dtype=torch.float32).to(device)
        batch_input_vi_3 = torch.zeros((batch_size, 3, patch_size_y, patch_size_x), dtype=torch.float32).to(device)
        batch_input_vi_3_hist = torch.zeros((batch_size, 3, patch_size_y, patch_size_x), dtype=torch.float32).to(device)

        for patch_id in range(batch_size):
            # idx = np.random.randint(len(train_ir_data))
            img_ir = train_ir_data[image_id]
            img_vi = train_vi_data[image_id]
            img_vi_3 = train_vi_3_data[image_id]

            h, w = img_ir.shape[:2]
            x = np.random.randint(0, w - patch_size_x)
            y = np.random.randint(0, h - patch_size_y)

            img_ir_patch = img_ir[y:y + patch_size_y, x:x + patch_size_x]
            img_vi_patch = img_vi[y:y + patch_size_y, x:x + patch_size_x]
            img_vi_3_patch = img_vi_3[y:y + patch_size_y, x:x + patch_size_x]

            batch_input_ir[patch_id, :, :, :] = torch.from_numpy(img_ir_patch).unsqueeze(0).to(device)
            batch_input_vi[patch_id, :, :, :] = torch.from_numpy(img_vi_patch).unsqueeze(0).to(device)
            batch_input_vi_3[patch_id, :, :, :] = torch.from_numpy(img_vi_3_patch).permute(2, 0, 1).to(device)
            batch_input_vi_3_hist[patch_id, 0, :, :] = torch.from_numpy(hist(img_vi_3_patch[:, :, 0])).to(device)
            batch_input_vi_3_hist[patch_id, 1, :, :] = torch.from_numpy(hist(img_vi_3_patch[:, :, 1])).to(device)
            batch_input_vi_3_hist[patch_id, 2, :, :] = torch.from_numpy(hist(img_vi_3_patch[:, :, 2])).to(device)

            image_id = (image_id + 1) % len(train_ir_data)

        # Forward pass
        ir_r, vi_e_r, l_r = model(batch_input_vi, batch_input_ir)
        # vi_e_r_3 = torch.zeros_like(batch_input_vi_3_hist)
        # vi_e_r_3[:, 0, :, :] = vi_e_r
        # vi_e_r_3[:, 1, :, :] = vi_e_r
        # vi_e_r_3[:, 2, :, :] = vi_e_r
        vi_e_r_3 = torch.cat([vi_e_r, vi_e_r, vi_e_r], dim=1)

        # Compute losses
        recon_vi_loss = criterion_recon_vi(vi_e_r, l_r, batch_input_vi)
        recon_ir_loss = criterion_recon_ir(ir_r, batch_input_ir)
        mutual_loss = criterion_mutual(l_r)
        perceptual_loss = criterion_perceptual(vi_e_r_3, batch_input_vi_3_hist)
        mutual_i_input_loss = criterion_mutual_i_input_loss(l_r, batch_input_vi)

        total_loss = 1000 * recon_vi_loss + 2000 * recon_ir_loss + 9 * mutual_loss + 40 * perceptual_loss + 7 * mutual_i_input_loss

        # Backward and optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        iter_num += 1

        if iter_num % print_every_iter == 0:
            print(
                'Epoch [{}/{}], Step [{}], Total Loss: {:.4f}, Recon VI Loss: {:.4f}, Recon IR Loss: {:.4f}, Mutual Loss: {:.4f}, Perceptual Loss: {:.4f}, Mutual I Input Loss: {:.4f}'
                .format(epoch + 1, end_epoch, iter_num, total_loss.item(),
                        recon_vi_loss.item(), recon_ir_loss.item(), mutual_loss.item(), perceptual_loss.item(),
                        mutual_i_input_loss.item()))

    # Evaluation
    if (epoch + 1) % eval_every_epoch == 0:
        print(f"Evaluating epoch {epoch + 1} results...")
        model.eval()
        with torch.no_grad():
            for idx in range(len(eval_ir_data)):
                input_ir_eval = torch.from_numpy(eval_ir_data[idx]).unsqueeze(0).unsqueeze(0).to(device)
                input_vi_eval = torch.from_numpy(eval_vi_data[idx]).unsqueeze(0).unsqueeze(0).to(device)

                result_1, result_2, result_3 = model(input_ir_eval, input_vi_eval)  # vi_e_r, l_r, ir_r
                result_1_numpy = result_1.cpu().numpy()
                result_2_numpy = result_2.cpu().numpy()
                result_3_numpy = result_3.cpu().numpy()

                save_images(os.path.join(eval_dir, 'vi_%d_%d.png' % (idx + 1, epoch + 1)), result_1_numpy)
                save_images(os.path.join(eval_dir, 'dl_%d_%d.png' % (idx + 1, epoch + 1)), result_2_numpy)
                save_images(os.path.join(eval_dir, 'ir_%d_%d.png' % (idx + 1, epoch + 1)), result_3_numpy)

        model.train()

    # Save Checkpoints
    if (epoch + 1) % save_every_epoch == 0:
        print(f"Saving epoch {epoch + 1} models...")
        model_save_filename = 'decom_{}.pth'.format(epoch + 1)
        model_save_path = os.path.join(ckpt_dir, model_save_filename)
        torch.save(model.state_dict(), model_save_path)
        opt_save_filename = 'optimizer_{}.pth'.format(epoch + 1)
        opt_save_path = os.path.join(ckpt_dir, opt_save_filename)
        torch.save(optimizer.state_dict(), opt_save_path)

print("[*] Finish training for Decom Net")
