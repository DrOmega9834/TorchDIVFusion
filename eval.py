import glob
import cv2
import numpy as np
import os
import torch
from model import SIDNet, load_images, save_images, save_colored_images, rgb_ycbcr_np
# from skimage.color import rgb2ycbcr, ycbcr2rgb
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

ir_data = []
vi_data = []
vi_cb_data = []
vi_cr_data = []

main_dir = '../SwinFusion/Dataset/valsets/MSRS/'
ir_data_dir = os.path.join(main_dir, 'IR/*.png')
vi_data_dir = os.path.join(main_dir, 'vi/*.png')
ir_data_names = glob.glob(ir_data_dir)
vi_data_names = glob.glob(vi_data_dir)
ir_data_names.sort()
vi_data_names.sort()
for idx in range(len(ir_data_names)):
    im_before_ir = load_images(ir_data_names[idx])
    # ir_gray = cv2.cvtColor(im_before_ir, cv2.COLOR_RGB2GRAY)
    ir_data.append(im_before_ir)
    im_before_vi = load_images(vi_data_names[idx])
    vi_gray = cv2.cvtColor(im_before_vi, cv2.COLOR_RGB2GRAY)
    vi_y = rgb_ycbcr_np(im_before_vi)[:, :, 0]
    vi_cb = rgb_ycbcr_np(im_before_vi)[:, :, 1]
    vi_cr = rgb_ycbcr_np(im_before_vi)[:, :, 2]
    vi_data.append(vi_y)
    vi_cb_data.append(vi_cb)
    vi_cr_data.append(vi_cr)
    print(f'[{idx + 1}/{len(ir_data_names)}] Image Loaded!')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SIDNet().to(device)
model.load_state_dict(torch.load('./checkpoint/decom_net_train/decom_2000.pth'))
model.eval()

vi_enh_y_dir = os.path.join(main_dir, 'VI_Enh_Y/')
vi_enh_dir = os.path.join(main_dir, 'VI_Enh/')
ir_enh_dir = os.path.join(main_dir, 'IR_Enh/')
if not os.path.isdir(vi_enh_y_dir):
    os.makedirs(vi_enh_y_dir)
if not os.path.isdir(vi_enh_dir):
    os.makedirs(vi_enh_dir)
if not os.path.isdir(ir_enh_dir):
    os.makedirs(ir_enh_dir)

with torch.no_grad():
    for idx in range(len(ir_data)):
        input_ir_eval = torch.from_numpy(ir_data[idx]).unsqueeze(0).unsqueeze(0).to(device)
        input_vi_eval = torch.from_numpy(vi_data[idx]).unsqueeze(0).unsqueeze(0).to(device)

        fullpath = vi_data_names[idx]
        filename = fullpath.split('/')[-1]

        result_1, result_2, result_3 = model(input_ir_eval, input_vi_eval)  # vi_e_r, l_r, ir_r
        result_1_numpy = result_1.cpu().numpy()
        # result_2_numpy = result_2.cpu().numpy()
        result_3_numpy = result_3.cpu().numpy()

        save_images(os.path.join(vi_enh_y_dir, filename), result_1_numpy)
        # save_images(os.path.join(dir, 'dl_%d_%d.png' % (idx + 1)), result_2_numpy)
        save_images(os.path.join(ir_enh_dir, filename), result_3_numpy)

        vi_y = np.squeeze(result_1_numpy)
        vi_cb = vi_cb_data[idx]
        vi_cr = vi_cr_data[idx]
        save_colored_images(os.path.join(vi_enh_dir, filename), vi_y, vi_cb, vi_cr)

        print(f'[{idx + 1}/{len(ir_data)}] Image Saved!')
