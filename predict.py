"""
Code references:
# >>> https://github.com/cvlab-stonybrook/DewarpNet
# >>> https://github.com/fh2019ustc/DocGeoNet
"""
import argparse
import time

import cv2
import glob
import numpy as np
import os

import torch
import torch.nn.functional as F

from PIL import Image
from model import DewarpTextlineMaskGuide


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type=int, default=224, help='image size')
    parser.add_argument('--model_path', type=str, default='pretrained_models/30.pt', help='model path')
    parser.add_argument('--img_path', type=str, default='dataset/Dewarp/DocUNet_dataset/crop',
                        help='image path or path to folder containing images')

    parser.add_argument('--save_path', type=str, default='infer/', help='save path')

    return parser.parse_args()


def predict(img_path, save_path, filename, recti_model):
    assert os.path.exists(img_path), 'Incorrect Image Path'
    assert os.path.exists(save_path), 'Incorrect Save Path'

    img_size = parser.input_size

    img = np.array(Image.open(img_path))[:, :, :3] / 255.
    img_h, img_w, _ = img.shape
    input_img = cv2.resize(img, (img_size, img_size))

    with torch.no_grad():
        recti_model.eval()
        input_ = torch.from_numpy(input_img).permute(2, 0, 1).cuda()
        input_ = input_.unsqueeze(0)
        start = time.time()

        bm = recti_model(input_.float())
        bm = (2 * (bm / 223.) - 1) * 0.99
        ps_time = time.time() - start

    bm = bm.detach().cpu()
    bm0 = cv2.resize(bm[0, 0].numpy(), (img_w, img_h))  # x flow
    bm1 = cv2.resize(bm[0, 1].numpy(), (img_w, img_h))  # y flow
    bm0 = cv2.blur(bm0, (3, 3))
    bm1 = cv2.blur(bm1, (3, 3))
    lbl = torch.from_numpy(np.stack([bm0, bm1], axis=2)).unsqueeze(0).float()  # h * w * 2

    out = F.grid_sample(torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float(), lbl, align_corners=True)
    img_geo = ((out[0] * 255.).permute(1, 2, 0).numpy()).astype(np.uint8)

    cv2.imwrite(filename, img_geo[:, :, ::-1])  # save

    return ps_time


if __name__ == '__main__':
    parser = get_args()

    recti_model = DewarpTextlineMaskGuide(image_size=parser.input_size)
    recti_model = torch.nn.DataParallel(recti_model)
    state_dict = torch.load(parser.model_path, map_location='cpu')

    recti_model.load_state_dict(state_dict)
    recti_model.cuda()
    print(f'model loaded')

    img_path = parser.img_path
    save_path = parser.save_path
    total_time = 0.0

    start = time.time()
    img_num = 0.0
    for file in glob.glob(img_path + "/*"):  # img_names:  #
        print("file: ", file)
        filename = (save_path + "/" + file[file.rindex("/") + 1:file.rindex(".")] + ".png")

        total_time += predict(file, save_path, filename, recti_model)
        print("Written ", file[file.rindex("/") + 1:file.rindex(".")])
        img_num += 1
    print('FPS: %.1f' % (1.0 / (total_time / img_num)))

