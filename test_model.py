import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import numpy as np
import torch
import cv2
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from PIL import Image
from components.metric_new_crop import *
import torch.nn.functional as F
from torch.autograd import Variable
import math
from components.cal_mean_std import Calculate_mean_std
from utilts import access_model


def make_dir(path):
    if os.path.exists(path) == False:
        os.makedirs(path)

choose_model = 'LCAE'
model_func = access_model(choose_model)
choose_dataset = 'SIRST_1_1_point_new'

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TEST_BATCH_SIZE = 1
NUM_WORKERS = 4
PIN_MEMORY = True
root_path = os.path.abspath('.')
dataset_path = os.path.join(root_path, 'dataset', choose_dataset)
test_dataset_path = os.path.join(dataset_path, 'val')
output_path = os.path.join(test_dataset_path, 'pre_results')
make_dir(output_path)

VAL_MASK_DIR = os.path.join(dataset_path, 'val', 'mask')
VAL_IMG_DIR = os.path.join(dataset_path, 'val', 'img')

num_images = len(os.listdir(VAL_MASK_DIR))
test_model_path = 'your_own.pth.tar'

def test_pred(img, net, batch_size, choose_model=choose_model):
    b, c, h, w = img.shape
    patch_size = 1024
    stride = 1024

    if h > patch_size and w > patch_size:
        img_unfold = F.unfold(img, kernel_size=patch_size, stride=stride)
        img_unfold = img_unfold.reshape(b, c, patch_size, patch_size, -1).permute(0, 4, 1, 2, 3)
        patch_num = img_unfold.size(1)

        preds_list = []
        for i in range(0, patch_num, batch_size):
            end = min(i + batch_size, patch_num)
            batch_patches = img_unfold[:, i:end, :, :, :].reshape(-1, c, patch_size, patch_size)
            batch_patches = Variable(batch_patches.float())
            batch_preds = net.forward(batch_patches)
            preds_list.append(batch_preds)

        preds_unfold = torch.cat(preds_list, dim=0).permute(1, 2, 3, 0)
        preds_unfold = preds_unfold.reshape(b, -1, patch_num)
        preds = F.fold(preds_unfold, kernel_size=patch_size, stride=stride, output_size=(h, w))
    else:
        preds = net.forward(img)

    return preds

class SirstDataset(Dataset):
    def __init__(self, image_dir, mask_dir, patch_size, transform=None, mode='None'):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = np.sort(os.listdir(image_dir))
        self.mode = mode
        self.patch_size = patch_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = (mask > 127.5).astype(float)

        h, w, c = image.shape
        times = 32
        pad_height = math.ceil(h / times) * times - h
        pad_width = math.ceil(w / times) * times - w
        image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
        mask = np.pad(mask, ((0, pad_height), (0, pad_width)), mode='constant')
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask, h, w, self.images[index]

def main():
    origin_img_dir = os.path.join(dataset_path, 'origin', 'img')
    cal_mean, cal_std = Calculate_mean_std(origin_img_dir)
    test_transforms = A.Compose(
        [
            A.Normalize(
                mean=cal_mean,
                std=cal_std,
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_ds = SirstDataset(
        image_dir=VAL_IMG_DIR,
        mask_dir=VAL_MASK_DIR,
        patch_size=256,
        transform=test_transforms,
        mode='test',
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=TEST_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
    )

    model = model_func().to(DEVICE)

    model.load_state_dict({k.replace('module.', ''): v for k, v in
                           torch.load(test_model_path, map_location=DEVICE)[
                               'state_dict'].items()})
    model.eval()

    iou_metric = SigmoidMetric()
    nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
    FA_PD_metric = PD_FA_2(1)
    iou_metric.reset()
    nIoU_metric.reset()
    FA_PD_metric.reset()

    loop = tqdm(val_loader)
    with torch.no_grad():
        for batch_idx, (x, y, h, w, name) in enumerate(loop):
            x = x.to(device=DEVICE).clone().detach()
            y = y.unsqueeze(1).to(device=DEVICE).clone().detach()
            preds_no_sigmoid = test_pred(x, model, 16)
            preds_no_sigmoid = preds_no_sigmoid[0, :, :h, :w]
            y = y[0, :, :h, :w]
            preds = torch.sigmoid(preds_no_sigmoid)
            pred = preds[0].cpu().data.numpy()
            pred_target = np.where(pred > 0.5, 255, 0)
            pred_target = np.array(pred_target, dtype='uint8')

            cv2.imwrite(os.path.join(output_path, name[0]), pred_target)

            iou_metric.update(preds, y)
            nIoU_metric.update(preds, y)
            FA_PD_metric.update(preds, y)
            _, IoU = iou_metric.get()
            _, nIoU = nIoU_metric.get()
            FA, PD = FA_PD_metric.get(num_images)
        print(IoU)
        print(nIoU)
        print(PD)
        print(FA)

if __name__ == "__main__":
    main()
