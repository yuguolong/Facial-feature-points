from torch.utils.data import Dataset
from torchvision import transforms
import torch
import cv2
import numpy as np

def resize_image(image, width, height):
    top, bottom, left, right = (0, 0, 0, 0)
    # 获取图像尺寸
    h, w, _ = image.shape
    # 计算短边需要增加多上像素宽度使其与长边等长
    if h < w:
        dh = w - h
        top = dh // 2
        bottom = dh - top
    else:
        dw = h - w
        left = dw // 2
        right = dw - left
    # else:   #考虑相等的情况（感觉有点多余，其实等于0时不影响结果）
    #     pass
    # RGB颜色
    BLACK = [0, 0, 0]
    # 给图像增加边界，是图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
    # 调整图像大小并返回
    constant = cv2.resize(constant, (width, height))
    return constant

class FaceLandmarksDataset(Dataset):
    def __init__(self, txt_file):
        self.transform = transforms.Compose([transforms.ToTensor()])
        lines = []
        with open(txt_file) as read_file:
            for line in read_file:
                line = line.replace('\n', '')
                lines.append(line)
        self.landmarks_frame = lines

    def __len__(self):
        return len(self.landmarks_frame)

    def num_of_samples(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
           idx = idx.tolist()
        contents = self.landmarks_frame[idx].split('\t')
        image_path = contents[0]
        img = cv2.imread(image_path)  # BGR order
        h, w, c = img.shape
        # rescale
        # img = cv2.resize(img, (64, 64))
        img = resize_image(img,64,64)
        img = (np.float32(img) /255.0 - 0.5) / 0.5
        landmarks = np.zeros(10, dtype=np.float32)
        for i in range(1, len(contents), 2):
           landmarks[i - 1] = np.float32(contents[i]) / w
           landmarks[i] = np.float32(contents[i + 1]) / h
        # landmarks = landmarks.astype('float32').reshape(-1, 2)
        # H, W C to C, H, W
        img = img.transpose((2, 0, 1))
        sample = {'image': torch.from_numpy(img), 'landmarks': torch.from_numpy(landmarks)}
        return sample
