import torch
import cv2 as cv
import numpy as np
import torch.nn as nn

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=3,padding=1),
            # nn.BatchNorm2d(8),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            torch.nn.Conv2d(8, 32, kernel_size=3,padding=1),
            # nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3,padding=1),
            # nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
			nn.Linear(64*8*8, 128),
            # nn.BatchNorm1d(128),
            # nn.Dropout(0.3),
			nn.Linear(128, 10))

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)  # reshape
        out = self.fc(out)
        # out = out.view(out.size(0),5,2)  # torch.Size([64, 5, 2])
        return out

cnn = CNN()

cnn.load_state_dict(torch.load("./logs/model.pkl"))    # capture = cv.VideoCapture(0)
frame = cv.imread('./images/4001.jpg')
w,h,_ = frame.shape


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
    constant = cv.copyMakeBorder(image, top, bottom, left, right, cv.BORDER_CONSTANT, value=BLACK)
    # 调整图像大小并返回
    constant = cv.resize(constant, (width, height))
    return constant
# capture = cv.VideoCapture(0)
# while True:
#     ret, frame = capture.read()

img1 = resize_image(frame,64, 64)
# img1 = cv.resize(frame, (64, 64))
img = (np.float32(img1) / 255.0 - 0.5) / 0.5
img = img.transpose((2, 0, 1))
x_input = torch.from_numpy(img).view(1, 3, 64, 64)
probs = cnn(x_input)

lm_pts = probs.view(5, 2).cpu().detach().numpy()
for x, y in lm_pts:
    x1 = x*h
    y1 = y*w
    # print(x1,y1)
    cv.circle(frame, (np.int32(x1), np.int32(y1)), 2, (0, 0, 255), 2, 8, 0)

cv.imshow('image',frame)
cv.imwrite('./result/result1.jpg',frame)
cv.waitKey(0)

