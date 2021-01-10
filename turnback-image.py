import mxnet as mx
import cv2
import os
from mxnet.gluon import nn
from mxnet import nd
from mxnet.gluon.model_zoo import vision
import numpy as np
from os.path import isfile, isdir, join, splitext
from os import listdir


resnet = vision.resnet18_v1(pretrained=False)
net=nn.Sequential()
net.add(resnet.features)
net.output=nn.Dense(8)
net.load_parameters('tmp.params', ctx=mx.cpu(0))

img=cv2.imread('./image/as.jpg').astype('float32') / 255
img=cv2.resize(img, (128,64), interpolation=cv2.INTER_CUBIC)
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img1=img
img=nd.array(img.transpose((2, 0, 1))).expand_dims(0)
a=net(img)[0].asnumpy()

dst_points = np.float32([[0,0], [128,0], [0,64], [128,64]])
src_points = np.float32([[a[0],a[1]], [a[2],a[3]], [a[4],a[5]], [a[6],a[7]]])

projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
img_output = cv2.warpPerspective(img1, projective_matrix, (128,64))*255

cv2.imwrite("./image/output.jpg",img_output)
#cv2.waitKey()