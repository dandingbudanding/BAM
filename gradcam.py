import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
import os

gpu_id = "0,1";  # 指定gpu id
# 配置环境  也可以在运行时临时指定 CUDA_VISIBLE_DEVICES='2,7' Python train.py
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id  # 这里的赋值必须是字符串，list会报错
device_ids = range(torch.cuda.device_count())  # torch.cuda.device_count()=2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []

        for name, module in self.model.module._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
                break
            elif "avg_pool" in name.lower():
                x = module(x)
                x = x.view(x.size(0), -1)
            else:
                x = module(x)

        return target_activations, x


def preprocess_image(img):
    # means = [0.485, 0.456, 0.406]
    # stds = [0.229, 0.224, 0.225]
    #
    preprocessed_img = img.copy()[:, :, ::-1]
    # for i in range(3):
    #     preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
    #     preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
    # cv2.imwrite("cam.jpg", np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)


        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input):
        return self.model.module(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image_path', type=str, default="../../classical_SR_datasets/Set5/LR_Down4_cubic/2.png",
                        help='Input image path')
    parser.add_argument('--image_path_save', type=str, default='./savedimg/',
                        help='save image path')
    # F://DATA//AMD_OCT_ZJU//test//3//active901.png
    # F://DATA//AMD_OCT_ZJU//test//2//inactive807.png
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)

from RRDBNet import RRDBNet
from myNet import myNet, myNet2 ,myNet3,myNet4,myNet4_,myNet4__,myNet5 #dataload 方式不一样
from RCAN import RCAN,RCAN_blancedattention
from MDSR.MDSR import MDSR,MDSR_blanced_attention
from SRCNN import SRCNN
from SRRESNET import _NetG
from IMDN import model3,model4,model5,model6,model7,model8,IMDN_BLANCED_CBAM,IMDN_CBAM,IMDN_BLANCED_ATTENTION,IMDN
from CARN.carn import CARN,CARN_blanced_attention
from CARN.carn_m import CARN_m,CARN_m_blanced_attention
from MSRN.msrn import MSRN,MSRN_blanced_attention
from EDSR.edsr import EDSR,EDSR_blanced_attention
from AWSRN.awsrn import AWSRN,AWSRN_blanced_attention
from OISR.oisr_LF_s import oisr_LF_s,oisr_LF_s_blanced_attention
from s_LWSR.s_LWSR import LWSR_blanced_attention
from RCAN_ORI.RCAN import RCAN_ori_blanced_attention
if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """
    parser = argparse.ArgumentParser()
    args = get_args()

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    # model = resnet34()

    model_weights = torch.load("./checkpoint/IMDN_BLANCED_ATTENTION/x4/best.pth")
    model = IMDN_BLANCED_ATTENTION().to(device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(model_weights)

    grad_cam = GradCam(model=model, feature_module=model.module.IMDB6.attention,
                       target_layer_names=["ca"], use_cuda=True)

    img = cv2.imread(args.image_path, 1)

    img = np.float32(img) / 255
    input = preprocess_image(img)
    # input=img

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = None
    mask = grad_cam(input, target_index)

    colorimg = show_cam_on_image(img, mask)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=True)
    print(model._modules.items())
    gb = gb_model(input, index=target_index)
    gb = gb.transpose((1, 2, 0))
    cam_mask = cv2.merge([mask, mask, mask])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    # cv2.imwrite(os.path.join(args.image_path_save,"gb",file), gb)
    # cv2.imwrite(os.path.join(args.image_path_save,"camgb",file), cam_gb)
    cv2.imwrite("./savedimg/baby.png", colorimg)  #
