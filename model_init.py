import torch
import timm
import numpy as np
from PIL import Image
from torchvision import models, transforms

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
device = torch.device("cuda") if torch.cuda.is_available() else device

def load_preprocess (name = "inceptionv3"):
    if(name == "resnet50v2" or name == "resnet101v2" or name == "resnet152v2" or name == "inceptionv3" or name == "mobilenetv2" or name == "vgg16" or name == "vgg19" or name == "densenet121" or name == "densenet169" or name == "densenet201"):
        def PreprocessImages(images):
            #images = np.array(images)
            transformer = transforms.Compose([
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            images = transformer(images)
            return images.requires_grad_(True)
        return PreprocessImages
    if(name == "xception"):
        def PreprocessImages(images):
            images = np.array(images)
            transformer = transforms.Compose([
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            images = torch.tensor(images, dtype=torch.float32)
            images = transformer(images)
            return images.requires_grad_(True)
        return PreprocessImages

def load_model(name = "inceptionv3"):
    model = None
    if(name == "resnet50v2"):
        model = models.resnet50(weights='IMAGENET1K_V2').to(device)
    if(name == "resnet101v2"):
        model = models.resnet101(weights='IMAGENET1K_V2').to(device)
    if(name == "resnet152v2"):
        model = models.resnet152(weights='IMAGENET1K_V2').to(device)
    if(name == "inceptionv3"):
        model = models.inception_v3(weights='IMAGENET1K_V1', init_weights=False).to(device)
    if(name == "mobilenetv2"):
        model = models.mobilenet_v2(weights='IMAGENET1K_V2').to(device)
    if(name == "vgg16"):
        model = models.vgg16(weights='IMAGENET1K_V1').to(device)
    if(name == "vgg19"):
        model = models.vgg19(weights='IMAGENET1K_V1').to(device)
    if(name == "densenet121"):
        model = models.densenet121(weights='IMAGENET1K_V1').to(device)
    if(name == "densenet169"):
        model = models.densenet169(weights='IMAGENET1K_V1').to(device)
    if(name == "densenet201"):
        model = models.densenet201(weights='IMAGENET1K_V1').to(device)
    if(name == "xception"):
        model = timm.create_model('xception', pretrained=True).to(device)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.zero_grad()
    return model

def load_image_loader(name = "inceptionv3"):
    if(name == "resnet50v2" or name == "resnet101v2" or name == "resnet152v2" or name == "mobilenetv2"):
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(232),
            transforms.CenterCrop(224)
        ])
    if(name == "inceptionv3"):
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(342),
            transforms.CenterCrop(299)
        ])
    if(name == "vgg16" or name == "vgg19" or name == "densenet121" or name == "densenet169" or name == "densenet201"):
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224)
        ])

    if(name == "xception"):
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(342),
            transforms.CenterCrop(299)
        ])
    def LoadImages(images, from_file_paths = True):
        if (from_file_paths):
            return np.asarray([transformer(Image.open(file_path).convert('RGB')) for file_path in images])
        else:
            return np.asarray([transformer(images)])
    return LoadImages

def model_init(name = "inceptionv3"):
    return load_preprocess(name), load_model(name), load_image_loader(name)
