import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import timm
import sys

class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        # Initialize the scaling factors
        self.scale_factors = nn.Parameter(torch.tensor([1.0, 1.0], dtype=torch.float))

    def forward(self, x):
        theta = torch.diag(self.scale_factors).view(1, 2, 2)
        theta = theta.repeat(x.size(0), 1, 1)  # Repeat for each image in the batch

        # Extend theta for affine_grid
        affine_mat = torch.zeros(theta.size(0), 2, 3).to(theta.device)
        affine_mat[:, :, :2] = theta
        affine_mat[:, :, 2] = 0  # No translation

        grid = F.affine_grid(affine_mat, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x
    
class ScaleAdaptiveModel(nn.Module):
    def __init__(self, pretrained_model):
        super(ScaleAdaptiveModel, self).__init__()
        self.backbone = pretrained_model
        self.fc_classifier = self.backbone.fc
        self.backbone.fc = nn.Identity()  # Use backbone for feature extraction only
        
        # STN Module
        self.stn = STN()

    def forward(self, x):
        # Apply scaling transformation
        x = self.stn(x)
        # Pass the transformed input through the backbone
        x = self.backbone(x)
        x = self.fc_classifier(x)
        return x
    

def crossvit(in_channels, n_classes, pretrained=False):
    model = timm.create_model('crossvit_15_240', pretrained=pretrained)
    if in_channels != 3:
        model.patch_embed[0].proj = nn.Conv2d(in_channels, 192, kernel_size=(12, 12), stride=(12, 12))
        model.patch_embed[1].proj = nn.Conv2d(in_channels, 384, kernel_size=(16, 16), stride=(16, 16))
    model.head[0] = nn.Linear(in_features=192, out_features=n_classes, bias=True)
    model.head[1] = nn.Linear(in_features=384, out_features=n_classes, bias=True)
    return model

# def crossvit(in_channels, n_classes, pretrained=False, img_size=256):
#     # Create the base model
#     model = timm.create_model('crossvit_15_240', pretrained=pretrained, img_size=img_size)
    
#     # Modify the input layers (patch embeddings) for both branches
#     model.patch_embed[0].proj = nn.Conv2d(in_channels, 192, kernel_size=(12, 12), stride=(12, 12))
#     model.patch_embed[1].proj = nn.Conv2d(in_channels, 384, kernel_size=(16, 16), stride=(16, 16))
    
#     # Modify the output layers (heads) for both branches
#     model.head[0] = nn.Linear(in_features=192, out_features=n_classes, bias=True)
#     model.head[1] = nn.Linear(in_features=384, out_features=n_classes, bias=True)
    
#     # Adjust positional embeddings for new image size
#     for i in range(2):
#         pos_embed = model.pos_embed_0 if i == 0 else model.pos_embed_1
#         num_patches = (img_size // model.patch_embed[i].patch_size[0]) ** 2
#         num_extra_tokens = 1  # cls token
#         new_size = num_patches + num_extra_tokens
#         if pos_embed.shape[1] != new_size:
#             # Resize pos embedding
#             pos_embed_resized = torch.nn.functional.interpolate(
#                 pos_embed.permute(0, 2, 1).unsqueeze(0), 
#                 size=new_size, 
#                 mode='linear'
#             )
#             pos_embed_resized = pos_embed_resized.squeeze(0).permute(0, 2, 1)
#             if i == 0:
#                 model.pos_embed_0 = nn.Parameter(pos_embed_resized)
#             else:
#                 model.pos_embed_1 = nn.Parameter(pos_embed_resized)
    
#     return model


def vit_base(in_channels, n_classes, pretrained=False):
    model = timm.create_model('vit_base_patch16_224', pretrained=pretrained, img_size=256)
    model.patch_embed.proj = nn.Conv2d(in_channels, 768, kernel_size=(16, 16), stride=(16, 16))
    model.head = nn.Linear(in_features=768, out_features=n_classes, bias=True)
    return model


def xcit_small(in_channels, n_classes, pretrained=False):
    model = timm.create_model('xcit_small_12_p8_224', pretrained=pretrained)
    model.patch_embed.proj[0][0] = nn.Conv2d(in_channels, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model.head = nn.Linear(in_features=384, out_features=n_classes, bias=True)
    return model

def xcit_medium(in_channels, n_classes, pretrained=False):
    model = timm.create_model('xcit_medium_24_p8_224', pretrained=pretrained)
    model.patch_embed.proj[0][0] = nn.Conv2d(in_channels, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model.head = nn.Linear(in_features=512, out_features=n_classes, bias=True)
    return model

def densenet161_(in_channels, n_classes, pretrained=False):
    model = models.densenet161(pretrained=pretrained)
    model.features.conv0 = nn.Conv2d(in_channels, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.classifier = nn.Sequential(nn.BatchNorm1d(2208),
                            nn.Dropout(p=0.5, inplace=False),
                            nn.Linear(in_features = 2208, out_features=512, bias=False),
                            nn.ReLU(inplace=True),

                            nn.BatchNorm1d(512),
                            nn.Dropout(p=0.5, inplace=False),
                            nn.Linear(in_features = 512, out_features=64, bias=False),
                            nn.ReLU(inplace=True),
                            
                            nn.BatchNorm1d(64),
                            nn.Dropout(p=0.5, inplace=False),
                            nn.Linear(in_features=64, out_features=n_classes, bias=True)
                            )
    return model

def resnet34_(in_channels, n_classes, dropout=0.5, weights=None):
    model = models.resnet34(weights=weights)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Sequential(
                            nn.BatchNorm1d(512),
                            nn.Dropout(p=dropout, inplace=False),
                            nn.Linear(in_features = 512, out_features=64, bias=False),
                            nn.ReLU(inplace=True),
                            
                            nn.BatchNorm1d(64),
                            nn.Dropout(p=dropout, inplace=False),
                            nn.Linear(in_features=64, out_features=n_classes, bias=True)
                            )
    return model


def resnet50_(in_channels, n_classes, dropout=0.5, weights=None):
    model = models.resnet50(weights=weights)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Sequential(nn.BatchNorm1d(2048),
                            nn.Dropout(p=dropout, inplace=False),
                            nn.Linear(in_features = 2048, out_features=512, bias=False),
                            nn.ReLU(inplace=True),

                            nn.BatchNorm1d(512),
                            nn.Dropout(p=dropout, inplace=False),
                            nn.Linear(in_features = 512, out_features=64, bias=False),
                            nn.ReLU(inplace=True),
                            
                            nn.BatchNorm1d(64),
                            nn.Dropout(p=dropout, inplace=False),
                            nn.Linear(in_features=64, out_features=n_classes, bias=True)
                            )
    return model


def resnet50_gn_(in_channels, n_classes, pretrained=False):
    model = models.resnet50(pretrained=pretrained)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Sequential(nn.GroupNorm(1, 2048),  # Using GroupNorm to prevent error when batch=1
                             nn.Dropout(p=0.5, inplace=False),
                             nn.Linear(in_features=2048, out_features=512, bias=False),
                             nn.ReLU(inplace=True),

                             nn.GroupNorm(1, 512),  # Using GroupNorm to prevent error when batch=1
                             nn.Dropout(p=0.5, inplace=False),
                             nn.Linear(in_features=512, out_features=64, bias=False),
                             nn.ReLU(inplace=True),
                             
                             nn.GroupNorm(1, 64),  # Using GroupNorm to prevent error when batch=1
                             nn.Dropout(p=0.5, inplace=False),
                             nn.Linear(in_features=64, out_features=n_classes, bias=True)
                            )
    return model

class fpn_model(nn.Module):
    def __init__(self, backbone):
        super(fpn_model, self).__init__() # Initialize self._modules as OrderedDict

        self.backbone = backbone
        
        self.classifier = nn.Sequential(  
                        nn.BatchNorm1d(256),
                        nn.Dropout(p=0.5, inplace=False),
                        nn.Linear(in_features = 256, out_features=64, bias=False),
                        nn.ReLU(inplace=True), 

                        nn.BatchNorm1d(64),
                        nn.Dropout(p=0.5, inplace=False),
                        nn.Linear(in_features=64, out_features=5, bias=True)
                        )
            
    def merge_orderdict(self, x):
        return torch.cat(( 
                        nn.AdaptiveAvgPool2d(output_size=(1))(x['0']), 
                        nn.AdaptiveAvgPool2d(output_size=(1))(x['1']), 
                        nn.AdaptiveAvgPool2d(output_size=(1))(x['2']), 
                        nn.AdaptiveAvgPool2d(output_size=(1))(x['3']), 
                        nn.AdaptiveAvgPool2d(output_size=(1))(x['pool'])), axis=1).squeeze()
    
    def forward(self, x):
        x = self.backbone(x)
        x_acc = x['pool']
#         x_fpn = x
        x_fpn = self.merge_orderdict(x)
#         x_fpn = nn.AdaptiveAvgPool2d(output_size=(1))(x_fpn).squeeze()

        x_acc = nn.AdaptiveAvgPool2d(output_size=(1))(x_acc).squeeze()
        x_acc = self.classifier(x_acc)        
        return x_fpn, x_acc

# backbone = resnet_fpn_backbone('resnet50', pretrained=True, trainable_layers=3)
# model = fpn_model(backbone)



'''from https://github.com/kuangliu/pytorch-fpn.git'''
'''FPN in PyTorch.

See the paper "Feature Pyramid Networks for Object Detection" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class feature_pyramid_network(nn.Module):
    def __init__(self, block, num_blocks):
        super(feature_pyramid_network, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Bottom-up layers
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        # Bottom-up
        c1 = self.conv1(x)
        c1 = self.bn1(c1)
        c1 = self.relu(c1)
        c1 = self.maxpool(c1)
        
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, F.relu(self.latlayer1(c4)))
        p3 = self._upsample_add(p4, F.relu(self.latlayer2(c3)))
        p2 = self._upsample_add(p3, F.relu(self.latlayer3(c2)))

        return p2, p3, p4, p5

class fpn_resnet50_classification(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(fpn_resnet50_classification, self).__init__()
        self.fpn = feature_pyramid_network(Bottleneck, [3,4,6,3])
        self.fpn.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.classifier = nn.Sequential(
                nn.BatchNorm1d(1024),
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(in_features = 1024, out_features=64, bias=False),
                nn.ReLU(inplace=True),
                
                nn.BatchNorm1d(64),
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(in_features=64, out_features=n_classes, bias=True)
                )

    def forward(self, x):
        feature_maps = self.fpn(x)
        preds = []
        for fm in feature_maps:
            fm = F.adaptive_avg_pool2d(fm, (1, 1))
            fm = fm.view(fm.size(0), -1)
            preds.append(fm)
            # print(fm.shape)
        preds = torch.cat(preds, dim=1)
        preds = self.classifier(preds)
        # print(preds.shape)
        return preds