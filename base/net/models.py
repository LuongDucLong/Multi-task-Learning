import torch
from torch import nn
import timm
from timm import create_model
from net.utils import ClassBlock


class Backbone(nn.Module):
    def __init__(self, class_num, model_name='nf_resnet50'):
        super(Backbone, self).__init__()
        self.model_name = model_name
        self.class_num = class_num

        model_ft = create_model(model_name=model_name, pretrained=True)
        
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.fc = nn.Sequential()
        self.features = model_ft
        self.num_ftrs = 1000
        
        for c in range(self.class_num):
            self.__setattr__('class_%d' % c, ClassBlock(input_dim=self.num_ftrs, class_num=1, activ='sigmoid') )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        pred_label = [self.__getattr__('class_%d' % c)(x) for c in range(self.class_num)]
        pred_label = torch.cat(pred_label, dim=1)
        return pred_label

# print(timm.list_models("*efficient*"))
# model_name = "nf_resnet50"
# num_label = 44
# model = Backbone(num_label, model_name)
# model = model.cuda()

# classifier_params = []
# for name, param in model.named_parameters():
#     if 'class_' in name:
#         classifier_params.append(param)
    
# print(classifier_params)
