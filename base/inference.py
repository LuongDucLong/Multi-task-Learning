import os
import json
import torch
import argparse
from PIL import Image
from torchvision import transforms as T
from net import get_model


######################################################################
# Settings
# ---------

transforms = T.Compose([
    T.Resize(size=(224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


######################################################################
# Argument
# ---------
parser = argparse.ArgumentParser()
parser.add_argument('image_path', help='Path to test image')
parser.add_argument('--backbone', default='swin_tiny_patch4_window7_224', type=str, help='model')
parser.add_argument('--weights', default='swin_tiny_patch4_window7_224', type=str, help='model')

args = parser.parse_args()

model_name = args.backbone
weight_path = args.weights

######################################################################
# Model and Data
# ---------
def load_network(network):
    save_path = os.path.join('./checkpoints', model_name, weight_path)
    network.load_state_dict(torch.load(save_path))
    print('Resume model from {}'.format(save_path))
    return network

def load_image(path):
    src = Image.open(path)
    if path.endswith("png"):
        src = src.convert("RGB")
    src = transforms(src)
    src = src.unsqueeze(dim=0)
    return src

num_label = 40
model = get_model(model_name, num_label)
model = load_network(model)
model.eval()

src = load_image(args.image_path)

######################################################################
# Inference
# ---------
class predict_decoder(object):

    def __init__(self, dataset):
        with open('./doc/label.json', 'r') as f:
            self.label_list = json.load(f)[dataset]
        with open('./doc/attribute.json', 'r') as f:
            self.attribute_dict = json.load(f)[dataset]
        self.dataset = dataset
        self.num_label = len(self.label_list)

    def decode(self, pred):
        pred = pred.squeeze(dim=0)
        for idx in range(self.num_label):
            name, chooce = self.attribute_dict[self.label_list[idx]]
            if chooce[pred[idx]]:
                print('{}: {}'.format(name, chooce[pred[idx]]))



out = model.forward(src)

pred = torch.gt(out, torch.ones_like(out) * 0.5 )  # threshold=0.5

Dec = predict_decoder("upar")
Dec.decode(pred)