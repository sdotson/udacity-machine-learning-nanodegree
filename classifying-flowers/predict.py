# third party imports
import argparse
import json
import torch

# local imports
from model import predict
from utils import determine_device
from validation import validate_predict_args

# CLI defaults
TOP_K_DEFAULT = 1

# configure argument parser
parser = argparse.ArgumentParser(description="Trains model and saves checkpoint")
parser.add_argument("image_path", help="the path for the image you wish to classify")
parser.add_argument("checkpoint", help="the model checkpoint you would like to use")
parser.add_argument("--category_names")
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--top_k", type=int, default=TOP_K_DEFAULT)

# parse and validate args
args = parser.parse_args()
validate_predict_args(args)

# Getting category to name mapping
cat_to_name = None
if args.category_names:
    with open(args.category_names, "r") as f:
        cat_to_name = json.load(f)

# use gpu if available and requested in args
device = determine_device(args.gpu)
print("Using device {}...".format(device.type))

print("Loading checkpoint...")
# Below is a solution for loading checkpoint saved on a gpu device and I believe vice versa
# https://discuss.pytorch.org/t/on-a-cpu-device-how-to-load-checkpoint-saved-on-gpu-device/349
checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
model = checkpoint["pretrained_model"]
model.classifier = checkpoint["classifier"]
model.load_state_dict(checkpoint["state_dict"])
model.class_to_idx = checkpoint["class_to_idx"]
model.to(device)

print("Predicting class for image...")
chart_data = predict(args.image_path, model, device, cat_to_name, args.top_k)

print("Printing chart of classes and probabilities...")
print(chart_data)
