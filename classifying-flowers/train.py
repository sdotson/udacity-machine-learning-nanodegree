# third party imports
import argparse
import os
import torch
from torchvision import models

# local imports
from model import create_dataloaders, create_model, train_model
from utils import determine_device
from validation import validate_train_args

# CLI defaults
HIDDEN_UNITS_DEFAULT = 2048
ARCH_DEFAULT = "vgg16"
LEARNING_RATE_DEFAULT = 0.001
EPOCHS_DEFAULT = 8

# other settings
BATCH_SIZE = 60

# configure argument parser
parser = argparse.ArgumentParser(description="Trains model and saves checkpoint")
parser.add_argument("data_directory", help="the directory for the training data")
parser.add_argument("--arch", default=ARCH_DEFAULT)
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE_DEFAULT)
parser.add_argument("--save_dir")
parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
parser.add_argument("--hidden_units", type=int, default=HIDDEN_UNITS_DEFAULT)

# parse CLI args
args = parser.parse_args()

# do some additional validation on args
validate_train_args(args)

# get dataloaders and class_to_idx map
print("Creating dataloaders...")
dataloaders, class_to_idx = create_dataloaders(args.data_directory, BATCH_SIZE)

# use gpu if available and requested in args
device = determine_device(args.gpu)
print("Using device {}...".format(device.type))

print("Creating model...")
training_directory = args.data_directory + "/train/"
output_units_size = sum(
    [os.path.isdir(training_directory + i) for i in os.listdir(training_directory)]
)
model, input_size = create_model(
    args.arch, args.hidden_units, output_units_size, device
)

# train the model in place
print("Training model...")
train_model(model, dataloaders, args.epochs, args.learning_rate, device)

# save checkpoint
print("Saving checkpoint...")
checkpoint = {
    "batch_size": BATCH_SIZE,
    "class_to_idx": class_to_idx,
    "classifier": model.classifier,
    "input_size": input_size,
    "pretrained_model": getattr(models, args.arch)(pretrained=True),
    "output_size": output_units_size,
    "state_dict": model.state_dict(),
}
save_path = args.save_dir + "/checkpoint.pth" if args.save_dir else "checkpoint.pth"
torch.save(checkpoint, save_path)
print("Done. Checkpoint has been saved at {}".format(save_path))
