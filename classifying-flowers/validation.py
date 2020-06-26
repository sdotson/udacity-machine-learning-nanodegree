from os import path
import torch
from torchvision import models

# validates train.py args
def validate_train_args(args):
    # check cuda
    if args.gpu and torch.cuda.is_available() == False:
        # we don't want to throw sand in the user's face
        # but let them know we are falling back to CPU
        print("GPU is not enabled for this device, falling back to CPU")

    # check arch
    if hasattr(models, args.arch) == False:
        # not perfect, since there are a lot of attrs that aren't archs
        raise ValueError(
            "{} is not a valid architecture. Please remove flag or choose a valid architecture in torchvision.models".format(
                args.arch
            )
        )

    # check data_directory existance
    if path.exists(args.data_directory) == False:
        raise ValueError(
            "data directory does not exist: {}".format(args.data_directory)
        )

    # check save_dir existance
    if args.save_dir and path.exists(args.save_dir) == False:
        raise ValueError("save directory does not exist: {}".format(args.save_dir))


# validates predict.py args
def validate_predict_args(args):
    # check cuda
    if args.gpu and torch.cuda.is_available() == False:
        # we don't want to throw sand in the user's face
        # but let them know we are falling back to CPU
        print("GPU is not enabled for this device, falling back to CPU")

    # check data_directory existance
    if path.exists(args.image_path) == False:
        raise ValueError("image path does not exist: {}".format(args.image_path))

    # check checkpoint existance
    if path.exists(args.checkpoint) == False:
        raise ValueError("checkpoint does not exist: {}".format(args.checkpoint))

    # check category names existance
    if args.category_names and path.exists(args.category_names) == False:
        raise ValueError(
            "category names does not exist: {}".format(args.category_names)
        )
