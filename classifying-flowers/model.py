import argparse
from collections import OrderedDict
from torchvision import datasets, models, transforms
import torch
from torch import nn, optim
from PIL import Image
import numpy as np
import pandas as pd
import time


def create_dataloaders(data_directory, batch_size):
    """Creates dataloaders for training, validation, and test data"""
    means = [0.485, 0.456, 0.406]
    std_deviations = [0.229, 0.224, 0.225]
    image_size = 224
    rotation = 30

    train_dir = data_directory + "/train"
    valid_dir = data_directory + "/valid"
    test_dir = data_directory + "/test"

    train_transform = transforms.Compose(
        [
            transforms.RandomRotation(rotation),
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(means, std_deviations),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(means, std_deviations),
        ]
    )

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=test_transform)

    dataloaders = {
        "train": torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        ),
        "test": torch.utils.data.DataLoader(test_dataset, batch_size=batch_size),
        "valid": torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size),
    }

    class_to_idx = train_dataset.class_to_idx

    return dataloaders, class_to_idx


def create_model(arch, hidden_units_size, output_units_size, device):
    """Creates pre-trained model with custom classifier for given architecture"""
    model = getattr(models, arch)(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    # define new classifier
    input_size = model.classifier[0].in_features
    classifier = nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(input_size, hidden_units_size)),
                ("relu", nn.ReLU()),
                ("dropout", nn.Dropout(p=0.5)),
                ("fc2", nn.Linear(hidden_units_size, output_units_size)),
                ("output", nn.LogSoftmax(dim=1)),
            ]
        )
    )

    model.classifier = classifier
    model.to(device)

    return model, input_size


def train_model(model, dataloaders, epochs, learning_rate, device):
    """Trains model and periodically logs validation stats"""
    images_trained = 0
    print_every = 5
    running_loss = 0

    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    train_start = time.time()
    for epoch in range(epochs):
        model.train()

        for inputs, labels in dataloaders["train"]:
            images_trained += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if images_trained % print_every == 0:
                model.eval()
                model.to(device)

                with torch.no_grad():
                    accuracy = 0
                    validation_loss = 0
                    for images, labels in dataloaders["valid"]:
                        images, labels = images.to(device), labels.to(device)

                        logps = model.forward(images)
                        validation_loss += criterion(logps, labels).item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(
                    f"Epoch {epoch+1}/{epochs} (image {images_trained}).. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Validation loss: {validation_loss/len(dataloaders['valid']):.3f}.. "
                    f"validation accuracy: {accuracy/len(dataloaders['valid']):.3f}"
                )

                running_loss = 0
                model.train()
    print("Training completed in {} seconds".format(time.time() - train_start))


def process_image(image_path):
    """ Scales, crops, and normalizes a PIL image for a PyTorch model,
      returns a Torch tensor
  """
    with Image.open(image_path) as image:
        shortest_side_length = 256
        is_width_bigger = image.size[0] > image.size[1]
        new_size = (
            [image.size[0], shortest_side_length]
            if is_width_bigger
            else [shortest_side_length, image.size[1]]
        )

        # return image with new size
        resized_image = image.resize(new_size)
        width, height = resized_image.size

        # determine center crop bounding box
        crop_size = 224
        left = (width - crop_size) / 2
        upper = (height - crop_size) / 2
        right = (width + crop_size) / 2
        lower = (height + crop_size) / 2

        # crop the image
        cropped_image = resized_image.crop((left, upper, right, lower))

        # transform to numpy array
        np_image = np.array(cropped_image)

        # squish and normalize
        np_image_squished = np_image / 255
        means = np.array([0.485, 0.456, 0.406])
        std_deviations = np.array([0.229, 0.224, 0.229])
        normalized_image = (np_image_squished - means) / std_deviations

        # we need to change order of dimensions to meet pytorch's expectations
        transposed_image = np.transpose(normalized_image, (2, 0, 1))
        return torch.from_numpy(transposed_image)


def predict(image_path, model, device, cat_to_name, top_k):
    """ Predict the class (or classes) of an image using a trained deep learning model.
  """
    predict_start = time.time()
    model.to(device)

    processed_image = process_image(image_path)

    # needs to be a float or computer gets angry with me
    image_float = processed_image.float().unsqueeze(0)

    # run image through model
    model.eval()
    model_output = model.forward(image_float.to(device))
    predictions = torch.exp(model_output)

    # top predictions and top labels
    top_preds, top_labels = predictions.topk(top_k)

    # need to detach in order to call numpy
    top_preds = top_preds.detach()

    if device.type != "cpu":
        top_preds = top_preds.cpu()

    top_preds = top_preds.numpy().tolist()
    top_labels = top_labels.tolist()

    data = {"class": pd.Series(model.class_to_idx)}

    # if there is cat_to_name translation dict around, we can add the flower_name column
    if cat_to_name:
        data["flower_name"] = pd.Series(cat_to_name)

    chart_data = pd.DataFrame(data)
    chart_data = chart_data.set_index("class")
    chart_data = chart_data.iloc[top_labels[0]]
    chart_data["probabilities"] = top_preds[0]

    print(
        "Processing and prediction completed in {} seconds".format(
            time.time() - predict_start
        )
    )

    return chart_data
