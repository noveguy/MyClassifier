from MyClassifier import myNNmodule
import argparse
import json
from os import path
from collections import OrderedDict
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from PIL import Image
import numpy as np
from torchvision import datasets, transforms, models


def load_data_for_training(data_dir):
    train_dir = data_dir + '/train'
    train_transforms = transforms.Compose(
        [transforms.Resize(255),
         transforms.RandomRotation(
            degrees=(0, 20),
            interpolation=transforms.functional.InterpolationMode.BICUBIC),
         transforms.RandomCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])
         ])
    train_datasets = datasets.ImageFolder(
        train_dir, transform=train_transforms)
    trainloaders = torch.utils.data.DataLoader(
        train_datasets, batch_size=64, shuffle=True)
    return trainloaders, train_datasets


def load_data_for_validating(data_dir):
    valid_dir = data_dir + '/valid'
    valid_transforms = transforms.Compose(
        [transforms.Resize(255),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])
         ])
    valid_datasets = datasets.ImageFolder(
        valid_dir, transform=valid_transforms)
    validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=64)
    return validloaders, valid_datasets


def load_data_for_testing(data_dir):
    test_dir = data_dir + '/test'
    test_transforms = transforms.Compose(
        [transforms.Resize(255),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])])
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
    testloaders = torch.utils.data.DataLoader(test_datasets, batch_size=64)
    return testloaders, test_datasets


def get_cat_to_name(json_file):
    with open(json_file, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name


def test_loop():
    pass


def load_pretrained_NN(arch):
    model_out_features = 0
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        model_out_features = 25088
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
        model_out_features = 25088
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        model_out_features = 512
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    return model, model_out_features


def assign_classifier_to_model(classifier, model, arch):
    print('assign_classifier_to_model() ==> {0}'.format(arch))
    if arch == 'vgg16' or arch == 'vgg13':
        model.classifier = classifier
    elif arch == 'resnet18':
        model.fc = classifier
    else:
        print('Unsupported arch')


def get_model_classifier(model, arch):
    if arch == 'vgg16':
        return model.classifier
    elif arch == 'vgg13':
        return model.classifier
    elif arch == 'resnet18':
        return model.fc
    else:
        return None


def train_classifier(model, model_classifier,
                     epochs, leanrate, dataloader, validloader):
    device_cuda0 = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')
    device_cpu = torch.device('cpu')
    # Push NN model to GPU
    model.to(device_cuda0)
    # training backpropagation
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model_classifier.parameters(), lr=leanrate)
    # turn on gradient for classifier
    for params in model_classifier.parameters():
        params.requires_grad = True
    # training loop
    print("Enter training loop")
    for e in range(epochs):
        running_loss = 0
        for images, labels in dataloader:
            optimizer.zero_grad()
            cuda_labels = labels.to(device_cuda0)
            cuda_images = images.to(device_cuda0)

            outputs = model.forward(cuda_images)
            loss = criterion(outputs, cuda_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            total_val_loss = 0
            val_correct = 0
            # turn off gradients for validation
            with torch.no_grad():
                for images, labels in validloader:
                    cuda_val_labels = labels.to(device_cuda0)
                    cuda_val_images = images.to(device_cuda0)
                    # pass images through network
                    cuda_val_outputs = model.forward(cuda_val_images)
                    cuda_val_loss = criterion(
                        cuda_val_outputs, cuda_val_labels)
                    # accumulate total loss of validation set
                    val_loss = cuda_val_loss.to(device_cpu)
                    total_val_loss += val_loss.item()
                    # convert from Log-probabilities to normal probabilities
                    val_outputs = cuda_val_outputs.to(device_cpu)
                    val_ps = torch.exp(val_outputs)
                    # Extract class that has highest probability
                    top_p, top_class = val_ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    val_correct += equals.sum().item()
        # training / validation loss of this epoch
        train_loss = running_loss / len(dataloader.dataset)
        valid_loss = total_val_loss / len(validloader.dataset)
        print("Epoch: {}/{}.. ".format(e + 1, epochs),
              "Train Loss: {:.6f}.. ".format(train_loss),
              "Valid Loss: {:.6f}.. ".format(valid_loss),
              "Valid Accuracy: {:.6f}".format(
            val_correct / len(validloader.dataset))
        )
    print("Done training")


def get_arch_input_size(arch):
    input_size = 0
    if arch == 'vgg16' or arch == 'vgg13':
        input_size = 25088
    elif arch == 'resnet18':
        input_size = 512
    else:
        print("Unsupported arch")
    return input_size


def save_checkpoint(save_dir, model_classifier, arch, epochs, hidden_units):
    checkpoint = {'input_size': get_arch_input_size(arch),
                  'output_size': 102,
                  'layers': [
                      (each.in_features, each.out_features)
                      for each in model_classifier
                      if isinstance(each, nn.Linear)],
                  'state_dict': model_classifier.state_dict(),
                  'class_to_idx': train_datasets.class_to_idx}
    save_path = str.format(
        "{0}\\{1}_{2}_{3}.pth",
        path.abspath(save_dir), arch, hidden_units, epochs)
    print(save_path)
    torch.save(checkpoint, save_path)
    print(checkpoint['layers'])
    print(checkpoint['layers'][0][0])
    print(checkpoint['layers'][0][1])


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description='Training app')
    argparser.add_argument('data_dir', type=str)
    argparser.add_argument('--save_dir', default='.')
    argparser.add_argument('--arch', type=str, default='vgg16')
    argparser.add_argument('--learning_rate', type=float, default=0.001)
    argparser.add_argument('--hidden_units', type=int, default=512)
    argparser.add_argument('--epochs', type=int, default=10)
    argparser.add_argument('--gpu', action='store_true')
    args = argparser.parse_args()
    print(args)
    dict_args = vars(args)
    # Load data
    train_dataloader, train_datasets = load_data_for_training(
        dict_args['data_dir'])
    print(len(train_dataloader.dataset))
    valid_dataloader, valid_datasets = load_data_for_validating(
        dict_args['data_dir'])
    print(len(valid_dataloader.dataset))
    test_dataloader, test_datasets = load_data_for_testing(
        dict_args['data_dir'])
    print(len(test_dataloader.dataset))
    # Load model
    model, model_out_features = load_pretrained_NN(dict_args['arch'])
    hidden_units = dict_args['hidden_units']
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(model_out_features, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    myModule = myNNmodule()
    myModule.classifier = myModule.create_Classifier_Type1(
        model_out_features, hidden_units)
    # myModule.classifier = classifier
    assign_classifier_to_model(myModule.classifier, model, dict_args['arch'])
    # train the classifier
    model_classifier = get_model_classifier(model, dict_args['arch'])
    train_classifier(
        model, model_classifier,
        dict_args['epochs'], dict_args['learning_rate'],
        train_dataloader, valid_dataloader)
    # save the model
    # save_checkpoint(dict_args['save_dir'], model_classifier, dict_args['arch'],
    #                 dict_args['epochs'], dict_args['hidden_units'])
    print("Hello World!")
