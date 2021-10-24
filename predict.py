import argparse
import json
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
from MyClassifier import myNNmodule


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
    else:
        print("Unsupported pretrained {}".format(arch))
        exit()
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    return model, model_out_features


def assign_classifier_to_model(classifier, model, arch):
    print('assign_classifier_to_model() ==> {0}'.format(arch))
    if arch == 'vgg16' or arch == 'vgg13':
        model.classifier = classifier
        return True
    elif arch == 'resnet18':
        model.fc = classifier
        return True
    else:
        print('Unsupported pretrained {}'.format(arch))
        exit()


def load_checkpoint(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path)
        model, model_out_features = load_pretrained_NN(checkpoint['arch'])
        myNN = myNNmodule()
        myNN.load_checkpoint_type1(checkpoint)
        assign_classifier_to_model(myNN.classifier, model, checkpoint['arch'])
        return model, myNN
    except Exception as e:
        print(e)
        exit()


def resizeKeepingAspect(PILimage, shortest_side):
    width = PILimage.width
    height = PILimage.height
    aspect_ratio = width / height
    if (aspect_ratio > 1.0):
        new_height = shortest_side
        new_width = round(aspect_ratio * new_height)
    elif (aspect_ratio < 1.0):
        new_width = shortest_side
        new_height = round(new_width / aspect_ratio)
    else:
        new_width = shortest_side
        new_height = shortest_side

    return new_width, new_height


def cropCenterBox(PILimage, width, height):
    image_width, image_height = PILimage.size
    left = (image_width - width) / 2
    upper = (image_height - height) / 2
    right = left + width
    lower = upper + height
    # print("{0}".format((left, upper, right, lower)))
    cropped = PILimage.crop((left, upper, right, lower))
    return cropped


def normalizePILimageAsNdarray(PILimage, mean, std_dev):
    np_image = np.array(PILimage)
    np_image_norm = np.zeros_like(np_image)
    if (np_image.dtype == 'uint8'):
        np_image_norm = np_image / 255
    np_image_norm = (np_image_norm - mean) / std_dev
    return np_image_norm


def ImageAsNdarrayToTensor(npImage):
    # Image as Ndarray: (row, column, pixel) ==> Pytorch (pixel, row, colum)
    return torch.FloatTensor(npImage.transpose((2, 0, 1)))


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    new_size = resizeKeepingAspect(image, 256)
    resized_image = image.resize(new_size, resample=Image.BICUBIC)
    cropped_image = cropCenterBox(resized_image, 224, 224)

    mean = np.array([0.485, 0.456, 0.406])
    std_dev = np.array([0.229, 0.224, 0.225])
    np_image_norm = normalizePILimageAsNdarray(cropped_image, mean, std_dev)
    return ImageAsNdarrayToTensor(np_image_norm)


def reverseKeyValuePair(dictToReverse):
    reversed_dict = dict()
    for key, value in dictToReverse.items():
        reversed_dict[value] = key
    return reversed_dict


def predict(image, model, device, topk=5):
    ''' Predict the class (or classes) of an image
        using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file
    tensor_image = process_image(image)
    tensor_image = tensor_image.unsqueeze(0)
    dev_tensor_image = tensor_image.to(device)

    model.to(device)
    model.eval()
    with torch.no_grad():
        output = model.forward(dev_tensor_image)
        ps = torch.exp(output)

        top_p, top_class = ps.topk(topk, dim=1)
    model.train()
    return top_p, top_class


def print_category_probs(lst_cat, lst_probs):
    print_lst = list(zip(lst_cat, lst_probs))
    print("\n[Item] - [Score]")
    for item in print_lst:
        print("{} - {:.2f}%".format(item[0], item[1] * 100))


def main(dict_args):
    checkpoint_path = dict_args['checkpoint']
    image_path = dict_args['image_path']
    top_k = dict_args['top_k']
    loaded_model, myNN = load_checkpoint(checkpoint_path)
    if loaded_model is None:
        print('No model loaded')
        exit()

    device_cuda0 = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')
    device_cpu = torch.device('cpu')
    chosen_device = device_cpu
    if dict_args['gpu'] is True:
        print("Running on GPU")
        chosen_device = device_cuda0
    else:
        print("Running on CPU")

    try:
        image = Image.open(image_path)
    except Exception as e:
        print(e)
        exit()

    probs, indices = predict(image, loaded_model, chosen_device, top_k)

    idx_to_class = reverseKeyValuePair(myNN.class_to_idx)
    probs = probs.to(device_cpu).squeeze().numpy()
    indices = indices.to(device_cpu).squeeze().numpy()

    flowers = []
    flowers_ps = []
    if dict_args['category_names'] is not None:
        cat_to_name = {}
        try:
            with open(dict_args['category_names'], 'r') as f:
                cat_to_name = json.load(f)
            if len(cat_to_name) > 0:
                if top_k > 1:
                    flowers = [cat_to_name[idx_to_class[idx]]
                               for idx in indices.tolist()]
                    flowers_ps = probs.tolist()
                else:
                    flowers.append(cat_to_name[idx_to_class[indices.item(0)]])
                    flowers_ps = [probs.item(0)]
                print_category_probs(flowers, flowers_ps)
                return
        except Exception as e:
            print(e)

    if top_k > 1:
        flowers = [idx_to_class[idx]
                   for idx in indices.tolist()]
        flowers_ps = probs.tolist()
    else:
        flowers.append(idx_to_class[indices.item(0)])
        flowers_ps = [probs.item(0)]

    print_category_probs(flowers, flowers_ps)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description='Predict app')
    argparser.add_argument('image_path', type=str)
    argparser.add_argument('checkpoint', type=str)
    argparser.add_argument('--category_names', type=str)
    argparser.add_argument('--top_k', type=int, default=5)
    argparser.add_argument('--gpu', action='store_true')
    args = argparser.parse_args()
    print(args)
    dict_args = vars(args)
    main(dict_args)
