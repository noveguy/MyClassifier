from torch import nn
from collections import OrderedDict


class myNNmodule():
    def __init__(self):
        self.classifier = nn.Sequential()
        self.class_to_idx = dict()
        print("Done init classifier")

    def apply_classifier_type1(self, model_out_features, hidden_units):
        type1 = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(model_out_features, hidden_units)),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        self.classifier = type1

    def save_checkpoint_type1(self):
        type1_checkpoint = {
            'layers': [
                (each.in_features, each.out_features)
                for each in self.classifier
                if isinstance(each, nn.Linear)],
            'state_dict': self.classifier.state_dict(),
        }
        return type1_checkpoint

    def load_checkpoint_type1(self, checkpoint):
        type1 = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(checkpoint['layers']
                              [0][0], checkpoint['layers'][0][1])),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear(
                checkpoint['layers'][1][0], checkpoint['layers'][1][1])),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        self.classifier = type1
        self.classifier.load_state_dict(checkpoint['state_dict'])
        self.class_to_idx = checkpoint['class_to_idx']
