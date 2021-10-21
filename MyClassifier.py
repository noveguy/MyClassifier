from torch import nn
from collections import OrderedDict

class myNNmodule():
    def __init__(self):
        self.classifier = nn.Sequential()
        self.class_to_idx = dict()
        print("Done init classifier")

    def create_Classifier_Type1(self, model_out_features, hidden_units):
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(model_out_features, hidden_units)),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        return classifier
