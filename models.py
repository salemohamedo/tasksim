import torch
from torchvision import models
import re

class WeightNorm_Classifier(torch.nn.Module):
    def __init__(self, in_dim, n_classes, bias=False):
        super().__init__()
        self.size_in, self.size_out = in_dim, n_classes
        self.weight = torch.nn.Parameter(torch.Tensor(n_classes, in_dim))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(n_classes))
        else:
            self.bias = None

        # initialize weights
        torch.nn.init.kaiming_normal_(self.weight)  # weight init

    def forward(self, x, *args, **kwargs):
        return torch.nn.functional.linear(x, self.weight / torch.norm(self.weight, dim=1, keepdim=True), self.bias)
        # return torch.nn.functional.linear(x, self.weight, self.bias)

class NMC_Classifier(torch.nn.Module):
    def __init__(self, in_dim, device):
        super().__init__()
        self.device = device
        self.in_dim = in_dim
        self.out_dim = 0
        self.class_means = None
        self.class_counts = None
        self.initiated = False

    def extend_head(self, n):
        if self.out_dim == 0: ## First time
            self.class_means = torch.zeros((n, self.in_dim), device=self.device)
            self.class_counts = torch.zeros(n, device=self.device)
        else:
            self.class_means = torch.cat((self.class_means, torch.zeros(n, self.in_dim, device=self.device)))
            self.class_counts = torch.cat((self.class_counts, torch.zeros(n, device=self.device)))
        self.out_dim += n

    def update_means(self, y, epoch):
        if epoch > 0: ## Only need to update once
            return
        for i in range(self.out_dim):
            indexes = torch.where(y == i)[0]
            self.class_means[i] = (
                self.class_means[i] * (1.0 * self.class_counts[i]) + self.x[indexes].sum(0))
            self.class_counts[i] += len(indexes)
            if self.class_counts[i] != 0:
                self.class_means[i] = self.class_means[i] / (1.0 * self.class_counts[i])
        self.initiated = True
        

    def forward(self, x):
        self.x = x
        if self.initiated:
            out = torch.cdist(x, self.class_means)
            # convert smaller is better into bigger in better
            out = out * -1
            return out
        else:
            # if mean are not initiate we return random predition
            return torch.randn((x.shape[0], self.out_dim)).to(self.device)

## Configure model

def get_feature_extractor(model):
    flatten_features = False
    if model == "resnet":
        full_model = models.resnet34(pretrained=True)
        latent_dim = list(full_model.children())[-1].in_features
        feature_extractor = torch.nn.Sequential(*list(full_model.children())[:-1])
        flatten_features = True
    elif model == "densenet":
        full_model = models.densenet121(pretrained=True)
        latent_dim = list(full_model.children())[-1].in_features
        features = list(full_model.children())[:-1]
        features.append(torch.nn.ReLU(inplace=True))
        features.append(torch.nn.AdaptiveAvgPool2d((1, 1)))
        feature_extractor = torch.nn.Sequential(*features)
        # feature_extractor = torch.nn.Sequential(
        #     *list(full_model.children())[:-1])
        flatten_features = True
    elif model == "vgg":
        full_model = models.vgg16(pretrained=True)
        latent_dim = full_model.classifier[-1].in_features
        full_model.classifier = full_model.classifier[:-1]
        feature_extractor = full_model
        flatten_features = True
    return feature_extractor, latent_dim, flatten_features

class PretrainedModel(torch.nn.Module):
    def __init__(self, model, device, freeze_features=False, multihead=False):
        super().__init__()
        self.feature_extractor, self.fc_in_features, self.flatten_features = get_feature_extractor(model)
        self.classifier = None
        self.head_size = 0
        self.device = device
        self.old_head_weights, self.old_head_bias = None, None
        self.nmc = False
        self.multihead = multihead
        self.frozen_features = freeze_features
        self.set_task2vec_mode(False)

        if self.multihead:
            if freeze_features:
                raise ValueError("Don't use frozen model with multihead setup")
            self.heads = torch.nn.ModuleList()

        if freeze_features:
            self.freeze_features()

    def add_head(self, head_size):
        self.heads.append(torch.nn.Linear(self.fc_in_features, head_size, device=self.device))
        # self.heads.append(WeightNorm_Classifier(self.fc_in_features, head_size, bias=True))
        # self.heads[-1] = self.heads[-1].to(self.device)
    
    def set_head(self, i):
        self.classifier = self.heads[i]

    def add_and_set_head(self, head_size):
        self.add_head(head_size)
        self.set_head(-1)

    def extend_head(self, n):
        if self.nmc == True:
            self.classifier.extend_head(n)
            return
        new_head_size = self.head_size + n
        # new_head = torch.nn.Linear(self.fc_in_features, new_head_size, bias=False)
        new_head = WeightNorm_Classifier(
            self.fc_in_features, new_head_size, bias=True)
        if self.head_size != 0:  # Save old class weights
            self.old_head_weights = self.classifier.weight.data.clone().to(self.device)
            self.old_head_bias = self.classifier.bias.data.clone().to(self.device)
            new_head.weight.data[:self.head_size] = self.classifier.weight.data.clone().to(
                self.device)
            new_head.bias.data[:self.head_size] = self.classifier.bias.data.clone(
            ).to(self.device)
        self.classifier = new_head
        self.head_size = new_head_size
        self.classifier.to(self.device)
        # self.encoder.to(self.device)

    def unfreeze_features(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = True

    def freeze_features(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def encode_features(self, x):
        out = self.feature_extractor(x)
        if self.flatten_features:
            out = torch.flatten(out, 1)
        return out

    def __forward(self, x):
        if self.frozen_features and self.training and not self.task2vec and not self.nmc:
            return self.classifier(x)
        else:
            features = self.encode_features(x)
            return self.classifier(features)

    def forward(self, x, y):
        if self.nmc or self.multihead:
            return self.__forward(x)
        ## Apply masking
        outs = self.__forward(x)
        classes_mask = torch.eye(self.head_size).cuda().float()
        label_unique = y.unique()
        ind_mask = classes_mask[label_unique].sum(0)
        full_mask = ind_mask.unsqueeze(0).repeat(outs.shape[0], 1)
        outs = torch.mul(outs, full_mask)
        return outs
    
    def configure_nmc(self):
        print(self.device)
        self.nmc = True
        self.classifier = NMC_Classifier(self.fc_in_features, self.device)
        self.classifier.to(self.device)
    
    def set_task2vec_mode(self, mode=False):
        self.task2vec = mode

def get_optimizer_lr_scheduler(optim, optim_params, lr):
    if optim == 'adam':
        # optim = torch.optim.Adam(optim_params, lr=3e-4, weight_decay=5e-4)
        optim = torch.optim.Adam(optim_params, lr=lr, weight_decay=5e-4)
    else:
        optim = torch.optim.SGD(optim_params, lr=lr,
                                    momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optim, step_size=7, gamma=0.1)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optim, T_max=args.num_epochs)
    return optim, lr_scheduler
