import torch
from torchvision import models, transforms
import re
import clip
from pretrained_models.encoders import encoders, EncoderTuple, PreparedModel

# PRETRAINED_MODELS = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "RN50_clip", "RN101_clip", "RN50x4_clip", "RN50x16_clip", "ViT-B/32_clip", "ViT-B/16_clip", "ViT-L/14_clip",
#                          "RN50x64_clip", "ViT-B/32", "ViT-B/16", "tf_efficientnet_l2_ns_475", "deit_base_distilled_patch16_224", "resnet50_timm", "ssl_resnet50", "wrn", "resnetv2_50x1_bitm", 
#                          "resnetv2_50x1_bitm_in21k", "resnetv2_101x1_bitm_in21k", "resnetv2_101x3_bitm_in21k", "resnetv2_152x2_bitm_in21k", "resnetv2_152x4_bitm_in21k", 
#                          "resnetv2_50x1_bit_distilled", "resnetv2_152x2_bit_teacher", "resnetv2_152x2_bit_teacher_384", "dino_vits16", "dino_vits8", "dino_vitb16", "dino_vitb8", 
#                          "dino_vitb8_hf", "dino_resnet50", "swsl_resnext101_32x16d", "efficient_net_nosy_teacher", "efficient_net_nosy_teacher_b7", "efficient_net_nosy_teacher_b6"]


PRETRAINED_MODELS = ["resnet18", "resnet34", "resnet50", "RN50_clip", "ViT-B/16_clip", "ViT-B/16", "tf_efficientnet_l2_ns_475", "deit_base_distilled_patch16_224", "ssl_resnet50", "wrn", "resnetv2_50x1_bitm", 
                         "resnetv2_50x1_bitm_in21k", "resnetv2_50x1_bit_distilled", "resnetv2_152x2_bit_teacher", "dino_vits16", "dino_vitb16", "dino_resnet50", "swsl_resnext101_32x16d", "efficient_net_nosy_teacher"]


CLIP_MODEL_NAME_DICT = {
    'resnet_clip' : 'RN50',
    'vit_clip': 'ViT-B/16'
}

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
        self.init_weights()
    
    def init_weights(self):
        torch.nn.init.kaiming_normal_(self.weight)
        if self.bias is not None:
            self.bias = torch.nn.Parameter(torch.zeros(self.size_out))

    def forward(self, x, *args, **kwargs):
        # return torch.nn.functional.linear(x.float(), self.weight / torch.norm(self.weight, dim=1, keepdim=True), self.bias)
        return torch.nn.functional.linear(x, self.weight, self.bias)

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
        return self.out_dim

    def init_weights(self):
        self.class_means = torch.zeros((self.out_dim, self.in_dim), device=self.device)
        self.class_counts = torch.zeros(self.out_dim, device=self.device)

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
        x = x.detach()
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

def get_feature_extractor(model, device, pretrained):
    flatten_features = False
    if model == "resnet":
        full_model = models.resnet18(pretrained=pretrained)
        latent_dim = list(full_model.children())[-1].in_features
        feature_extractor = torch.nn.Sequential(*list(full_model.children())[:-1])
        flatten_features = True
    elif model == "densenet":
        full_model = models.densenet121(pretrained=pretrained)
        latent_dim = list(full_model.children())[-1].in_features
        features = list(full_model.children())[:-1]
        features.append(torch.nn.ReLU(inplace=True))
        features.append(torch.nn.AdaptiveAvgPool2d((1, 1)))
        feature_extractor = torch.nn.Sequential(*features)
        # feature_extractor = torch.nn.Sequential(
        #     *list(full_model.children())[:-1])
        flatten_features = True
    elif model == "vgg":
        full_model = models.vgg11(pretrained=pretrained)
        latent_dim = full_model.classifier[-1].in_features
        full_model.classifier = full_model.classifier[:-1]
        feature_extractor = full_model
        flatten_features = True
    elif model == "vit":
        full_model = models.vit_b_16(pretrained=pretrained)
        latent_dim = list(full_model.heads.children())[-1].in_features
        feature_extractor = full_model.encoder
        flatten_features = True
    elif model in PRETRAINED_MODELS:
        encoder_tuple = encoders[model]
        encoder = encoder_tuple.partial_encoder(device=device, input_shape=16,
                                                                    fix_batchnorms_encoder=True,
                                                                    width_factor=1,
                                                                    droprate=0.)

        transform = encoder.transformation_val

        if transform is None:
            transform = [transforms.Resize((224, 224)), 
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        
        feature_extractor = encoder.encoder
        feature_extractor.to(device)
        image = torch.Tensor(torch.ones(3, 224, 224)).unsqueeze(0).to(device)
        feature_extractor.eval()
        with torch.no_grad():
            latent_dim = feature_extractor(image).shape[1]
        flatten_features = True
        return feature_extractor, latent_dim, flatten_features, transform
    elif "clip" in model:
        if pretrained == False:
            raise ValueError("All clip models are already pretrained!")
        clip_model_name = CLIP_MODEL_NAME_DICT[model]
        feature_extractor, clip_transforms = clip.load(clip_model_name, device=device)
        image = clip_transforms(transforms.ToPILImage()(
            torch.Tensor(torch.ones(3, 224, 224)))).unsqueeze(0).to(device)
        feature_extractor.eval()
        with torch.no_grad():
            latent_dim = feature_extractor.encode_image(image).shape[1]
        flatten_features = True

    return feature_extractor, latent_dim, flatten_features

class TasksimModel(torch.nn.Module):
    def __init__(self, model, device, freeze_features=False, multihead=False, pretrained=True, nmc=False, no_masking=False):
        super().__init__()
        if model in PRETRAINED_MODELS:
            self.feature_extractor, self.fc_in_features, self.flatten_features, self.transform = get_feature_extractor(
            model, device, pretrained)
        else:
            self.feature_extractor, self.fc_in_features, self.flatten_features = get_feature_extractor(
            model, device, pretrained)
        self.is_clip = True if "clip" in model else False
        self.classifier = None
        self.head_size = 0
        self.device = device
        self.nmc = nmc
        self.old_head_weights, self.old_head_bias = None, None
        self.multihead = multihead
        self.frozen_features = freeze_features
        self.set_task2vec_mode(False)
        self.no_masking = no_masking

        if self.multihead:
            if freeze_features:
                raise ValueError("Don't use frozen model with multihead setup")
            self.heads = torch.nn.ModuleList()

        if self.frozen_features:
            self.freeze_features()
        
        if self.nmc:
            self.classifier = NMC_Classifier(self.fc_in_features, self.device)
            self.classifier.to(self.device)

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
            self.head_size = self.classifier.extend_head(n)
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
    
    def encode_features(self, x) -> torch.Tensor:
        if self.is_clip:
            out = self.feature_extractor.encode_image(x)
        else:
            out = self.feature_extractor(x)
        if self.flatten_features:
            out = torch.flatten(out, 1)
        return out

    def __forward(self, x):
        if self.frozen_features and self.training and not self.task2vec:
            return self.classifier(x)
        else:
            features = self.encode_features(x)
            if self.is_clip:
                features = features.float()
            return self.classifier(features)

    def forward(self, x, y=None):
        if self.nmc or self.multihead or self.task2vec or not self.training or self.no_masking:
            return self.__forward(x)
        ## Apply masking
        assert y is not None
        outs = self.__forward(x)
        classes_mask = torch.eye(self.head_size).to(self.device).float()
        label_unique = y.unique()
        ind_mask = classes_mask[label_unique].sum(0)
        full_mask = ind_mask.unsqueeze(0).repeat(outs.shape[0], 1)
        outs = torch.mul(outs, full_mask)
        return outs
        
    def set_task2vec_mode(self, mode=False):
        self.task2vec = mode

def get_optimizer_lr_scheduler(optim, optim_params, lr, momentum):
    if optim == 'adam':
        optim = torch.optim.Adam(optim_params, lr=lr, weight_decay=2e-4)
    else:
        optim = torch.optim.SGD(optim_params, lr=lr,momentum=momentum, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optim, step_size=7, gamma=0.1)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optim, T_max=args.num_epochs)
    return optim, lr_scheduler
