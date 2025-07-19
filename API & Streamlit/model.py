import torch.nn as nn
import torchvision.models as models

class MultiOutputModel(nn.Module):
    def __init__(self, n_colours, n_product_types, n_seasons, n_genders):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()  # type: ignore
        self.fc_colour = nn.Linear(2048, n_colours)
        self.fc_product_type = nn.Linear(2048, n_product_types)
        self.fc_season = nn.Linear(2048, n_seasons)
        self.fc_gender = nn.Linear(2048, n_genders)
    def forward(self, x):
        feat = self.backbone(x)
        return {
            'colour': self.fc_colour(feat),
            'product_type': self.fc_product_type(feat),
            'season': self.fc_season(feat),
            'gender': self.fc_gender(feat)
        } 