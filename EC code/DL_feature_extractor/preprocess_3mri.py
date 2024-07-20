import time

import cv2
import h5py
import numpy as np
import openslide
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import numpy as np
from monai.transforms import (
    Compose,
    ToTensor,
)


class MRIDataset(Dataset):
    def __init__(self, image_name, data_folder):
        self.image_name = image_name
        self.data_folder = data_folder

        self.transforms = Compose([ToTensor()])

    def __len__(self):
        return len(self.image_name)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_folder, self.image_name[idx])
        img = nifty_loader(image_path)

        img = self.transforms(img)
        return img, self.image_name[idx]


def load_encoder(backbone, checkpoint_file, use_imagenet_weights, device):
    import torch.nn as nn
    import torchvision.models as models

    class DecapitatedResnet(nn.Module):
        def __init__(self, base_encoder, pretrained):
            super(DecapitatedResnet, self).__init__()
            self.encoder = base_encoder(num_classes=128, pretrained=pretrained)

        def forward(self, x):
            # Same forward pass function as used in the torchvision 'stock' ResNet code
            # but with the final FC layer removed.
            x = self.encoder.conv1(x)
            x = self.encoder.bn1(x)
            x = self.encoder.relu(x)
            x = self.encoder.maxpool(x)

            x = self.encoder.layer1(x)
            x = self.encoder.layer2(x)
            x = self.encoder.layer3(x)
            x = self.encoder.layer4(x)

            x = self.encoder.avgpool(x)
            x = torch.flatten(x, 1)

            return x

    model = DecapitatedResnet(models.__dict__[backbone], use_imagenet_weights)

    if use_imagenet_weights:
        if checkpoint_file is not None:
            raise Exception(
                "Either provide a weights checkpoint or the --imagenet flag, not both."
            )
        print(f"Created encoder with Imagenet weights")
    else:
        checkpoint = torch.load(checkpoint_file, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith("encoder_q") and not k.startswith(
                "encoder_q.fc"
            ):
                # remove prefix from key names
                state_dict[k[len("encoder_q.") :]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        # Verify that the checkpoint did not contain data for the final FC layer
        msg = model.encoder.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        print(f"Loaded checkpoint {checkpoint_file}")

    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    return model


def extract_features(model, device, train_loader):
    total_feature = []
    total_name = []
    with torch.no_grad():
        for batch, names in train_loader:
            print(names)
            batch = batch.float().to(device, non_blocking=True)
            features = model(batch).cpu().numpy().tolist()
            total_feature += features
            total_name += names
    return total_feature, total_name


def nifty_loader(path: str):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Preprocessing script")
    parser.add_argument(
        "--input_data",
        type=str,
        default=r'E:\study\EC\subtype\Data_slice_3mri\data2',
        help="Path to input nii folder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r'E:\study\EC\subtype\20240521 DL模型\moco_feature',
        help="Directory to save output data",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=r'E:\python_DL\model_best.pth.tar',
        help="Feature extractor weights checkpoint",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default='resnet50',
        help="Backbone of the feature extractor. Should match the shape of the weights file, if provided.",
    )
    parser.add_argument(
        "--imagenet",
        action="store_true",
        help="Use imagenet pretrained weights instead of a custom feature extractor weights checkpoint.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--out_size",
        help="Resize the square tile to this output size (in pixels).",
        type=int,
        default=160,
    )
    parser.add_argument(
        "--workers",
        help="The number of workers to use for the data loader. Only relevant when using a GPU.",
        type=int,
        default=0,
    )
    args = parser.parse_args()

    # Derive the slide ID from its name
    # subtype = ['1', '2', '3', '4']
    subtype = ['1']
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = load_encoder(
        backbone=args.backbone,
        checkpoint_file=args.checkpoint,
        use_imagenet_weights=args.imagenet,
        device=device,
    )

    feature_4type = []
    case_4type = []
    for types in subtype:
        folder_path = os.path.join(args.input_data, types)
        files = os.listdir(folder_path)
        train_dataset = MRIDataset(files, folder_path)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                                   num_workers=args.workers, pin_memory=True, drop_last=True)
        total_feature, total_name = extract_features(model, device, train_loader)
        feature_4type += total_feature
        case_4type += total_name
    print(np.array(feature_4type).shape)
    print(np.array(case_4type).shape)
    new_dataframe = pd.DataFrame(data=np.array(feature_4type), index=case_4type,
                                 columns=[str(i) for i in range(np.array(feature_4type).shape[1])])
    new_dataframe.to_excel(r'E:\study\EC\subtype\20240521 DL模型\moco_feature\features_800epoch.xlsx')