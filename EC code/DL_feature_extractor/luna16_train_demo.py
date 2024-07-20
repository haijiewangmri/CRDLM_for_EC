import argparse
import json
import logging
import time
import gc
import sys
import math

import numpy as np
import torch
from vgg3D import VGG11_3D

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

print('CUDA_available', torch.cuda.is_available())
import torch.optim as optim
import google.protobuf
print(google.protobuf.__version__)
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter

from warmup_scheduler import GradualWarmupScheduler

import monai
from monai.data import DataLoader, CacheDataset
from monai.transforms import (
    Compose,
    LoadImageD,
    EnsureChannelFirstD,
    SpacingD,
    LambdaD,
    RandCropByPosNegLabelD,
    CastToTyped,
    RandRotate90D,
    RandAdjustContrastD,
    RandAffineD,
    RandGaussianNoiseD,
    ResizeWithPadOrCropD,
    RandFlipD,
    Compose,
    adaptor,
)

from monai.visualize import plot_2d_or_3d_image
from monai.utils import set_determinism
from monai.data.utils import no_collation

from monai.networks.nets.resnet import resnet10

# from visualize_image import (
#     box_points_train,
# )
# from models.convnext import convnext_tiny

def main():
    #参数设置
    parser = argparse.ArgumentParser(description="PyTorch Object Classification Training")
    parser.add_argument(
        "-e",
        "--environment-file",
        default="./config/luna_environment.json",
        help="environment json file that stores environment path",
    )

    parser.add_argument(
        "-GPU",
        "--Choose-GPU",
        default=2,
        help="Select the GPU you want to use",
    )

    args = parser.parse_args()# 解析命令行参数

    amp = True
    if amp:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    monai.config.print_config()#打印monai配置库的信息。 包括当前的环境设置、MONAI 版本信息、PyTorch 版本信息等

    torch.set_num_threads(4)#多线程

    env_dict = json.load(open(args.environment_file, "r"))

    for k, v in env_dict.items():
        setattr(args, k, v)

    set_determinism(seed=0)
    writer = SummaryWriter(args.tfevent_path)
    
    # 设置全局参数
    val_interval = 5
    batch_size = 40
    max_epochs = 500
    num_classes = 2
    num_auc = 0.0
    best_val_epoch = -1  # the epoch that gives best validation metrics
    #早停
    early_stop_patience = 10
    best_val_loss = float('inf')
    no_improvement_count = 0

    
    
    #数据参数
    spatial_size=(64, 64, 32)
    aug_ratio: float = 0.4
    debug: bool = False
    
    spacing = (0.7, 0.7, 1.5)
    image_key = 'image'
    label_key = 'label'
    mask_key = 'mask'
    img_msk_key = [image_key, mask_key]
    larger_patch_size = [spatial_size[0] + 16, spatial_size[1] + 16, spatial_size[2] + 16]
    file = "/homes/clwang/Data/LIDC-IDRI-Crops-Norm/data-minmax/train_datalist_8-2_minmax_remove3(ver3)_feature_sphericity.json"
    with open(file) as f:
        content = json.load(f)

    additional_transforms = [
        RandRotate90D(
            keys=img_msk_key,
            prob=aug_ratio,
        ),
        RandFlipD(
            keys=img_msk_key,
            prob=aug_ratio,
            spatial_axis=(0,1),
        ),
        RandAdjustContrastD(
            keys=image_key,
            prob=aug_ratio,
            gamma=(0.8, 1.2)
        ),
        RandAffineD(
            keys=img_msk_key,
            prob=aug_ratio,
            rotate_range=[math.pi / 30, math.pi / 30],
            shear_range=[0.1, 0.1],
            translate_range=[5, 5],
            scale_range=[0.1, 0.1],
            mode=["bilinear", "nearest"],
            padding_mode=['reflection', 'border'],
        ),
        RandGaussianNoiseD(
            keys=image_key,
            prob=aug_ratio/2,
            mean=0.0,
            std=0.05,
        )
    ]

    train_transforms = Compose([
        LoadImageD(keys=img_msk_key, dtype=np.float32),
        EnsureChannelFirstD(keys=img_msk_key),
        SpacingD(keys=img_msk_key, pixdim=spacing, mode=['bilinear', 'nearest']),
        LambdaD(keys=label_key, func=lambda x: float(x>3)),
        RandCropByPosNegLabelD(
            keys=img_msk_key,
            label_key=mask_key,
            spatial_size=larger_patch_size,
            pos=1.0, neg=0, num_samples=1,
            allow_smaller=True,
        ),
        *additional_transforms,
        ResizeWithPadOrCropD(keys=img_msk_key, spatial_size=spatial_size, mode="reflect"),
        CastToTyped(keys=img_msk_key, dtype=np.float32),
    ])

    valid_transforms = Compose([
        LoadImageD(keys=img_msk_key, dtype=np.float32),
        EnsureChannelFirstD(keys=img_msk_key),
        SpacingD(keys=img_msk_key, pixdim=spacing, mode=['bilinear', 'nearest']),
        LambdaD(keys=label_key, func=lambda x: float(x>3)),
        RandCropByPosNegLabelD(
            keys=img_msk_key,
            label_key=mask_key,
            spatial_size=larger_patch_size,
            pos=1.0, neg=0, num_samples=1,
            allow_smaller=True,
        ),
        ResizeWithPadOrCropD(keys=img_msk_key, spatial_size=spatial_size, mode="reflect"),
        CastToTyped(keys=img_msk_key, dtype=np.float32),
    ])


    train_set, valid_set = train_test_split(content, test_size=0.2, shuffle=True)
    
    train_label_list = [int(item['label'] > 3) for item in train_set]
    class_sample_count = np.bincount(train_label_list)
    w = 1. / class_sample_count
    weights = np.zeros(len(train_label_list))
    for i, cls in enumerate(train_label_list):
        weights[i] = w[cls]#哲理没用上

    cache_rate = 0 if debug else 1
    
    train_ds = CacheDataset(
            data=train_set,
            transform=train_transforms,
            cache_rate= 1,
        )   
    val_ds = CacheDataset(
            data=valid_set,
            transform=valid_transforms,
            cache_rate= 1,
        )

    train_loader = DataLoader(
            train_ds,
            batch_size = batch_size,
            shuffle = True,
            num_workers = 1,#default
            pin_memory = torch.cuda.is_available(),
            collate_fn = no_collation,
            persistent_workers = True,
        )
    val_loader = DataLoader(
            val_ds,
            batch_size = batch_size,
            shuffle = True,
            num_workers = 1,#default 0 
            pin_memory = torch.cuda.is_available(),
            collate_fn = no_collation,
            persistent_workers = True,
    ) 

    # 3. build model
    if torch.cuda.is_available():
        DEVICE = torch.device(f"cuda:{args.Choose_GPU}")
    else:
        DEVICE = torch.device("cpu") 

    # 实例化模型并且移动到GPU
    criterion = nn.BCEWithLogitsLoss()#设置loss函数

    # model_ft = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(DEVICE)
    model_ft = resnet10(spatial_dims = 3, n_input_channels=1, num_classes=2).to(DEVICE)
    # model_ft = VGG11_3D(num_classes=2, input_channels=1).to(DEVICE)


    # num_ftrs = model_ft.head.in_features#模型最后一层的输入特征数
    # model_ft.head = nn.Linear(num_ftrs, num_classes)
    # model_ft.to(DEVICE)

    # initlize optimizer 选择简单暴力的Adam优化器，学习率调低
    optimizer = optim.Adam(model_ft.parameters(), lr=1e-4)#优化器
    cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=5,eta_min= 1e-6)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=cosine_schedule)#看一下能不能用
    #在total_epoch个epoch后达到目标学习率，也就是warmup持续的代数
    
    # use amp to accelerate training
    scaler = torch.cuda.amp.GradScaler() if amp else None

    epoch_len = len(train_ds) // batch_size

    for epoch in range(max_epochs):
        #----------Training-----------
        print(f"epoch {epoch + 1}/{max_epochs}".center(40,'*'))
        model_ft.train()
        scheduler_warmup.step()#调整、逐步增加学习率，学习率热身
        epoch_loss = 0
        total_num = len(train_loader.dataset)
        step = 0

        start_time = time.time()
        train_outputs_all = []
        train_targets_all = []
        train_outputs_softmax_all = []
        for train_data in train_loader:
             #梯度置零，因为反向传播过程中梯度会累加上一次循环的梯度
            step += 1
            print(step)
            #input_images: List[Tensor])
            input_images = [batch_data_ii["image"].to(DEVICE) for batch_data_i in train_data for batch_data_ii in batch_data_i] 
            input_images = torch.stack(input_images, dim=0)#[20,1,64,64,32]
            
            #label:List[Tensor])
            label_tensors = [batch_data_ii["label"] for batch_data_i in train_data for batch_data_ii in batch_data_i]
            label_array_values = [np.unique(np.array(label).astype(np.int64)).tolist() for label in label_tensors]
            # Creating one-hot labels
            label_origin = torch.zeros((len(train_data), num_classes), dtype=torch.int)
            for i in range(len(label_array_values)):
                value = label_array_values[i]
                for j in range(len(value)):
                    index = int(value[j])
                    label_origin[i][index] = 1        
            target = label_origin.to(DEVICE)#[20,2]
            
            optimizer.zero_grad()
            
            if amp and (scaler is not None):
                with torch.cuda.amp.autocast():
                    output = model_ft(input_images)#torch.size([40,3])
                    loss = criterion(output.float(), target.float())
                    
                scaler.scale(loss).backward()#Loss 反向传播
                scaler.step(optimizer)#反向传播后参数更新
                scaler.update()# 清除缩放的梯度以供下一次使用
            else:
                output = model_ft(input_images)
                loss = criterion(output.float(), target.float())
                loss.backward()
                optimizer.step()
                
            
            train_outputs_softmax = torch.softmax(output, dim=1)
            
            # save outputs for evaluation
            train_outputs_all += [output_i.cpu().detach().numpy() for output_i in output]
            train_targets_all += [target_i.cpu().detach().numpy()[1] for target_i in target]
            train_outputs_softmax_all += [train_outputs_softmax_i.cpu()[1].detach().numpy() for train_outputs_softmax_i in train_outputs_softmax]

            # save to tensorboard
            epoch_loss += loss.detach().item()
            print('print_loss', loss.detach().item())
            epoch_len = total_num // train_loader.batch_size

            writer.add_scalar("train_loss", loss.detach().item(), epoch_len * epoch + step)
            
        end_time = time.time()
        print(f"Training time: {end_time-start_time}s")
        # 假设 val_targets_all 和 val_outputs_all 分别是真实标签和模型的预测概率
            
        fpr, tpr, thresholds = roc_curve(train_targets_all, train_outputs_softmax_all)
        # 计算 ROC 曲线下方的面积 AUC
        roc_auc = auc(fpr, tpr)
        for i in range(len(fpr)):
            writer.add_scalar('train_FPR', fpr[i], global_step=i)
            writer.add_scalar('train_TPR', tpr[i], global_step=i)
        writer.add_scalar('train_AUC', roc_auc, epoch) 
        print('train_AUC', roc_auc)
        del input_images, train_data
        torch.cuda.empty_cache()#手动释放GPU占用显存
        gc.collect()

        epoch_loss /= step
        writer.add_scalar("avg_train_loss", epoch_loss, epoch + 1)
        print('epoch:{},loss:{}'.format(epoch, epoch_loss))
        writer.add_scalar("train_lr", scheduler_warmup.get_last_lr()[0], epoch + 1)

        # save last trained model
        torch.save(model_ft, env_dict["model_path"][:-3] + "_last.pt")# save model
        print("saved last model")

    # ------------- Validation for model selection -------------

    # # 验证过程
    # def val(model, device, val_loader):
        if (epoch + 1) % val_interval == 0:
            model_ft.eval()
            val_outputs_all = []
            val_targets_all = []
            val_outputs_softmax_all = []
            start_time = time.time()
            #不启用Batch Normalization 和 Dropout
            #在eval模式下，Dropout层会让所有的激活单元都通过，而Batch Normalization层会停止计算和更新mean和var，直接使用在训练阶段已经学出的mean和var值。
            test_loss = 0
            total_num_val = len(val_loader.dataset)#没错
            step1 = 0
            with torch.no_grad():#反向传播不设置自动求导==大大节约显存和内存
                for key in list(locals().keys()):    
                    if key.startswith("correct"):
                        locals().pop(key)

                for val_data in val_loader:
                    #input_images: List[Tensor])
                    input_images = [batch_data_ii["image"].to(DEVICE) for batch_data_i in val_data for batch_data_ii in batch_data_i] 
                    input_images = torch.stack(input_images, dim=0)

                    #label:List[Tensor])
                    label_tensors = [batch_data_ii["label"] for batch_data_i in val_data for batch_data_ii in batch_data_i]
                    label_array_values = [np.unique(np.array(label).astype(np.int64)).tolist() for label in label_tensors]
                    # Creating one-hot labels
                    label_origin = torch.zeros((len(val_data), num_classes), dtype=torch.int)     
                    for i in range(len(label_array_values)):
                        value = label_array_values[i]
                        for j in range(len(value)):
                            index = int(value[j])
                            label_origin[i][index] = 1        
                    target = label_origin.to(DEVICE)              
                                        
                    step1 += 1
                    if amp:
                        with torch.cuda.amp.autocast():#显式触发垃圾回收
                            val_outputs = model_ft(input_images)#validation
                    else:
                        val_outputs = model_ft(input_images)

                    loss = criterion(val_outputs.float(), target.float())
                    test_loss += loss.detach().item()
                    
                    val_outputs_softmax = torch.softmax(val_outputs, dim=1)

                    
                    epoch_len1 = total_num_val // val_loader.batch_size
                    writer.add_scalar("valid_loss", loss.detach().item(), epoch_len1 * epoch + step1)

                    # save outputs for evaluation
                    val_outputs_all += [val_outputs_i.cpu().detach().numpy() for val_outputs_i in val_outputs]
                    val_targets_all += [target_i.cpu().detach().numpy()[1] for target_i in target]
                    val_outputs_softmax_all += [val_outputs_softmax_i.cpu()[1].detach().numpy() for val_outputs_softmax_i in val_outputs_softmax]

            test_loss /= step
            end_time = time.time()
            print(f"Validation time: {end_time-start_time}s")   
            writer.add_scalar("avg_valid_loss", loss.detach().item(), epoch)

            # compute metrics
            del input_images
            torch.cuda.empty_cache()   
            # 假设 val_targets_all 和 val_outputs_all 分别是真实标签和模型的预测概率
            fpr, tpr, thresholds = roc_curve(val_targets_all, val_outputs_softmax_all)
            # 计算 ROC 曲线下方的面积 AUC
            roc_auc = auc(fpr, tpr)
            for i in range(len(fpr)):
                writer.add_scalar('val_FPR', fpr[i], global_step=i)
                writer.add_scalar('val_TPR', tpr[i], global_step=i)
            writer.add_scalar('val_AUC', roc_auc, epoch) 
            print('val_roc_auc', roc_auc)
            writer.add_scalar('best_val_AUC', num_auc, epoch) 
            # save best trained model
            if roc_auc > num_auc:
                num_auc = roc_auc
                best_val_epoch = epoch + 1
                torch.save(model_ft, env_dict["model_path"])# save model
                print("saved new best metric model")
                plt.figure()
                plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend(loc='lower right')
                plt.savefig(env_dict["roc_path"])
                plt.show()
            print(
                "current epoch: {} current metric: {:.4f} "
                "best metric: {:.4f} at epoch {}".format(
                    epoch + 1, roc_auc, num_auc, best_val_epoch
                )
            )       
            #early stop
            if test_loss < best_val_loss:
                best_val_loss = test_loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= early_stop_patience:
                    print(f'Early stopping at epoch {epoch}')
                    break
            

    writer.close()

if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()

                




    

