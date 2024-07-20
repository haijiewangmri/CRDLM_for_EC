import argparse
import os
import random
import pandas as pd
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

# data_path = r'E:\study\EC\subtype\total_data\fudan_subtype_292'
# df = pd.read_csv(r'E:\study\EC\subtype\20240403 单独复旦\fudan292clinical.csv')

# data_path = r'E:\study\EC\subtype\total_data\chaoyang_154'
# df = pd.read_csv(r'E:\study\EC\subtype\20240403 单独复旦\chaoyang154clinical.csv')

data_path = r'E:\study\EC\subtype\total_data\shzl_80'
df = pd.read_csv(r'E:\study\EC\subtype\20240403 单独复旦\SHZL80clinical.csv')

ouput_path = r'E:\study\EC\subtype\Data_slice'
cases = df['CaseName'].tolist()
labels = df['label'].tolist()


def get_random_crop_coordinates(roi, crop_size):
    """ 获取随机裁剪的坐标，确保ROI区域在裁剪图像内 """
    h, w = roi.shape
    crop_h, crop_w = crop_size

    # 找到ROI区域的边界
    roi_indices = np.argwhere(roi == 1)
    y_min, x_min = roi_indices.min(axis=0)
    y_max, x_max = roi_indices.max(axis=0)

    # 确保裁剪框能包含整个ROI区域
    y_start = random.randint(max(0, y_max - crop_h), min(y_min, h - crop_h))
    x_start = random.randint(max(0, x_max - crop_w), min(x_min, w - crop_w))

    return y_start, y_start + crop_h, x_start, x_start + crop_w


def get_mid_crop_coordinates(roi, crop_size):
    """ 获取随机裁剪的坐标，确保ROI区域在裁剪图像内 """
    h, w = roi.shape
    crop_h, crop_w = crop_size

    # 找到ROI区域的边界
    roi_indices = np.argwhere(roi == 1)
    y_min, x_min = roi_indices.min(axis=0)
    y_max, x_max = roi_indices.max(axis=0)

    # 确保裁剪框能包含整个ROI区域
    y_start = max(0, (y_max + y_min)//2 - crop_h//2)
    x_start = max(0, (x_max + x_min)//2 - crop_w//2)

    return y_start, y_start + crop_h, x_start, x_start + crop_w


def crop_image(image, crop_coordinates):
    y_start, y_end, x_start, x_end = crop_coordinates
    return image[y_start:y_end, x_start:x_end]


def add_zero(input_data, output_size=(16, 36, 960)):
    """
        给输入的矩阵和输出的尺寸，差的地方填零
        在3，73，762的末尾数据上补全0，变成160，240，240
        """
    x, y, z = input_data.shape  # 举例3,73,762
    target_x, target_y, target_z = output_size

    if x == target_x and y == target_y and z == target_z:
        return input_data

    result = None
    if x <= target_x:
        padding = round((target_x - x) / 2)
        result = np.zeros([target_x, y, z], dtype=input_data.dtype)
        result[padding:padding + x, :, :] = input_data
    if x > target_x:
        padding = (x - target_x) // 2
        result = input_data[padding: padding + target_x, :, :]
    # 这里输出 x, y, target_z的图像

    result0 = None
    if y <= target_y:
        padding = round((target_y - y) / 2)
        result0 = np.zeros([target_x, target_y, z], dtype=input_data.dtype)
        result0[:, padding:padding + y, :] = result
    if y > target_y:
        padding = (y - target_y) // 2
        result0 = result[:, padding: padding + target_y, :]

    result1 = None
    if z <= target_z:
        padding = round((target_z - z) / 2)
        result1 = np.zeros(output_size, dtype=input_data.dtype)
        result1[:, :, padding:padding + z] = result0
    if z > target_z:
        padding = (z - target_z) // 2
        result1 = result0[:, :, padding: padding + target_z]
    return result1

n = 0
total_layers = 0
divide_labels = [0, 0, 0, 0]
for i, case in enumerate(cases):
# for case in [102528417, 102729703, 102624361]:
    i = cases.index(case)
    case = str(case)
    label = str(labels[i])
    # t1, t2, t1ce, dwi
    t1 = sitk.ReadImage(os.path.join(data_path, case, 't1_CLAHE.nii.gz'))
    t2 = sitk.ReadImage(os.path.join(data_path, case, 't2_CLAHE.nii.gz'))
    t1ce = sitk.ReadImage(os.path.join(data_path, case, 'ce_CLAHE_resize_t2.nii.gz'))
    dwi = sitk.ReadImage(os.path.join(data_path, case, 'dwi_CLAHE_resize_t2.nii.gz'))
    roi = sitk.ReadImage(os.path.join(data_path, case, 't2_roi.nii.gz'))

    # t1 = sitk.ReadImage(os.path.join(data_path, case, 't1_CLAHE.nii.gz'))
    # t2 = sitk.ReadImage(os.path.join(data_path, case, 't2_CLAHE.nii.gz'))
    # t1ce = sitk.ReadImage(os.path.join(data_path, case, 'ce_CLAHE.nii.gz'))
    # dwi = sitk.ReadImage(os.path.join(data_path, case, 'dwi_CLAHE.nii.gz'))
    # roi = sitk.ReadImage(os.path.join(data_path, case, 'seg.nii.gz'))

    t1_array = sitk.GetArrayFromImage(t1)
    t2_array = sitk.GetArrayFromImage(t2)
    t1ce_array = sitk.GetArrayFromImage(t1ce)
    dwi_array = sitk.GetArrayFromImage(dwi)
    roi_array = sitk.GetArrayFromImage(roi)

    layers = np.where(roi_array.sum(axis=(1, 2)) != 0)[0]
    spacing = t2.GetSpacing()

    n += 1
    total_layers += len(layers)
    divide_labels[int(label)-1] += len(layers)
    print(n, len(layers), total_layers, divide_labels)
    # for l in layers:
    #     temp_t1 = t1_array[l, :, :]
    #     temp_t2 = t2_array[l, :, :]
    #     temp_t1ce = t1ce_array[l, :, :]
    #     temp_dwi = dwi_array[l, :, :]
    #     temp_roi = roi_array[l, :, :]
    #
    #     crop_coords = get_mid_crop_coordinates(temp_roi, (160, 160))
    #     # crop_coords = get_random_crop_coordinates(temp_roi, (128, 128))
    #     cropped_t1 = crop_image(temp_t1, crop_coords)
    #     cropped_t2 = crop_image(temp_t2, crop_coords)
    #     cropped_t1ce = crop_image(temp_t1ce, crop_coords)
    #     cropped_dwi = crop_image(temp_dwi, crop_coords)
    #     cropped_roi = crop_image(temp_roi, crop_coords)
    #
    #     temp_image = np.stack((cropped_t1, cropped_t2, cropped_t1ce, cropped_dwi), axis=0)
    #     temp_roi = np.stack((cropped_roi, cropped_roi, cropped_roi, cropped_roi), axis=0)
    #
    #     # temp_image = np.stack((cropped_t2, cropped_t1ce, cropped_dwi), axis=0)
    #     # temp_roi = np.stack((cropped_roi, cropped_roi, cropped_roi), axis=0)
    #
    #     temp_image = add_zero(temp_image, (4, 160, 160))
    #     temp_roi = add_zero(temp_roi, (4, 160, 160))
    #
    #     new_image = sitk.GetImageFromArray(temp_image)
    #     new_roi = sitk.GetImageFromArray(temp_roi)
    #     new_image.SetSpacing(spacing)
    #     new_roi.SetSpacing(spacing)
    #
    #     if not os.path.exists(os.path.join(ouput_path, 'data', label)):
    #         os.makedirs(os.path.join(ouput_path, 'data', label))
    #         os.makedirs(os.path.join(ouput_path, 'roi', label))
    #     output_temp = os.path.join(ouput_path, 'data', label, f'{case}_{l}_3mri.nii.gz')
    #     sitk.WriteImage(new_image, output_temp)
    #
    #     output_temp = os.path.join(ouput_path, 'roi', label, f'{case}_{l}_roi.nii.gz')
    #     sitk.WriteImage(new_roi, output_temp)
