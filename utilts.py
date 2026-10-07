import os
import numpy as np
import cv2
import importlib
import shutil
import sys
import torch
import torch.nn.functional as F


def make_dir(path):
    if os.path.exists(path)==False:
        os.makedirs(path)

def access_model(choose_model):
    choose_model_dir_name = choose_model + '_no_sigmoid'
    model_function = choose_model + '_No_Sigmoid'
    module_name = f"model.{choose_model}.{choose_model_dir_name}"
    module = importlib.import_module(module_name)
    model_func = getattr(module, model_function)
    return model_func


def check_path(path):
    if os.path.exists(path) == True:
        print("Error: The workspace of the training pool already exists.")
        print("The conflicting directories are:", path)
        sys.exit(0)

def PKEIPG(dataset_name, origin_img_dir, origin_points_dir,TRAIN_IMG_DIR,TRAIN_MASK_DIR,train_points_dir,nc_img_dir,nc_mask_dir,nc_points_dir):
    input_image_path = origin_img_dir
    input_points_path = origin_points_dir

    output_image_path = TRAIN_IMG_DIR
    output_masks_path = TRAIN_MASK_DIR
    output_points_path = train_points_dir

    def get_img_norm(dataset_name):
        if dataset_name == 'SIRST_1_1_point_new':
            norm_params = dict(mean=101.06385040283203, std=34.619606018066406)
        elif dataset_name == 'NUDT_SIRST_1_1_point':
            norm_params = dict(mean=107.80905151367188, std=33.02274703979492)
        elif dataset_name == 'IRSTD_1K_point':
            norm_params = dict(mean=87.4661865234375, std=39.71953201293945)
        elif dataset_name == 'SIRST3':
            norm_params = dict(mean=101.06383514404297, std=34.619606018066406)
        return norm_params

    input_img_list = os.listdir(input_image_path)

    for i in range(len(input_img_list)):

        img_path = os.path.join(input_image_path, input_img_list[i])
        points_path = os.path.join(input_points_path, input_img_list[i])

        out_img_path = os.path.join(output_image_path, input_img_list[i])
        out_mask_path = os.path.join(output_masks_path, input_img_list[i])
        out_points_path = os.path.join(output_points_path, input_img_list[i])

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        oriimg = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(points_path, cv2.IMREAD_GRAYSCALE)
        kernel1 = np.array([[0, 0, -0.5],
                            [0, 1, 0],
                            [-0.5, 0, 0]])
        kernel2 = np.array([[-0.5, 0, 0],
                            [0, 1, 0],
                            [0, 0, -0.5]])
        kernel3 = np.array([[0, -0.5, 0],
                            [0, 1, 0],
                            [0, -0.5, 0]])
        kernel4 = np.array([[0, 0, 0],
                            [-0.5, 1, -0.5],
                            [0, 0, 0]])
        newimage = (img - get_img_norm(dataset_name)['mean']) / get_img_norm(dataset_name)['std']

        result1 = cv2.filter2D(newimage, -1, kernel1)
        result2 = cv2.filter2D(newimage, -1, kernel2)
        result3 = cv2.filter2D(newimage, -1, kernel3)
        result4 = cv2.filter2D(newimage, -1, kernel4)

        attention = F.sigmoid(torch.from_numpy(result1 * result2 + result3 * result4)).numpy()
        img = np.rint(img * attention).astype(np.uint8)
        if img is None or mask is None:
            print(f"图像或mask读取失败：{input_img_list[i]}")
            continue

        points = np.where(mask == 255)
        if len(points[0]) == 0:
            cv2.imwrite(out_img_path, img)
            cv2.imwrite(out_mask_path, mask)
            cv2.imwrite(out_points_path, mask)
            continue

        merged_result = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        base_L_ep = 15
        base_L_dp = 8
        base_alpha = 0.0125
        height, width = img.shape
        for i in range(len(points[0])):
            center_y, center_x = points[0][i], points[1][i]
            L_ep = base_L_ep
            L_dp = base_L_dp
            alpha = base_alpha
            '''
            Left baseline
            '''
            left = max(center_x - L_ep - 1, 0)
            upper = max(center_y - L_dp, 0)
            down = min(center_y + L_dp + 1, img.shape[0])
            roi = img[upper:down, left:center_x]
            edges = cv2.Canny(roi, 25, 50)
            max_vec = np.max(edges, axis=0)
            flag = 0
            for i in range(len(max_vec) - 1, 1, -1):
                if max_vec[i] == 0:
                    flag = len(max_vec) - 1 - i
                    break
            if flag == 0:
                flag = 2
            left_boundary = center_x - flag

            '''
            Right baseline
            '''
            right = min(center_x + L_ep + 1, img.shape[1])
            upper = max(center_y - L_dp, 0)
            down = min(center_y + L_dp + 1, img.shape[0])

            roi = img[upper:down, center_x:right]
            edges = cv2.Canny(roi, 25, 50)
            max_vec = np.max(edges, axis=0)
            flag = 0
            for i in range(len(max_vec)):
                if max_vec[i] == 0:
                    flag = i + 1
                    break
            if flag == 0:
                flag = 3
            right_boundary = center_x + flag

            '''
            Up baseline
            '''
            left = max(center_x - L_dp, 0)
            right = min(center_x + L_dp + 1, img.shape[1])
            upper = max(center_y - L_ep - 1, 0)

            roi = img[upper:center_y, left:right]
            edges = cv2.Canny(roi, 25, 50)
            max_vec = np.max(edges, axis=1)
            flag = 0
            for i in range(len(max_vec) - 1, 1, -1):
                if max_vec[i] == 0:
                    flag = len(max_vec) - 1 - i
                    break
            if flag == 0:
                flag = 2
            upper_boundary = center_y - flag

            '''
            Down baseline
            '''
            left = max(center_x - L_dp, 0)
            right = min(center_x + L_dp + 1, img.shape[1])
            down = min(center_y + L_ep + 1, img.shape[0])
            roi = img[center_y:down, left:right]
            edges = cv2.Canny(roi, 25, 50)
            max_vec = np.max(edges, axis=1)
            flag = 0
            for i in range(len(max_vec)):
                if max_vec[i] == 0:
                    flag = i + 1
                    break
            if flag == 0:
                flag = 3
            down_boundary = center_y + flag

            left_boundary = max(left_boundary, 0)
            right_boundary = min(right_boundary, width)
            upper_boundary = max(upper_boundary, 0)
            down_boundary = min(down_boundary, height)

            center_pixel_value = img[center_y, center_x]
            '''
            Pa
            '''
            roi = np.uint8(oriimg[upper_boundary:down_boundary, left_boundary:right_boundary])
            P_f = np.zeros_like(roi).astype(np.float32)
            xita = alpha * center_pixel_value + (1 - alpha) * np.mean(roi)
            max_roi = np.max(roi)
            min_roi = np.min(roi)
            for a in range(roi.shape[0]):
                for b in range(roi.shape[1]):
                    if roi[a, b] > xita:
                        P_f[a, b] = 1 - ((max_roi - roi[a, b]) / (max_roi - xita)) * 0.5
                    else:
                        P_f[a, b] = ((roi[a, b] - min_roi) / (xita - min_roi)) * 0.5
            '''
            Pl
            '''
            roi_left = np.uint8(oriimg[upper_boundary:down_boundary, left_boundary:center_x])
            P_left = np.zeros_like(roi_left).astype(np.float32)
            xita = alpha * center_pixel_value + (1 - alpha) * np.mean(roi_left)
            max_roi = np.max(roi_left)
            min_roi = np.min(roi_left)
            for a in range(roi_left.shape[0]):
                for b in range(roi_left.shape[1]):
                    if roi_left[a, b] > xita:
                        P_left[a, b] = 1 - ((max_roi - roi_left[a, b]) / (max_roi - xita)) * 0.5
                    else:
                        P_left[a, b] = ((roi_left[a, b] - min_roi) / (xita - min_roi)) * 0.5
            '''
            Pr
            '''
            roi_right = np.uint8(oriimg[upper_boundary:down_boundary, center_x:right_boundary])
            P_right = np.zeros_like(roi_right).astype(np.float32)
            xita = alpha * center_pixel_value + (1 - alpha) * np.mean(roi_right)
            max_roi = np.max(roi_right)
            min_roi = np.min(roi_right)
            for a in range(roi_right.shape[0]):
                for b in range(roi_right.shape[1]):
                    if roi_right[a, b] > xita:
                        P_right[a, b] = 1 - ((max_roi - roi_right[a, b]) / (max_roi - xita)) * 0.5
                    else:
                        P_right[a, b] = ((roi_right[a, b] - min_roi) / (xita - min_roi)) * 0.5
            P_v = np.hstack((P_left, P_right))
            '''
            Pu
            '''
            roi_upper = np.uint8(oriimg[upper_boundary:center_y, left_boundary:right_boundary])
            P_upper = np.zeros_like(roi_upper).astype(np.float32)
            xita = alpha * center_pixel_value + (1 - alpha) * np.mean(roi_upper)
            max_roi = np.max(roi_upper)
            min_roi = np.min(roi_upper)
            for a in range(roi_upper.shape[0]):
                for b in range(roi_upper.shape[1]):
                    if roi_upper[a, b] > xita:
                        P_upper[a, b] = 1 - ((max_roi - roi_upper[a, b]) / (max_roi - xita)) * 0.5
                    else:
                        P_upper[a, b] = ((roi_upper[a, b] - min_roi) / (xita - min_roi)) * 0.5
            '''
            Pd
            '''
            roi_down = np.uint8(oriimg[center_y:down_boundary, left_boundary:right_boundary])
            P_down = np.zeros_like(roi_down).astype(np.float32)
            xita = alpha * center_pixel_value + (1 - alpha) * np.mean(roi_down)
            max_roi = np.max(roi_down)
            min_roi = np.min(roi_down)
            for a in range(roi_down.shape[0]):
                for b in range(roi_down.shape[1]):
                    if roi_down[a, b] > xita:
                        P_down[a, b] = 1 - ((max_roi - roi_down[a, b]) / (max_roi - xita)) * 0.5
                    else:
                        P_down[a, b] = ((roi_down[a, b] - min_roi) / (xita - min_roi)) * 0.5
            P_h = np.vstack((P_upper, P_down))
            P_final = (P_f + P_v + P_h) / 3
            mask_temp = np.zeros_like(P_final).astype(np.uint8)
            mask_index = np.where(P_final >= 0.5)
            mask_temp[mask_index] = 255

            temp_contour_mask_2 = np.zeros_like(img).astype(np.uint8)
            temp_contour_mask_2[upper_boundary:down_boundary, left_boundary:right_boundary] = mask_temp

            merged_result[upper_boundary:down_boundary, left_boundary:right_boundary] += mask_temp
            merged_result[center_y, center_x] = 255

        merged_result = merged_result + mask
        merged_result = np.where(merged_result > 0, 255, 0)
        cv2.imwrite(out_img_path, oriimg)
        cv2.imwrite(out_mask_path, merged_result)
        cv2.imwrite(out_points_path,mask)

    print("完成初始伪标签生成，生成的样本张数：", len(os.listdir(output_image_path)))

def move_files(file_list, src_path, dst_path):
    for file_name in file_list:
        src_file = os.path.join(src_path, file_name)
        dst_file = os.path.join(dst_path, file_name)
        if os.path.exists(src_file):
            try:
                shutil.copy(src_file, dst_file)
                os.remove(src_file)
            except PermissionError as e:
                print(f"PermissionError: {e}")
        else:
            print(f"File {src_file} does not exist.")

def update_gt_update_degen_corr(pred, gt_masks, thresh_Tb, thresh_k, size,degen=0.9):

    update_gt_masks = gt_masks.copy()

    background_length = 33
    target_length = 3

    num_labels, label_image = cv2.connectedComponents((gt_masks > 0.5).astype(np.uint8))

    background_kernel = np.ones((background_length, background_length), np.uint8)
    target_kernel = np.ones((target_length, target_length), np.uint8)

    max_limitation = size[0] * size[1] * 0.0015

    combined_thresh_mask = np.zeros_like(pred, dtype=np.float32)

    for region_num in range(1, num_labels):
        region_coords = np.argwhere(label_image == region_num)
        centroid = np.mean(region_coords, axis=0).astype(int)

        cur_point_mask = np.zeros_like(pred, dtype=np.uint8)
        cur_point_mask[centroid[0], centroid[1]] = 1

        nbr_mask = cv2.dilate(cur_point_mask, background_kernel) > 0
        targets_mask = cv2.dilate(cur_point_mask, target_kernel) > 0

        region_size_ratio = len(region_coords) / max_limitation
        threshold_start = (pred * nbr_mask).max() * thresh_Tb
        threshold_delta = thresh_k * ((pred * nbr_mask).max() - threshold_start) * region_size_ratio
        threshold = threshold_start + threshold_delta
        threshold = threshold.cpu().numpy() if isinstance(threshold, torch.Tensor) else threshold

        thresh_mask = (pred * nbr_mask > threshold).astype(np.float32)

        num_labels_thresh, label_image_thresh = cv2.connectedComponents(thresh_mask.astype(np.uint8))
        for num_cur in range(1, num_labels_thresh):
            curr_mask = (label_image_thresh == num_cur).astype(np.float32)
            if np.sum(curr_mask * targets_mask) == 0:
                thresh_mask -= curr_mask

        combined_thresh_mask = np.maximum(combined_thresh_mask, thresh_mask)

    target_patch = (update_gt_masks * combined_thresh_mask + pred * combined_thresh_mask) / 2
    background_patch = update_gt_masks * (1 - combined_thresh_mask) * degen
    update_gt_masks = background_patch + target_patch

    update_gt_masks = np.maximum(update_gt_masks, (gt_masks == 1).astype(np.float32))

    return update_gt_masks
