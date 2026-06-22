# import logging
# import multiprocessing
# from multiprocessing import Lock, Pool

# multiprocessing.set_start_method("spawn", True)  # ! must be at top for VScode debugging
# import argparse
# import glob
# import json
# import math
# import multiprocessing as mp
# import os
# import pathlib
# import pickle
# import re
# import sys
# import warnings
# from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, as_completed, wait
# from functools import reduce
# from importlib import import_module
# from multiprocessing import Lock, Pool
# from run_utils.callbacks.serialize import fig2data, serialize
# import cv2
# import numpy as np
# import psutil
# import scipy.io as sio
# import torch
# import torch.utils.data as data
# import tqdm
# from dataloader.infer_loader import SerializeFileList
# from scipy.ndimage import measurements
# from misc.utils import (
#     color_deconvolution,
#     cropping_center,
#     get_bounding_box,
#     log_debug,
#     log_info,
#     rm_n_mkdir,
# )
# from misc.viz_utils import colorize, visualize_instances_dict
# from skimage import color

import convert_format
# from . import base


# ####
# def _prepare_patching(img, window_size, mask_size, return_src_top_corner=False):
#     """Prepare patch information for tile processing.
#     为infer步骤extract patch
#     Args:
#         img: original input image
#         window_size: input patch size
#         mask_size: output patch size
#         return_src_top_corner: whether to return coordiante information for top left corner of img
        
#     """

#     win_size = window_size
#     msk_size = step_size = mask_size

#     def get_last_steps(length, msk_size, step_size):
#         nr_step = math.ceil((length - msk_size) / step_size)
#         last_step = (nr_step + 1) * step_size
#         return int(last_step), int(nr_step + 1)

#     im_h = img.shape[0]
#     im_w = img.shape[1]

#     last_h, _ = get_last_steps(im_h, msk_size, step_size)
#     last_w, _ = get_last_steps(im_w, msk_size, step_size)

#     diff = win_size - step_size
#     padt = padl = diff // 2
#     padb = last_h + win_size - im_h
#     padr = last_w + win_size - im_w

#     img = np.lib.pad(img, ((padt, padb), (padl, padr), (0, 0)), "reflect")

#     # generating subpatches index from orginal
#     coord_y = np.arange(0, last_h, step_size, dtype=np.int32)
#     coord_x = np.arange(0, last_w, step_size, dtype=np.int32)
#     row_idx = np.arange(0, coord_y.shape[0], dtype=np.int32)
#     col_idx = np.arange(0, coord_x.shape[0], dtype=np.int32)
#     coord_y, coord_x = np.meshgrid(coord_y, coord_x)
#     row_idx, col_idx = np.meshgrid(row_idx, col_idx)
#     coord_y = coord_y.flatten()
#     coord_x = coord_x.flatten()
#     row_idx = row_idx.flatten()
#     col_idx = col_idx.flatten()
#     #
#     patch_info = np.stack([coord_y, coord_x, row_idx, col_idx], axis=-1)
#     if not return_src_top_corner:
#         return img, patch_info
#     else:
#         return img, patch_info, [padt, padl]


# ####
# # def _post_process_patches(post_proc_func, post_proc_kwargs, patch_info, image_info, overlay_kwargs,
# # ):
# #     """Apply post processing to patches.

# #     Args:
# #         post_proc_func: post processing function to use 使用的后处理函数
# #         post_proc_kwargs: keyword arguments used in post processing function 后处理函数中使用的关键参数
# #         patch_info: patch data and associated information
# #         image_info: input image data and associated information
# #         overlay_kwargs: overlay keyword arguments

# #     """
# #     # re-assemble the prediction, sort according to the patch location within the original image
# #     patch_info = sorted(patch_info, key=lambda x: [x[0][0], x[0][1]])
# #     patch_info, patch_data = zip(*patch_info)

    
# #     src_shape = image_info["src_shape"]
# #     src_image = image_info["src_image"]

# #     patch_shape = np.squeeze(patch_data[0]).shape
# #     ch = 1 if len(patch_shape) == 2 else patch_shape[-1]
# #     axes = [0, 2, 1, 3, 4] if ch != 1 else [0, 2, 1, 3]

# #     nr_row = max([x[2] for x in patch_info]) + 1
# #     nr_col = max([x[3] for x in patch_info]) + 1
# #     # 将预测出的patch拼接好
# #     pred_map = np.concatenate(patch_data, axis=0)
# #     pred_map = np.reshape(pred_map, (nr_row, nr_col) + patch_shape)
# #     pred_map = np.transpose(pred_map, axes)
# #     pred_map = np.reshape(
# #         pred_map, (patch_shape[0] * nr_row, patch_shape[1] * nr_col, ch)
# #     )
# #     # crop back to original shape
# #     # 将拼接好的patch剪裁为原来大小
# #     pred_map = np.squeeze(pred_map[: src_shape[0], : src_shape[1]])

# #     # * Implicit protocol
# #     # * a prediction map with instance of ID 1-N
# #     # * and a dict contain the instance info, access via its ID
# #     # * each instance may have type

# #     ######## 7.调用后处理函数对预测输出的进行处理 post_proc.py文件内容
# #     pred_inst, inst_info_dict = post_proc_func(pred_map, **post_proc_kwargs)

# #     overlaid_img = visualize_instances_dict(
# #         src_image.copy(), inst_info_dict, **overlay_kwargs
# #     )

# #     return image_info["name"], pred_map, pred_inst, inst_info_dict, overlaid_img

# def _post_process_patches_hover(post_proc_func, post_proc_kwargs, patch_info, image_info, overlay_kwargs,
# ):
#     """Apply post processing to patches.

#     Args:
#         post_proc_func: post processing function to use 使用的后处理函数
#         post_proc_kwargs: keyword arguments used in post processing function 后处理函数中使用的关键参数
#         patch_info: patch data and associated information
#         image_info: input image data and associated information
#         overlay_kwargs: overlay keyword arguments

#     """
#     # re-assemble the prediction, sort according to the patch location within the original image
#     src_image = image_info["src_image"]
#     patch_info = sorted(patch_info, key=lambda x: [x[0][0], x[0][1]])
#     patch_info, patch_data = zip(*patch_info)
#     # 对每一个patch做后处理，再拼接，并对重复部分做判断
#     win_size=384
#     step_size = 172

#     output_inst_info_dict = {}
#     output_seg = np.zeros((1006,1006),dtype = np.int32)
    
#     # seg_mask = np.zeros((win_size,win_size),dtype=np.bool)
#     # 记录一张图中instance的flag
#     inst_flag=0
#     for patch_id,pred_patch in enumerate(patch_data):
        
#         pred_patch = np.squeeze(pred_patch)
#         pred_inst, inst_info_dict = post_proc_func(pred_patch, **post_proc_kwargs)

#         patch_info_ = patch_info[patch_id]
#         # 填充图像验证后处理程序是否有错误
#         # img = np.lib.pad(src_image, ((106, 388), (106, 388), (0, 0)), "reflect")
#         # save_path = "%s/%s_%s.png" % ("/home/ZXIAO/datasets/Neure_512/output", image_info["name"],patch_id)
#         # cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

#         # overlaid_img = visualize_instances_dict(img.copy()[patch_info_[0]:patch_info_[0]+384,patch_info_[1]:patch_info_[1]+384], inst_info_dict, **overlay_kwargs)
#         # save_path = "%s/overlay/%s_%s.png" % ("/home/ZXIAO/datasets/Neure_512/output", image_info["name"],patch_id)
#         # cv2.imwrite(save_path, cv2.cvtColor(overlaid_img, cv2.COLOR_RGB2BGR))


#         output_seg_ = output_seg[patch_info_[0]:patch_info_[0]+384,patch_info_[1]:patch_info_[1]+384]
#         for inst_id in np.unique(pred_inst):
#             if inst_id==0: continue
#             inst_info_=inst_info_dict[inst_id]
#             y_min=inst_info_['bbox'][0][0]
#             x_min=inst_info_['bbox'][0][1]
#             y_max=inst_info_['bbox'][1][0]
#             x_max=inst_info_['bbox'][1][1]
            
#             seg_mask= pred_inst[y_min:y_max,x_min:x_max]
#             seg_mask_out = output_seg_[y_min:y_max,x_min:x_max]
#             seg_mask[seg_mask!=inst_id]=0
#             seg_mask[seg_mask>0]=1
#             values,counts = np.unique(seg_mask_out,return_counts=True)
#             if counts.shape[0]>1:
#                 values=values[1:]
#                 counts=counts[1:]
#                 most_frequent_value = values[np.argmax(counts)]
#                 seg_mask_out = np.where(seg_mask_out == most_frequent_value, 1, 0)
                
#                 # 计算IOU
#                 intersection = np.sum(seg_mask & seg_mask_out)  # 交集
#                 union = np.sum(seg_mask | seg_mask_out)          # 并集
#                 iou = intersection / union if union != 0 else 0
#                 # 相交较多视为同一个
#                 if iou>0.7:
#                     if output_inst_info_dict[most_frequent_value]["type_prob"]>inst_info_["type_prob"]:
#                         # 舍弃当前新patch上的instance
#                         continue
#                     else:
#                         # 将当前新的代替之前的instance
#                         output_seg[output_seg==most_frequent_value]=0
#                         output_seg_[output_seg_==most_frequent_value]=0
#                         seg_mask_inter=np.zeros_like(seg_mask,dtype=int)
#                         seg_mask_inter[seg_mask&seg_mask_out] = most_frequent_value
#                         output_seg_[y_min:y_max,x_min:x_max]=seg_mask_inter

#                         output_inst_info_dict[most_frequent_value]['type']=inst_info_['type']
#                         output_inst_info_dict[most_frequent_value]['type_prob']=inst_info_['type_prob']
#                 # 相交较少视为两个
#                 # 为0的情况
#                 else:
#                     inst_flag+=1
#                     seg_mask[seg_mask==1]=inst_flag
#                     output_seg_[y_min:y_max,x_min:x_max] = seg_mask
#                     output_inst_info_dict[inst_flag]={}
#                     output_inst_info_dict[inst_flag]['type']=inst_info_['type']
#                     output_inst_info_dict[inst_flag]['type_prob']=inst_info_['type_prob']
#             else:
#                 inst_flag+=1
#                 seg_mask[seg_mask==1]=inst_flag
#                 output_seg_[y_min:y_max,x_min:x_max] = seg_mask
#                 output_inst_info_dict[inst_flag]={}
#                 output_inst_info_dict[inst_flag]['type']=inst_info_['type']
#                 output_inst_info_dict[inst_flag]['type_prob']=inst_info_['type_prob']
#         output_seg[patch_info_[0]:patch_info_[0]+384,patch_info_[1]:patch_info_[1]+384]=output_seg_
        
#     # 裁切并计算该张图中的每个实例的inst_info中的bbox centroid contour
#     pred_inst_map=output_seg[106:106+512,106:106+512]
#     pred_inst_info_dict={}
#     instance_flag=0
#     pred_map=np.zeros_like(pred_inst_map)
#     for inst_id in np.unique(pred_inst_map):
        
#         if inst_id==0: continue
#         instance_flag+=1
#         inst_mask =(pred_inst_map == inst_id).astype(np.uint8)
#         contours,_ = cv2.findContours(inst_mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if contours:
#             contour = contours[0]  # 选择主轮廓
#         else:
#             contour = None
#         x, y, w, h = cv2.boundingRect(inst_mask)
#         centroid_y, centroid_x = measurements.center_of_mass(inst_mask)
#         pred_map[pred_inst_map==inst_id]=instance_flag

#         instance_info = {
#             "bbox":[[y,x],[y+h,x+w]],
#             "centroid":[centroid_x,centroid_y],
#             "contour":contour,
#         }
#         instance_info.update(output_inst_info_dict[inst_id])
#         pred_inst_info_dict[instance_flag]=instance_info

#     src_shape = image_info["src_shape"]
    
#     # crop back to original shape
#     # 将拼接好的patch剪裁为原来大小
#     pred_map = np.squeeze(pred_inst[: src_shape[0], : src_shape[1]])

#     overlaid_img = visualize_instances_dict(
#         src_image.copy(), inst_info_dict, **overlay_kwargs
#     )

#     return image_info["name"], pred_inst, pred_map, pred_inst_info_dict, overlaid_img
# ##以下为myney版本
# def _post_process_patches(post_proc_func, post_proc_kwargs, patch_info, image_info, overlay_kwargs):
#     src_image = image_info["src_image"]
#     patch_info = sorted(patch_info, key=lambda x: [x[0][0], x[0][1]])
#     patch_info, patch_data = zip(*patch_info)

#     win_size = 384
#     step_size = 172
#     output_inst_info_dict = {}
#     output_seg = np.zeros((1006, 1006), dtype=np.int32)
#     inst_flag = 0

#     for patch_id, pred_patch in enumerate(patch_data):
#         pred_patch = np.squeeze(pred_patch)
#         pred_inst, inst_info_dict = post_proc_func(pred_patch, **post_proc_kwargs)
#         patch_info_ = patch_info[patch_id]

#         h, w = pred_inst.shape[:2]
#         y0, x0 = patch_info_[0], patch_info_[1]
#         output_seg_ = output_seg[y0:y0+h, x0:x0+w]

#         for inst_id in np.unique(pred_inst):
#             if inst_id == 0:
#                 continue
#             inst_info_ = inst_info_dict[inst_id]
#             y_min = inst_info_['bbox'][0][0]
#             x_min = inst_info_['bbox'][0][1]
#             y_max = inst_info_['bbox'][1][0]
#             x_max = inst_info_['bbox'][1][1]

#             h_target = y_max - y_min
#             w_target = x_max - x_min

#             # 边界检查：当前 patch 的最大可写区域
#             h_max = output_seg_.shape[0] - y_min
#             w_max = output_seg_.shape[1] - x_min
#             h_crop = min(h_target, h_max)
#             w_crop = min(w_target, w_max)

#             seg_mask = pred_inst[y_min:y_min+h_crop, x_min:x_min+w_crop]
#             seg_mask_out = output_seg_[y_min:y_min+h_crop, x_min:x_min+w_crop]

#             seg_mask[seg_mask != inst_id] = 0
#             seg_mask[seg_mask > 0] = 1

#             values, counts = np.unique(seg_mask_out, return_counts=True)
#             if counts.shape[0] > 1:
#                 values = values[1:]
#                 counts = counts[1:]
#                 most_frequent_value = values[np.argmax(counts)]
#                 seg_mask_out_bin = np.where(seg_mask_out == most_frequent_value, 1, 0)

#                 intersection = np.sum(seg_mask & seg_mask_out_bin)
#                 union = np.sum(seg_mask | seg_mask_out_bin)
#                 iou = intersection / union if union != 0 else 0

#                 if iou > 0.7:
#                     if output_inst_info_dict[most_frequent_value]["type_prob"] > inst_info_["type_prob"]:
#                         continue
#                     else:
#                         output_seg[output_seg == most_frequent_value] = 0
#                         output_seg_[output_seg_ == most_frequent_value] = 0

#                         seg_mask_inter = np.zeros_like(seg_mask, dtype=int)
#                         seg_mask_inter[seg_mask & seg_mask_out_bin] = most_frequent_value
#                         output_seg_[y_min:y_min+h_crop, x_min:x_min+w_crop] = seg_mask_inter

#                         output_inst_info_dict[most_frequent_value]['type'] = inst_info_['type']
#                         output_inst_info_dict[most_frequent_value]['type_prob'] = inst_info_['type_prob']
#                 else:
#                     inst_flag += 1
#                     seg_mask[seg_mask == 1] = inst_flag
#                     output_seg_[y_min:y_min+h_crop, x_min:x_min+w_crop] = seg_mask
#                     output_inst_info_dict[inst_flag] = {
#                         'type': inst_info_['type'],
#                         'type_prob': inst_info_['type_prob']
#                     }
#             else:
#                 inst_flag += 1
#                 seg_mask[seg_mask == 1] = inst_flag
#                 output_seg_[y_min:y_min+h_crop, x_min:x_min+w_crop] = seg_mask
#                 output_inst_info_dict[inst_flag] = {
#                     'type': inst_info_['type'],
#                     'type_prob': inst_info_['type_prob']
#                 }

#         output_seg[y0:y0+h, x0:x0+w] = output_seg_

#     pred_inst_map = output_seg[106:106+512, 106:106+512]
#     pred_inst_info_dict = {}
#     instance_flag = 0
#     pred_map = np.zeros_like(pred_inst_map)

#     for inst_id in np.unique(pred_inst_map):
#         if inst_id == 0:
#             continue
#         instance_flag += 1
#         inst_mask = (pred_inst_map == inst_id).astype(np.uint8)
#         contours, _ = cv2.findContours(inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         contour = contours[0] if contours else None
#         x, y, w, h = cv2.boundingRect(inst_mask)
#         centroid_y, centroid_x = measurements.center_of_mass(inst_mask)
#         pred_map[pred_inst_map == inst_id] = instance_flag

#         instance_info = {
#             "bbox": [[y, x], [y+h, x+w]],
#             "centroid": [centroid_x, centroid_y],
#             "contour": contour,
#         }
#         instance_info.update(output_inst_info_dict[inst_id])
#         pred_inst_info_dict[instance_flag] = instance_info

#     src_shape = image_info["src_shape"]
#     overlaid_img = visualize_instances_dict(
#         src_image.copy(), inst_info_dict, **overlay_kwargs
#     )

#     return image_info["name"], pred_inst, pred_map, pred_inst_info_dict, overlaid_img


# class InferManager(base.InferManager):
#     """Run inference on tiles."""

#     ####
#     def process_file_list(self, run_args):
#         """
#         Process a single image tile < 5000x5000 in size.
#         """
#         for variable, value in run_args.items():
#             self.__setattr__(variable, value)
#         assert self.mem_usage < 1.0 and self.mem_usage > 0.0

#         # * depend on the number of samples and their size, this may be less efficient
#         # 读取infer数据集list
#         patterning = lambda x: re.sub("([\[\]])", "[\\1]", x)
#         file_path_list = glob.glob(patterning("%s/*" % self.input_dir))
#         file_path_list.sort()  # ensure same order
#         assert len(file_path_list) > 0, 'Not Detected Any Files From Path'

#         ####### 1.创建文件夹
#         rm_n_mkdir(self.output_dir + '/json/')
#         rm_n_mkdir(self.output_dir + '/mat/')
#         rm_n_mkdir(self.output_dir + '/overlay/')
#         if self.save_qupath:
#             rm_n_mkdir(self.output_dir + "/qupath/")

#         def proc_callback(results):
#             """Post processing callback.
#             后处理程序callback
#             Output format is implicit assumption, taken from `_post_process_patches`
#             输出处理
#             """
#             img_name, pred_map, pred_inst, inst_info_dict, overlaid_img = results

#             nuc_val_list = list(inst_info_dict.values())
#             # need singleton to make matlab happy
#             nuc_uid_list = np.array(list(inst_info_dict.keys()))[:,None]
#             nuc_type_list = np.array([v["type"] for v in nuc_val_list])[:,None]
#             nuc_coms_list = np.array([v["centroid"] for v in nuc_val_list])

#             mat_dict = {
#                 "inst_map" : pred_inst,
#                 "inst_uid" : nuc_uid_list,
#                 "inst_type": nuc_type_list,
#                 "inst_centroid": nuc_coms_list
#             }
#             if self.nr_types is None: # matlab does not have None type array
#                 mat_dict.pop("inst_type", None) 

#             if self.save_raw_map:
#                 mat_dict["raw_map"] = pred_map
#             save_path = "%s/mat/%s.mat" % (self.output_dir, img_name)
#             sio.savemat(save_path, mat_dict)

#             save_path = "%s/overlay/%s.png" % (self.output_dir, img_name)
#             cv2.imwrite(save_path, cv2.cvtColor(overlaid_img, cv2.COLOR_RGB2BGR))

#             if self.save_qupath:
#                 nuc_val_list = list(inst_info_dict.values())
#                 nuc_type_list = np.array([v["type"] for v in nuc_val_list])
#                 nuc_coms_list = np.array([v["centroid"] for v in nuc_val_list])
#                 save_path = "%s/qupath/%s.tsv" % (self.output_dir, img_name)
#                 convert_format.to_qupath(
#                     save_path, nuc_coms_list, nuc_type_list, self.type_info_dict
#                 )

#             save_path = "%s/json/%s.json" % (self.output_dir, img_name)
#             self.__save_json(save_path, inst_info_dict, None)
#             return img_name

#         def detach_items_of_uid(items_list, uid, nr_expected_items):
#             item_counter = 0
#             detached_items_list = []
#             remained_items_list = []
#             while True:
#                 pinfo, pdata = items_list.pop(0)
#                 pinfo = np.squeeze(pinfo)
#                 if pinfo[-1] == uid:
#                     detached_items_list.append([pinfo, pdata])
#                     item_counter += 1
#                 else:
#                     remained_items_list.append([pinfo, pdata])
#                 if item_counter == nr_expected_items:
#                     break
#             # do this to ensure the ordering
#             remained_items_list = remained_items_list + items_list
#             return detached_items_list, remained_items_list

#         # 创建进程池
#         proc_pool = None
#         if self.nr_post_proc_workers > 0:
#             proc_pool = ProcessPoolExecutor(self.nr_post_proc_workers)

#         while len(file_path_list) > 0:
#             # 获取当前系统的可用内存，并基于一个内存使用率的比例来计算可以使用的内存
#             # virtual_memory 函数获取当前系统的内存使用情况
#             hardware_stats = psutil.virtual_memory()
#             available_ram = getattr(hardware_stats, "available")
#             # 将可用内存乘以 self.mem_usage，计算出实际可以使用的内存
#             available_ram = int(available_ram * self.mem_usage)
#             # available_ram >> 20 for MB, >> 30 for GB

#             # TODO: this portion looks clunky but seems hard to detach into separate func

#             # * caching N-files into memory such that their expected (total) memory usage
#             # * does not exceed the designated percentage of currently available memory
#             # * the expected memory is a factor w.r.t original input file size and
#             # * must be manually provided
#             # 将需要infer的图读入内存中
#             file_idx = 0
#             use_path_list = []
#             cache_image_list = []
#             cache_patch_info_list = []
#             cache_image_info_list = []
#             ############# 2.循环处理每一个图片，file_path是当前这一张，file_path_list是整个列表
#             while len(file_path_list) > 0:
#                 file_path = file_path_list.pop(0)

#                 img = cv2.imread(file_path)
#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                 src_shape = img.shape
#                 ############ 3.切成patch，img是填充后的图片，patch_info是每个path对应的信息[列像素，行像素，列数，行数，图片id]
#                 img, patch_info, top_corner = _prepare_patching(
#                     img, self.patch_input_shape, self.patch_output_shape, True
#                 )

#                 self_idx = np.full(patch_info.shape[0], file_idx, dtype=np.int32)
#                 patch_info = np.concatenate([patch_info, self_idx[:, None]], axis=-1)
#                 # ? may be expensive op
#                 patch_info = np.split(patch_info, patch_info.shape[0], axis=0)
#                 patch_info = [np.squeeze(p) for p in patch_info]

#                 # * this factor=5 is only applicable for HoVerNet
#                 expected_usage = sys.getsizeof(img) * 5
#                 available_ram -= expected_usage
#                 if available_ram < 0:
#                     break

#                 file_idx += 1
#                 # use_path_list为当前切过patch的图片，cache_image_list存储填充过的图片，cache_image_list为存储patch信息，cache_image_info_list存储图片的对应信息[[图片大小]，patch数,原本的左上顶点在填充后的图里的坐标]
#                 # if file_idx == 4: break
#                 use_path_list.append(file_path)
#                 cache_image_list.append(img)
#                 cache_patch_info_list.extend(patch_info)
#                 # TODO: refactor to explicit protocol
#                 cache_image_info_list.append([src_shape, len(patch_info), top_corner])

#             # * apply neural net on cached data
#             dataset = SerializeFileList(
#                 cache_image_list, cache_patch_info_list, self.patch_input_shape
#             )

#             dataloader = data.DataLoader(
#                 dataset,
#                 num_workers=self.nr_inference_workers,
#                 batch_size=self.batch_size,
#                 drop_last=False,
#             )

#             pbar = tqdm.tqdm(
#                 desc="Process Patches",
#                 leave=True,
#                 total=int(len(cache_patch_info_list) / self.batch_size) + 1,
#                 ncols=80,
#                 ascii=True,
#                 position=0,
#             )
#             from tensorboardX import SummaryWriter
#             writer = SummaryWriter(log_dir='logs/infer')
#             accumulated_patch_output = []
#             ######### 4.对每个batch进行预测
#             for batch_idx, batch_data in enumerate(dataloader):
#                 sample_data_list, sample_info_list = batch_data
#                 sample_output_list = self.run_step(sample_data_list)

#                 start_h = (self.patch_input_shape - self.patch_output_shape) // 2
#                 end_h = start_h + self.patch_output_shape
#                 start_w = (self.patch_input_shape - self.patch_output_shape) // 2
#                 end_w = start_w + self.patch_output_shape

#                 # 裁剪中心部分
#                 sample_output_list = sample_output_list[:, start_h:end_h, start_w:end_w, :]

#                 ########### 重叠部分处理


#                 # writer.add_image(f'output', torch.tensor(sample_output_list))
#                 sample_info_list = sample_info_list.numpy()
#                 curr_batch_size = sample_output_list.shape[0]
#                 sample_output_list = np.split(
#                     sample_output_list, curr_batch_size, axis=0
#                 )
#                 # for i, img_tensor in enumerate(sample_output_list):
#                 #     # 去除批次维度，将形状调整为 [384, 384, 5]
#                 #     img_tensor = torch.tensor(img_tensor).squeeze(0)
#                 #     img_viz = torch.zeros(160,960)
#                 #     # 遍历每个通道
#                 #     for c in range(img_tensor.shape[2]):
#                 #         # 选择第 c 个通道，形状为 [384, 384]
                
#                 #         img_viz[0:160,160*c:160*(c+1)]= img_tensor[:, :, c]
                
#                 #         # 添加通道维度，将形状调整为 [1, 384, 384]
#                 #         img_viz = img_viz.unsqueeze(0)
                
#                 #         # 将形状调整为 [384, 384, 1]，以符合 add_image 的输入要求
#                 #         # channel_img = channel_img.permute(1, 2, 0)
                
#                 #         # 添加图片到 TensorBoard
#                 #     writer.add_image(f'batch_{batch_idx}_image_{i}_channel_{c}', img_viz)
#                 sample_info_list = np.split(sample_info_list, curr_batch_size, axis=0)
#                 sample_output_list = list(zip(sample_info_list, sample_output_list))
#                 #accumulated_patch_output存储了所有的预测后的patch及每个patch所属图片和patch坐标
#                 accumulated_patch_output.extend(sample_output_list)
#                 pbar.update()
#             pbar.close()

#             ######## 5.选出一张图的所有patch进行拼接和后处理
#             # * parallely assemble the processed cache data for each file if possible
#             future_list = []
#             # use_path_list 存储图片filename
#             for file_idx, file_path in enumerate(use_path_list):
#                 image_info = cache_image_info_list[file_idx]
#                 file_ouput_data, accumulated_patch_output = detach_items_of_uid(
#                     accumulated_patch_output, file_idx, image_info[1]
#                 )

#                 # * detach this into func and multiproc dispatch it
#                 src_pos = image_info[2]  # src top left corner within padded image
#                 src_image = cache_image_list[file_idx]
#                 src_image = src_image[
#                     src_pos[0] : src_pos[0] + image_info[0][0],
#                     src_pos[1] : src_pos[1] + image_info[0][1],]

#                 base_name = pathlib.Path(file_path).stem
#                 file_info = {
#                     "src_shape": image_info[0],
#                     "src_image": src_image,
                    
#                     "name": base_name,
#                 }

#                 post_proc_kwargs = {
#                     "nr_types": self.nr_types,
#                     "return_centroids": True,
#                 }  # dynamicalize this

#                 overlay_kwargs = {
#                     "draw_dot": self.draw_dot,
#                     "type_colour": self.type_info_dict,
#                     "line_thickness": 2,
#                 }
#                 func_args = (
#                     self.post_proc_func,
#                     post_proc_kwargs,
#                     file_ouput_data,
#                     file_info,
#                     overlay_kwargs,
#                 )

#                 ####### 6.对选出一张图的所有patch调用_post_process_patches进行处理
#                 # dispatch for parallel post-processing
#                 if proc_pool is not None:
#                     proc_future = proc_pool.submit(_post_process_patches, *func_args)
#                     # ! manually poll future and call callback later as there is no guarantee
#                     # ! that the callback is called from main thread
#                     future_list.append(proc_future)
#                 else:
#                     proc_output = _post_process_patches(*func_args)
#                     ###### 10.调用proc_callback,存储对应的文件
#                     proc_callback(proc_output)

#             if proc_pool is not None:
#                 # loop over all to check state a.k.a polling
#                 for future in as_completed(future_list):
#                     # TODO: way to retrieve which file crashed ?
#                     # ! silent crash, cancel all and raise error
#                     if future.exception() is not None:
#                         log_info("Silent Crash")
#                         # ! cancel somehow leads to cascade error later
#                         # ! so just poll it then crash once all future
#                         # ! acquired for now
#                         # for future in future_list:
#                         #     future.cancel()
#                         # break
#                     else:
#                         file_path = proc_callback(future.result())
#                         log_info("Done Assembling %s" % file_path)
#         return

import logging
import multiprocessing
from multiprocessing import Lock, Pool

multiprocessing.set_start_method("spawn", True)  # ! must be at top for VScode debugging
import argparse
import glob
import json
import math
import multiprocessing as mp
import os
import pathlib
import pickle
import re
import sys
import warnings
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, as_completed, wait
from functools import reduce
from importlib import import_module
from multiprocessing import Lock, Pool
from run_utils.callbacks.serialize import fig2data, serialize
import cv2
import numpy as np
import psutil
import scipy.io as sio
import torch
import torch.utils.data as data
import tqdm
from dataloader.infer_loader import SerializeFileList
from misc.utils import (
    color_deconvolution,
    cropping_center,
    get_bounding_box,
    log_debug,
    log_info,
    rm_n_mkdir,
)
from misc.viz_utils import colorize, visualize_instances_dict
from skimage import color

import convert_format
from . import base


####
def _prepare_patching(img, window_size, mask_size, return_src_top_corner=False):
    """Prepare patch information for tile processing.
    为infer步骤extract patch
    Args:
        img: original input image
        window_size: input patch size
        mask_size: output patch size
        return_src_top_corner: whether to return coordiante information for top left corner of img
        
    """

    win_size = window_size
    msk_size = step_size = mask_size

    def get_last_steps(length, msk_size, step_size):
        nr_step = math.ceil((length - msk_size) / step_size)
        last_step = (nr_step + 1) * step_size
        return int(last_step), int(nr_step + 1)

    im_h = img.shape[0]
    im_w = img.shape[1]

    last_h, _ = get_last_steps(im_h, msk_size, step_size)
    last_w, _ = get_last_steps(im_w, msk_size, step_size)

    diff = win_size - step_size
    padt = padl = diff // 2
    padb = last_h + win_size - im_h
    padr = last_w + win_size - im_w

    img = np.lib.pad(img, ((padt, padb), (padl, padr), (0, 0)), "reflect")

    # generating subpatches index from orginal
    coord_y = np.arange(0, last_h, step_size, dtype=np.int32)
    coord_x = np.arange(0, last_w, step_size, dtype=np.int32)
    row_idx = np.arange(0, coord_y.shape[0], dtype=np.int32)
    col_idx = np.arange(0, coord_x.shape[0], dtype=np.int32)
    coord_y, coord_x = np.meshgrid(coord_y, coord_x)
    row_idx, col_idx = np.meshgrid(row_idx, col_idx)
    coord_y = coord_y.flatten()
    coord_x = coord_x.flatten()
    row_idx = row_idx.flatten()
    col_idx = col_idx.flatten()
    #
    patch_info = np.stack([coord_y, coord_x, row_idx, col_idx], axis=-1)
    if not return_src_top_corner:
        return img, patch_info
    else:
        return img, patch_info, [padt, padl]


####
def _post_process_patches(post_proc_func, post_proc_kwargs, patch_info, image_info, overlay_kwargs,
):
    """Apply post processing to patches.

    Args:
        post_proc_func: post processing function to use 使用的后处理函数
        post_proc_kwargs: keyword arguments used in post processing function 后处理函数中使用的关键参数
        patch_info: patch data and associated information
        image_info: input image data and associated information
        overlay_kwargs: overlay keyword arguments

    """
    # re-assemble the prediction, sort according to the patch location within the original image
    patch_info = sorted(patch_info, key=lambda x: [x[0][0], x[0][1]])
    patch_info, patch_data = zip(*patch_info)

    src_shape = image_info["src_shape"]
    src_image = image_info["src_image"]

    patch_shape = np.squeeze(patch_data[0]).shape
    ch = 1 if len(patch_shape) == 2 else patch_shape[-1]
    axes = [0, 2, 1, 3, 4] if ch != 1 else [0, 2, 1, 3]

    nr_row = max([x[2] for x in patch_info]) + 1
    nr_col = max([x[3] for x in patch_info]) + 1
    # 将预测出的patch拼接好
    pred_map = np.concatenate(patch_data, axis=0)
    pred_map = np.reshape(pred_map, (nr_row, nr_col) + patch_shape)
    pred_map = np.transpose(pred_map, axes)
    pred_map = np.reshape(
        pred_map, (patch_shape[0] * nr_row, patch_shape[1] * nr_col, ch)
    )
    # crop back to original shape
    # 将拼接好的patch剪裁为原来大小
    pred_map = np.squeeze(pred_map[: src_shape[0], : src_shape[1]])

    # * Implicit protocol
    # * a prediction map with instance of ID 1-N
    # * and a dict contain the instance info, access via its ID
    # * each instance may have type

    ######## 7.调用后处理函数对预测输出的进行处理 post_proc.py文件内容
    pred_inst, inst_info_dict = post_proc_func(pred_map, **post_proc_kwargs)

    overlaid_img = visualize_instances_dict(
        src_image.copy(), inst_info_dict, **overlay_kwargs
    )

    return image_info["name"], pred_map, pred_inst, inst_info_dict, overlaid_img


class InferManager(base.InferManager):
    """Run inference on tiles."""

    ####
    def process_file_list(self, run_args):
        """
        Process a single image tile < 5000x5000 in size.
        """
        for variable, value in run_args.items():
            self.__setattr__(variable, value)
        assert self.mem_usage < 1.0 and self.mem_usage > 0.0

        # * depend on the number of samples and their size, this may be less efficient
        # 读取infer数据集list
        patterning = lambda x: re.sub("([\[\]])", "[\\1]", x)
        file_path_list = glob.glob(patterning("%s/*" % self.input_dir))
        file_path_list.sort()  # ensure same order
        assert len(file_path_list) > 0, 'Not Detected Any Files From Path'

        ####### 1.创建文件夹
        rm_n_mkdir(self.output_dir + '/json/')
        rm_n_mkdir(self.output_dir + '/mat/')
        rm_n_mkdir(self.output_dir + '/overlay/')
        if self.save_qupath:
            rm_n_mkdir(self.output_dir + "/qupath/")

        def proc_callback(results):
            """Post processing callback.
            后处理程序callback
            Output format is implicit assumption, taken from `_post_process_patches`
            输出处理
            """
            img_name, pred_map, pred_inst, inst_info_dict, overlaid_img = results

            nuc_val_list = list(inst_info_dict.values())
            # need singleton to make matlab happy
            nuc_uid_list = np.array(list(inst_info_dict.keys()))[:,None]
            nuc_type_list = np.array([v["type"] for v in nuc_val_list])[:,None]
            nuc_coms_list = np.array([v["centroid"] for v in nuc_val_list])

            mat_dict = {
                "inst_map" : pred_inst,
                "inst_uid" : nuc_uid_list,
                "inst_type": nuc_type_list,
                "inst_centroid": nuc_coms_list
            }
            if self.nr_types is None: # matlab does not have None type array
                mat_dict.pop("inst_type", None) 

            if self.save_raw_map:
                mat_dict["raw_map"] = pred_map
            save_path = "%s/mat/%s.mat" % (self.output_dir, img_name)
            sio.savemat(save_path, mat_dict)

            save_path = "%s/overlay/%s.png" % (self.output_dir, img_name)
            cv2.imwrite(save_path, cv2.cvtColor(overlaid_img, cv2.COLOR_RGB2BGR))

            if self.save_qupath:
                nuc_val_list = list(inst_info_dict.values())
                nuc_type_list = np.array([v["type"] for v in nuc_val_list])
                nuc_coms_list = np.array([v["centroid"] for v in nuc_val_list])
                save_path = "%s/qupath/%s.tsv" % (self.output_dir, img_name)
                convert_format.to_qupath(
                    save_path, nuc_coms_list, nuc_type_list, self.type_info_dict
                )

            save_path = "%s/json/%s.json" % (self.output_dir, img_name)
            self.__save_json(save_path, inst_info_dict, None)
            return img_name

        def detach_items_of_uid(items_list, uid, nr_expected_items):
            item_counter = 0
            detached_items_list = []
            remained_items_list = []
            while True:
                pinfo, pdata = items_list.pop(0)
                pinfo = np.squeeze(pinfo)
                if pinfo[-1] == uid:
                    detached_items_list.append([pinfo, pdata])
                    item_counter += 1
                else:
                    remained_items_list.append([pinfo, pdata])
                if item_counter == nr_expected_items:
                    break
            # do this to ensure the ordering
            remained_items_list = remained_items_list + items_list
            return detached_items_list, remained_items_list

        # 创建进程池
        proc_pool = None
        if self.nr_post_proc_workers > 0:
            proc_pool = ProcessPoolExecutor(self.nr_post_proc_workers)

        while len(file_path_list) > 0:
            # 获取当前系统的可用内存，并基于一个内存使用率的比例来计算可以使用的内存
            # virtual_memory 函数获取当前系统的内存使用情况
            hardware_stats = psutil.virtual_memory()
            available_ram = getattr(hardware_stats, "available")
            # 将可用内存乘以 self.mem_usage，计算出实际可以使用的内存
            available_ram = int(available_ram * self.mem_usage)
            # available_ram >> 20 for MB, >> 30 for GB

            # TODO: this portion looks clunky but seems hard to detach into separate func

            # * caching N-files into memory such that their expected (total) memory usage
            # * does not exceed the designated percentage of currently available memory
            # * the expected memory is a factor w.r.t original input file size and
            # * must be manually provided
            # 将需要infer的图读入内存中
            file_idx = 0
            use_path_list = []
            cache_image_list = []
            cache_patch_info_list = []
            cache_image_info_list = []
            ############# 2.循环处理每一个图片，file_path是当前这一张，file_path_list是整个列表
            while len(file_path_list) > 0:
                file_path = file_path_list.pop(0)

                img = cv2.imread(file_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                src_shape = img.shape
                ############ 3.切成patch，img是填充后的图片，patch_info是每个path对应的信息[列像素，行像素，列数，行数，图片id]
                img, patch_info, top_corner = _prepare_patching(
                    img, self.patch_input_shape, self.patch_output_shape, True
                )

                self_idx = np.full(patch_info.shape[0], file_idx, dtype=np.int32)
                patch_info = np.concatenate([patch_info, self_idx[:, None]], axis=-1)
                # ? may be expensive op
                patch_info = np.split(patch_info, patch_info.shape[0], axis=0)
                patch_info = [np.squeeze(p) for p in patch_info]

                # * this factor=5 is only applicable for HoVerNet
                expected_usage = sys.getsizeof(img) * 5
                available_ram -= expected_usage
                if available_ram < 0:
                    break

                file_idx += 1
                # use_path_list为当前切过patch的图片，cache_image_list存储填充过的图片，cache_image_list为存储patch信息，cache_image_info_list存储图片的对应信息[[图片大小]，patch数,原本的左上顶点在填充后的图里的坐标]
                # if file_idx == 4: break
                use_path_list.append(file_path)
                cache_image_list.append(img)
                cache_patch_info_list.extend(patch_info)
                # TODO: refactor to explicit protocol
                cache_image_info_list.append([src_shape, len(patch_info), top_corner])

            # * apply neural net on cached data
            dataset = SerializeFileList(
                cache_image_list, cache_patch_info_list, self.patch_input_shape
            )

            dataloader = data.DataLoader(
                dataset,
                num_workers=self.nr_inference_workers,
                batch_size=self.batch_size,
                drop_last=False,
            )

            pbar = tqdm.tqdm(
                desc="Process Patches",
                leave=True,
                total=int(len(cache_patch_info_list) / self.batch_size) + 1,
                ncols=80,
                ascii=True,
                position=0,
            )
            from tensorboardX import SummaryWriter
            writer = SummaryWriter(log_dir='logs/infer')
            accumulated_patch_output = []
            ######### 4.对每个batch进行预测
            for batch_idx, batch_data in enumerate(dataloader):
                sample_data_list, sample_info_list = batch_data
                sample_output_list = self.run_step(sample_data_list)

                start_h = (self.patch_input_shape - self.patch_output_shape) // 2
                end_h = start_h + self.patch_output_shape
                start_w = (self.patch_input_shape - self.patch_output_shape) // 2
                end_w = start_w + self.patch_output_shape

                # 裁剪中心部分
                sample_output_list = sample_output_list[:, start_h:end_h, start_w:end_w, :]

                # writer.add_image(f'output', torch.tensor(sample_output_list))
                sample_info_list = sample_info_list.numpy()
                curr_batch_size = sample_output_list.shape[0]
                sample_output_list = np.split(
                    sample_output_list, curr_batch_size, axis=0
                )
                # for i, img_tensor in enumerate(sample_output_list):
                #     # 去除批次维度，将形状调整为 [384, 384, 5]
                #     img_tensor = torch.tensor(img_tensor).squeeze(0)
                #     img_viz = torch.zeros(160,960)
                #     # 遍历每个通道
                #     for c in range(img_tensor.shape[2]):
                #         # 选择第 c 个通道，形状为 [384, 384]
                
                #         img_viz[0:160,160*c:160*(c+1)]= img_tensor[:, :, c]
                
                #         # 添加通道维度，将形状调整为 [1, 384, 384]
                #         img_viz = img_viz.unsqueeze(0)
                
                #         # 将形状调整为 [384, 384, 1]，以符合 add_image 的输入要求
                #         # channel_img = channel_img.permute(1, 2, 0)
                
                #         # 添加图片到 TensorBoard
                #     writer.add_image(f'batch_{batch_idx}_image_{i}_channel_{c}', img_viz)
                sample_info_list = np.split(sample_info_list, curr_batch_size, axis=0)
                sample_output_list = list(zip(sample_info_list, sample_output_list))
                #accumulated_patch_output存储了所有的预测后的patch及每个patch所属图片和patch坐标
                accumulated_patch_output.extend(sample_output_list)
                pbar.update()
            pbar.close()

            ######## 5.选出一张图的所有patch进行拼接和后处理
            # * parallely assemble the processed cache data for each file if possible
            future_list = []
            # use_path_list 存储图片filename
            for file_idx, file_path in enumerate(use_path_list):
                image_info = cache_image_info_list[file_idx]
                file_ouput_data, accumulated_patch_output = detach_items_of_uid(
                    accumulated_patch_output, file_idx, image_info[1]
                )

                # * detach this into func and multiproc dispatch it
                src_pos = image_info[2]  # src top left corner within padded image
                src_image = cache_image_list[file_idx]
                src_image = src_image[
                    src_pos[0] : src_pos[0] + image_info[0][0],
                    src_pos[1] : src_pos[1] + image_info[0][1],]

                base_name = pathlib.Path(file_path).stem
                file_info = {
                    "src_shape": image_info[0],
                    "src_image": src_image,
                    "name": base_name,
                }

                post_proc_kwargs = {
                    "nr_types": self.nr_types,
                    "return_centroids": True,
                }  # dynamicalize this

                overlay_kwargs = {
                    "draw_dot": self.draw_dot,
                    "type_colour": self.type_info_dict,
                    "line_thickness": 2,
                }
                func_args = (
                    self.post_proc_func,
                    post_proc_kwargs,
                    file_ouput_data,
                    file_info,
                    overlay_kwargs,
                )

                ####### 6.对选出一张图的所有patch调用_post_process_patches进行处理
                # dispatch for parallel post-processing
                if proc_pool is not None:
                    proc_future = proc_pool.submit(_post_process_patches, *func_args)
                    # ! manually poll future and call callback later as there is no guarantee
                    # ! that the callback is called from main thread
                    future_list.append(proc_future)
                else:
                    proc_output = _post_process_patches(*func_args)
                    ###### 10.调用proc_callback,存储对应的文件
                    proc_callback(proc_output)

            if proc_pool is not None:
                # loop over all to check state a.k.a polling
                for future in as_completed(future_list):
                    # TODO: way to retrieve which file crashed ?
                    # ! silent crash, cancel all and raise error
                    if future.exception() is not None:
                        log_info("Silent Crash")
                        # ! cancel somehow leads to cascade error later
                        # ! so just poll it then crash once all future
                        # ! acquired for now
                        # for future in future_list:
                        #     future.cancel()
                        # break
                    else:
                        file_path = proc_callback(future.result())
                        log_info("Done Assembling %s" % file_path)
        return

