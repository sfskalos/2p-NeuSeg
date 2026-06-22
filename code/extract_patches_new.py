import os

import numpy as np
import torch
import shutil
import pathlib
from dataset import get_dataset

if __name__ == "__main__":
    type_classification = True
    save_root = "/home/ZXIAO/datasets/Neure_512/training_data_512_512"
    dataset_name = "neure"
    dataset_info = {
            "train": {
                "img": (".tif", "/home/ZXIAO/datasets/Neure_512/train/images/"),
                "ann": (".mat", "/home/ZXIAO/datasets/Neure_512/train/labels/"),
                    },
            "valid": {
                "img": (".tif", "/home/ZXIAO/datasets/Neure_512/valid/images/"),
                "ann": (".mat", "/home/ZXIAO/datasets/Neure_512/valid/labels/"),
            },
    }

    def rm_n_mkdir(dir_path):
        """创建文件夹"""
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
        return os.makedirs(dir_path)

    parser = get_dataset(dataset_name)

    for split_name, split_desc in dataset_info.items():
        img_extension, img_dir = split_desc["img"]
        ann_extension, ann_dir = split_desc["ann"]
        out_dir = "%s/%s/"%(save_root,split_name)
        # 读取文件列表并排序，确保顺序一样
        file_list = os.listdir(img_dir)
        file_list.sort()
        rm_n_mkdir(out_dir)
        print(split_name)
        for file_idx,file_path in enumerate(file_list):
            base_name = pathlib.Path(file_path).stem
            img = parser.load_img("%s/%s"%(img_dir,file_path))
            ann = parser.load_ann("%s%s%s"%(ann_dir,base_name,ann_extension),type_classification)

            try:
                #导入时为[H,W,C],网络输入为[N,C,H,W]
                img = np.concatenate([img,ann],axis=-1)
                img = torch.Tensor(img)
                np.save("%s/%s.npy"%(out_dir,base_name),img)
            except:
                print(file_path+"错误")