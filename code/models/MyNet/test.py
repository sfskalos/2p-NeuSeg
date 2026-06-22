from targets import gen_targets
import numpy as np
import matplotlib.pyplot as plt

file_path ='/home/ZXIAO/datasets/Neure_512/iii1_000.npy'
data = np.load(file_path)
ann = data[:,:,3:]
inst_type = ann[...,0]
dict = gen_targets(img=data[:,:,0:3],ann=inst_type,crop_shape=[320,320])
plt.imshow(dict['og_map'], cmap='gray')  # cmap='gray' 表示使用灰度颜色映射
plt.colorbar()  # 显示颜色条
plt.title('Random Array Visualization')
plt.show()