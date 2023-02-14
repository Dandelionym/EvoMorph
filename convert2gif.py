# -*- coding: utf-8 -*-
"""
    INFORMATION TITLE
    -----------------------------------------------------------------
    AUTHOR: Mellen Y.Pu
    DATE: 2023/2/9 下午5:22
    FILE: convert2gif.py
    PROJ: EvoMorph
    IDE: PyCharm
    EMAIL: yingmingpu@gmail.com
    ----------------------------------------------------------------- 
                                      ONE DOOR OPENS ALL THE WINDOWS.

    @INTRODUCTION: 
     - 
    @FUNCTIONAL EXPLATION:
     - 
    @LAUNCH:
     - 
"""
import imageio
import os


def get_file_list(file_path):
    dir_list = os.listdir(file_path)
    if not dir_list:
        return
    else:
        # 注意，这里使用lambda表达式，将文件按照最后修改时间顺序升序排列
        # os.path.getmtime() 函数是获取文件最后修改时间
        # os.path.getctime() 函数是获取文件最后创建时间
        dir_list = sorted(dir_list,  key=lambda x: os.path.getmtime(os.path.join(file_path, x)))
        # print(dir_list)
        return dir_list


BASE_DIR = 'exp_03'

gif_images = []
image_list = get_file_list(BASE_DIR)
print(image_list)



for i in range(len(image_list)):
    gif_images.append(imageio.imread(os.path.join(BASE_DIR, image_list[i])))

imageio.mimsave(f"evolution_{BASE_DIR}.gif", gif_images, fps=5)