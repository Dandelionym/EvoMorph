# -*- coding: utf-8 -*-
"""
    INFORMATION TITLE
    -----------------------------------------------------------------
    AUTHOR: Mellen Y.Pu
    DATE: 2023/1/29 下午2:10
    FILE: chain_actiavtion_field.py
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
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def distance(a, b) -> float:
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def min_max_norm(X, feature_range=(0, 1)):
    r1, r2 = feature_range  # 将数据归一化到 [r1, r2] 之间
    xmin, xmax = X.min(), X.max() # 得到数据的最大最小值
    X_std = (X - xmin) / (xmax - xmin)     # 标准化到 [0, 1]
    return X_std

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Surface:
    def __init__(self, width):
        # Config:
        self.random_weight = 0
        self.normal_func = min_max_norm
        self.comp_alpha = 64

        self.width = width
        self.material_local_dist = int(width / 12)
        self.bridge_local_dist = int(width / 12)
        self.grow_comp = np.ones((width, width))
        self.before_norm = np.ones((width, width))
        self.miu = 0
        self.sigma = 64
        self.miu_bridge = 0
        self.sigma_bridge = 48
        self.comp_func = lambda x: self.comp_alpha * np.exp(-(x - self.miu) ** 2 / (self.sigma ** 2)) / (np.sqrt(2 * np.pi) * self.sigma)
        self.brdg_func = lambda x: self.comp_alpha * np.exp(-(x - self.miu) ** 2 / (self.sigma_bridge ** 2)) / (np.sqrt(2 * np.pi) * self.sigma_bridge)
        self.located_points = np.zeros((width, width))
        self.points = []
        self.iter = 0


    def evolute(self):
        # Status Initialization
        current_map = np.ones((self.width, self.width))

        # Step 1: Each iter have three times loop, select 3 points as active
        for _ in range(6):
            u = 1
            while True:
                x0, y0 = np.random.randint(0, self.width), np.random.randint(0, self.width)
                if self.width > x0 > 0 and self.width > y0 > 0:
                    break
                u += 1
            self.points.append((x0, y0))
            self.located_points[x0, y0] = 1
            print(f"{self.iter}-{_} :: U -> ", u)
            # Active area should have the normal status for growth
            for i in range(x0 - self.material_local_dist, x0 + self.material_local_dist):
                for j in range(y0 - self.material_local_dist, y0 + self.material_local_dist):
                    if (0 < i < self.width) and (0 < j < self.width):
                        single_dist = distance((i, j), (x0, y0))
                        # Two points, each of them will follow the general distance to grow.
                        if self.material_local_dist > single_dist > 0:
                            current_map[i, j] += 0.1 * self.comp_func(single_dist * current_map[i, j])

        # current_map = sigmoid(current_map)

        # Step 2: For each active point, it neighbors may affact itself, especially the direct bridge between them.
        for m, n in self.points:
            for p, q in self.points:
                _distance_ = np.sqrt((p - m) ** 2 + (q - n) ** 2)
                if 0 < _distance_ < self.bridge_local_dist:
                    # If two points are close, then select them, then this local area is treated as sub-active area (a bridge).
                    A = n - q
                    B = p - m
                    C = p * (q - n) + q * (m - p)
                    # Local area should have decreased active status.
                    for i in range(p, m):
                        for j in range(q, n):
                            dist_p2l = np.abs(A * i + B * j + C) / np.sqrt(A ** 2 + B ** 2)
                            if self.bridge_local_dist * 2 > dist_p2l > 0:
                                current_map[i, j] += (1/2) * self.brdg_func(dist_p2l * current_map[i, j])

        # Make random growth iteraly.
        self.before_norm = current_map
        random = np.random.randn(self.width ** 2).reshape(self.width, self.width)
        self.grow_comp += current_map * (1-self.random_weight) + random * self.random_weight
        self.grow_comp = self.normal_func(self.grow_comp)
        self.iter += 1


    def view_2d(self, title="", save="4", vrange=(-1, 1), single_plot=False):
        if single_plot:
            if vrange:
                plt.matshow(self.grow_comp, vmin=vrange[0], vmax=vrange[1])
            else:
                plt.matshow(self.grow_comp)
            plt.title(title)
            plt.colorbar()
            if save is not None:
                os.makedirs(f'./exp_0{save}/', exist_ok=True)
                id_ = len(os.listdir(f'./exp_0{save}/')) + 1
                plt.savefig(f'./exp_0{save}/{id_}_{title}.jpeg')
            plt.show()
            plt.close()
        else:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
            fig.suptitle(f'Mode: debug - Status Record - Round {self.iter}')
            ax1.matshow(self.grow_comp, vmin=0, vmax=1)
            ax2.plot([self.comp_func(x) for x in range(int(-self.width/5), int(self.width/5))])
            ax2.axis("off")
            ax3.matshow(self.located_points)
            ax4.matshow(self.before_norm)
            plt.show()

    def view_3d(self, title=""):
        """
            plot of the surfce with 3d mode.
        """
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        x = np.linspace(0, 10, self.width)
        y = np.linspace(0, 10, self.width)
        X, Y = np.meshgrid(x, y)
        Z = self.area
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.hot)
        if np.max(Z) < 100:
            ax.set_zlim(0, 100)
        elif np.max(Z) < 1000:
            ax.set_zlim(0, 1000)
        else:
            ax.set_zlim(0, 20000)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        plt.axis('off')
        plt.title(title)
        plt.show()



if __name__ == '__main__':
    import time
    from tqdm import tqdm

    surface = Surface(1000)
    surface.view_2d("Initalized State", save="9_zero1", vrange=(-1, 2))
    for _ in range(100):
        surface.evolute()
        if _ % 1 == 0:
            surface.view_2d(f"Particle Surface Evolution Step: {_}", vrange=False, save="9_zero1")


