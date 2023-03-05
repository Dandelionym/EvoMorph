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
        self.width = width
        self.area = np.zeros((width, width))
        self.grow_comp = np.ones((width, width))
        self.bridge_comp = np.ones((width, width))
        self.every_growth = 1.0
        self.miu = 0
        self.sigma = 16
        self.sigma_inhibt = 64
        self.alpha = 100
        self.comp_func = lambda x: self.alpha * np.exp(-(x - self.miu) ** 2 / (2 * self.sigma ** 2)) / (np.sqrt(2 * np.pi) * self.sigma)
        self.inhabit_func = lambda x: -1 * 10 * np.exp(-(x - self.miu) ** 2 / (2 * self.sigma_inhibt ** 2)) / (np.sqrt(2 * np.pi) * self.sigma_inhibt)
        self.iter = 0
        self.au_active_dist = 200
        self.ligand_limit_dist = 60
        self.au_speed = 4
        self.bridges = []
        self.line_length = 30

        # Status
        self.mem_bridge_lst = []

    def evolute(self):
        # FIND TWO RANDOM POINT PAIRS
        _mem_bridge_ = 1
        while True:
            x0, y0 = np.random.randint(0, self.width), np.random.randint(0, self.width)
            x1, y1 = x0 + np.random.randint(-self.line_length, self.line_length), y0 + np.random.randint(-self.line_length, self.line_length)
            if min([x0, x1, y0, y1]) > 0 and max([x0, x1, y0, y1]) < self.width and distance((x0, x1), (y0, y1)) > self.line_length * 0.5:
                self.bridges.append((x0, y0, x1, y1))
                break
            else:
                _mem_bridge_ += 1
        self.mem_bridge_lst.append(_mem_bridge_)


        # ADD BRIDGES BETWEEN TWO POINTS
        for m, n, p, q in self.bridges:
            # Two end points are here:
            for i in range(m - self.au_active_dist, m + self.au_active_dist):
                for j in range(n - self.au_active_dist, n + self.au_active_dist):
                    _distance_ = np.sqrt((i - m) ** 2 + (j - n) ** 2)
                    if _distance_ > 0 and (self.width > i > 0) and (0 < j < self.width):
                        if _distance_ < self.au_active_dist:
                            self.grow_comp[i, j] += self.comp_func(_distance_ * self.grow_comp[i, j] * 1.2)
            for i in range(p - self.au_active_dist, p + self.au_active_dist):
                for j in range(n - self.au_active_dist, q + self.au_active_dist):
                    _distance_ = np.sqrt((i - p) ** 2 + (j - q) ** 2)
                    if _distance_ > 0 and (self.width > i > 0) and (0 < j < self.width):
                        if _distance_ < self.au_active_dist:
                            self.grow_comp[i, j] += self.comp_func(_distance_ * self.grow_comp[i, j] * 1.2)


        random = np.random.randn(self.width ** 2).reshape(self.width, self.width)
        self.grow_comp += 0.0000005 * random
        self.alpha /= 1.2

        # mean_ = self.grow_comp.mean()
        # for i in range(len(self.grow_comp)):
        #     for j in range((len(self.grow_comp))):
        #         if self.grow_comp[i, j] < mean_ * 1.2:
        #             self.grow_comp[i, j ] /= 1 + self.grow_comp[i, j ]
        #         else:
        #             self.grow_comp[i, j] *= 1 + self.grow_comp[i, j ]

        self.grow_comp = min_max_norm(self.grow_comp)
        self.iter += 1


    def view_2d(self, title="", save="4", vrange=(-1, 1)):
        if vrange:
            plt.matshow(self.grow_comp, vmin=vrange[0], vmax=vrange[1])
        else:
            plt.matshow(self.grow_comp)
        # fig, axs = plt.subplots(2, 1)
        # axs[0].matshow(self.area)
        # axs[1].matshow(self.grow_comp)
        plt.title(title)
        plt.colorbar()
        if save is not None:
            os.makedirs(f'./exp_0{save}/', exist_ok=True)
            id_ = len(os.listdir(f'./exp_0{save}/')) + 1
            plt.savefig(f'./exp_0{save}/{id_}_{title}.jpeg')
        plt.show()
        plt.close()

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


    surface = Surface(600)
    surface.view_2d("Initalized State", save="8", vrange=(-1, 2))
    for _ in range(300):
        print(_, surface.mem_bridge_lst)
        time.sleep(0.1)
        surface.evolute()
        if _ % 1 == 0:
            surface.view_2d(f"Particle Surface Evolution Step: {_}", vrange=False, save="8")


