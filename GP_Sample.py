# -*- coding: utf-8 -*-
# @Time    : 2020-04-12 17:17
# @Author  : zxl
# @FileName: GP_Sample.py


import numpy as np
from scipy.interpolate import spline
import matplotlib.pyplot as plt

def gaussian_kernel(x,y,l=1):
    return np.exp(-(x-y)**2/(2*l*l))

def laplace_kernel(x,y,l=1):
    return np.exp(-abs(x-y)/(2*l*l))


def cal_cov(x,f):
    """
    给定x，以高斯核函数计算协方差矩阵
    :param x: 参数
    :param f: 核函数
    :return:
    """
    cov=[]
    for i in range(len(x)):
        cur_cov=[]
        for j in range(len(x)):
            v=f(x[i],x[j])
            cur_cov.append(v)
        cov.append(cur_cov)
    return np.array(cov)

def linear_kernel(x,y):
    return x*y

def zero_mean(x):
    """
    给定x，计算均值向量
    :param x: 向量
    :return:
    """
    return np.full(shape=(len(x),),fill_value=0)


def draw(x,Y,mean,title):
    idx=np.argsort(x)
    new_x=[x[i] for i in idx]
    for y in Y:
        new_y=[y[i] for i in idx]
        plt.plot(new_x,new_y)

    plt.xlabel('n')
    plt.ylabel('fn')
    plt.title(title)
    plt.show()


if __name__ == "__main__":

    a=-1#x范围
    b=1
    x=np.array(np.random.random(1000))
    x=[a+(b-a)*item for item in x]

    mean=zero_mean(x)
    cov=cal_cov(x,linear_kernel)

    n=10#采10组
    Y=np.random.multivariate_normal(mean,cov,(n,))
    title="Linear kernel "
    draw(x,Y,mean,title)





