# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 21:06:11 2022

@author: Métaturgiste
"""

import time
import random
import math as m
import matplotlib.pyplot as plt
import numpy as np
import sys

J = 1
K = -0.35
ki = []
ji = []
dt = 0.1

def popin():
    return random.random()*2-1

def initpop(n):
    x = [[popin()] for i in range(n)]
    y = [[popin()] for i in range(n)]
    o = [[popin()*m.pi] for i in range(n)]
    global ki, ji
    ki = [K for i in range(n)]
    ji = [J for i in range(n)]
    #print('yup', x, y, o, 'yay')
    return x, y, o

def dist(x, y, i, j, t):
    d = m.sqrt((x[j][t]-x[i][t])**2+(y[j][t]-y[i][t])**2)
    if d == 0:
        return 1*10**-9
    return d

def sumpartx(x, y, o, i, j, t):
    return (x[j][t]-x[i][t])/dist(x,y,i,j,t)*(1+ji[i]*m.cos(o[j][t]-o[i][t]))-(x[j][t]-x[i][t])/(dist(x,y,i,j,t)**2)

def sumparty(x, y, o, i, j, t):
    return (y[j][t]-y[i][t])/dist(x,y,i,j,t)*(1+ji[i]*m.cos(o[j][t]-o[i][t]))-(y[j][t]-y[i][t])/(dist(x,y,i,j,t)**2)

def sumparto(x, y, o, i, j, t):
    return ki[i]*m.sin(o[j][t]-o[i][t])/dist(x,y,i,j,t)

def forw(x, y, o, n):
    t = len(x[0]) - 1
    for i in range(n):
        x[i].append(x[i][-1] + 1/n * sum([sumpartx(x, y, o, i, j, t) for j in range(n)]) * dt)
        y[i].append(y[i][-1] + 1/n * sum([sumparty(x, y, o, i, j, t) for j in range(n)]) * dt)
        o[i].append(o[i][-1] + 1/n * sum([sumparto(x, y, o, i, j, t) for j in range(n)]) * dt)
    return x, y, o

def simu(T, n):
    x, y, o = initpop(n)
    for t in range(T):
        forw(x, y, o, n)
        print('This is turn : ', t)
    return x, y, o

def pwety(x, y, o, s):
    ax = plt.axes(projection ='3d')
    
    n = len(x)
    T = len(x[0])
    time = [i*dt for i in range(T)]
    for i in range(n):
        ax.plot3D(time, x[i], y[i])
    plt.savefig(s)
    plt.show()
    plt.close()

def exply(x, y, o, s):
    n = len(x)
    for i in range(n):
        plt.plot([x[i][-1], x[i][-1]+0.03*m.cos(o[i][-1])], [y[i][-1],y[i][-1]+0.03*m.sin(o[i][-1])])
    plt.savefig(s)
    plt.show()
    plt.close()

def sep(y, n):
    global ki, ji
    top = [[], []]
    mid = [[], []]
    bot = [[], []]
    for i in range(n):
        if y[i][-1] >= 0.5:
            top[0].append(ki[i])
            top[1].append(ji[i])
        elif y[i][-1] <= -0.5:
            bot[0].append(ki[i])
            bot[1].append(ji[i])
        else :
            mid[0].append(ki[i])
            mid[1].append(ji[i])
    fig, axs = plt.subplots()
    axs.hist(top[0])
    axs.hist(top[1])
    plt.savefig("HistTop.jpeg")
    plt.show()
    plt.close()
    
    fig, axs = plt.subplots()
    axs.hist(mid[0])
    axs.hist(mid[1])
    plt.savefig("HistMid.jpeg")
    plt.show()
    plt.close()
    
    fig, axs = plt.subplots()
    axs.hist(bot[0])
    axs.hist(bot[1])
    plt.savefig("HistBot.jpeg")
    plt.show()
    plt.close()
    
    return
    
def whr(x, y, o, s):
    n = len(x)
    hi = []
    for i in range(n):
        hi.append(x[i][-1])
        print(x[i][-1])
    
    ax = plt.axes(projection ='3d')
    ax.plot_trisurf(np.asarray(ji), np.asarray(ki), np.asarray(hi),  cmap ='viridis', edgecolor ='green')
    plt.savefig("ThePropreties")

def main(k, j):
    global K, J
    K = (k-3)*0.44
    J = (j-3)*0.44
    n = 1000
    x, y, o = simu(4000, n)
    pwety(x, y, o, "Hist for k " +  str(K) + " and J "+ str(J) + ".jpeg")
    exply(x, y, o, "Last for k " +  str(K) + " and J "+ str(J) + ".jpeg")
    #sep(y, n)
    whr(x, y, o, "Rep for K's and J's")


i,j = sys.argv[1:]
i,j = int(i),int(j)


a = time.process_time()
main(i, j)
b = time.process_time()
print(b-a)