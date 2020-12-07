# -*- coding: utf-8 -*-
#####################################################
#  混合蛙跳算法解决Fn5
#  Fn5: DeJong's F5 function
#  -66 <= x1, x2 <= 66
#
#####################################################
import numpy as np
import random
import matplotlib.pyplot as plt

from optim_func import fn5

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

#根据实际问题和经验确定相关参数
#社群的个数
memeplexes_number = 100
#每个社群中青蛙的个数
frog_n = 30
#整个蛙群的数量
total = memeplexes_number * frog_n
#进化的次数
circulation_N = 30
#蛙群
populations = []
#社群  例：memplexes[0]代表第一个社群
memeplexes = [[] for _ in range(memeplexes_number)]
#适应度列表  适应度越小越好
fitness = []
#社群适应度
mem_fitness = [[] for _ in range(memeplexes_number)]
#全局最好青蛙的位置
frog_g = []
#子群的大小
submemep_q = 20
#最大步长                              #增加一个维度变量增加抽象
step_max = [1,1]   #待定
up_limit = 66
low_limit = -66
#收敛精度
accuracy = 0.0000000001
#全局最大循环次数
total_eval = 100

def globalSearch():
    #初始化青蛙群,并计算相应青蛙的适应度
    populations = [[random.randint(-66,66), random.randint(-66,66)] for _ in range(total)]
    
    # while 收敛精度满足 或达到最大循环次数 stop , 否则循环
    evalution = 0
    show_fitness = []
    while(evalution < total_eval):
        fitness = [fn5(x1, x2) for x1, x2 in populations]

        #将蛙群以适应度降序，并记录全局最好青蛙的位置
        zip_sorted = sorted(zip(populations, fitness), key=lambda x: (x[1],x[0]))#排序次序
        # print(zip_sorted)
        populations, fitness = [list(x) for x in zip(*zip_sorted)]
        global frog_g #全局变量修改
        frog_g = populations[0]
        print("The best frog's position and fitness: ", frog_g, fitness[0])
        #画图
        show_fitness.append(fitness[0]) 
        plot(populations, show_fitness)
        # print(global_best_position)

        #将蛙群划分为社群
        for j in range(frog_n):
            for k in range(memeplexes_number):
                memeplexes[k].append(populations[k + memeplexes_number * j])
                mem_fitness[k].append(fitness[k + memeplexes_number * j])

        #调用局部搜索函数
        #可以使用多线程进行加速
        localSearch()
        #社群混洗  np的reshape更快
        for i in range(memeplexes_number):
            for j in range(frog_n):
                populations[i * frog_n + j] =  memeplexes[i][j]

        evalution += 1

def localSearch():
    '''

    Argument:
    入参--所有社群
    出参 -- 更新完的所有社群
    '''
    #当前社群编号
    im = 0
    while im < memeplexes_number:
        #当前进化次数
        iN = 0
        while iN < circulation_N:
            submemep = constructSubmemep(memeplexes[im])
            #子群的适应度     都能重构
            sub_fitness = [fn5(x1, x2) for x1, x2 in submemep]
            #将子群以适应度降序，并记录子群最好\最坏青蛙的位置
            sub_sorted = sorted(zip(submemep, sub_fitness), key=lambda x: (x[1],x[0]))#排序次序
            submemep, sub_fitness = [list(x) for x in zip(*sub_sorted)]
            sub_best = submemep[0]
            sub_worst = submemep[-1] 
            new_position = updateWorst(sub_best, sub_worst)
            #最坏青蛙在相应社群的位置
            index =  mem_fitness[im].index(sub_fitness[-1])
            #更新社群中青蛙及适应度
            memeplexes[im][index] = [int(new_position[0]),int(new_position[1])]
            mem_fitness[im][index] = fn5(int(new_position[0]), int(new_position[1]))

            iN += 1
        im += 1


def constructSubmemep(current_memep):
    '''构造当前社群的子群
    '''
    #转盘赌选择submemep_q个个体 概率和为1 
    select_prob = [2*(frog_n - j) / (frog_n * (frog_n + 1)) for j in range(frog_n)]
    wheel_select = [sum(select_prob[:i+1]) for i in range(frog_n)]
    submemep = []
    i = 0
    while i < submemep_q:
        rand = random.random()
        for j in range(frog_n):
            if rand < wheel_select[j]:
                seleted = current_memep[j]
                break
        #个体不能重复
        if seleted not in submemep:
            submemep.append(seleted) 
        else:
            i -= 1
        i += 1
    return submemep  #最好返回q在社群中的对应位置

def updateWorst(local_best, local_wrost):
    '''更新最差青蛙的位置
    先通过局部最优来提高最差青蛙的位置，还不行，则通过全局最优来提高

    Arguments:
        local_best {list} -- 子群最好青蛙的位置

    Return:
        {list} -- 返回更新后的位置

    '''
    #局部最优来更新worst
    #暂时忽略step_max, 直接求
    # if diff = (local_best - local_wrost) > 0:
    #     step_size = min(int(random.random() * diff), step_max)
    # else:
    #     step_size = min(int(random.random() * diff), -step_max)
    # new = local_wrost + step_
    forg_b = np.array(local_best)
    forg_w = np.array(local_wrost)
    step_size = random.random() * (forg_b - forg_w)
    new = (forg_w + step_size).tolist()
    #如果不优或不在可行域中
    if int(new[0]) > up_limit or int(new[0]) < low_limit \
        or int(new[1]) > up_limit or int(new[1]) < low_limit \
            or fn5(int(new[0]), int(new[1])) < fn5(forg_w[0], forg_w[1]):
        step_size = random.random() * (np.array(frog_g) - forg_w)
        new = (forg_w + step_size).tolist()
        #还不是最优的，则产生一个新的青蛙
        if int(new[0]) > up_limit or int(new[0]) < low_limit \
            or int(new[1]) > up_limit or int(new[1]) < low_limit \
                or fn5(int(new[0]), int(new[1])) < fn5(forg_w[0], forg_w[1]):
            new = [random.randint(-66,66), random.randint(-66,66)]
    return new

def plot(populations, best_fitness):
    '''
    '''
    x1 = [] 
    x2 = [] 
    for i, j in populations:
        x1.append(i)
        x2.append(j)

    plt.ion()
    plt.figure(1)
    #进化动态图
    plt.subplot(121)
    plt.plot(x1, x2, 'r^')
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.title('SFLA进化动态')
    # plt.text(1,1, 'x=1')
    plt.axis([-66,66,-66,66])
    plt.grid(True)

    #进化曲线
    plt.subplot(122)
    plt.plot(best_fitness, 'g')
    plt.xlabel('number of evalution')
    plt.ylabel('best fitness value')
    plt.title('进化曲线')
    plt.axis([1,100,0,5])
    plt.draw() 
    plt.pause(0.8)
    # plt.ioff()
    # plt.close()
    plt.clf()
    # plt.show()

# def plot(populations, best_fitness):
#     '''
#     '''
#     x1 = [] 
#     x2 = [] 
#     for i, j in populations:
#         x1.append(i)
#         x2.append(j)
#     plot1(x1,x2)
#     plot2(best_fitness)

# def plot1(x1, x2):
#     plt.ion()
#     plt.figure(1)
#     #进化动态图
#     plt.plot(x1, x2, 'r^')
#     plt.ylabel('x2')
#     plt.xlabel('x1')
#     plt.title('SFLA进化动态')
#     plt.axis([-66,66,-66,66])
#     plt.grid(True)
#     plt.draw() 
#     plt.pause(0.8)
#     # plt.ioff()
#     # plt.close()
#     plt.clf()
#     # plt.show()

# def plot2(best):
#     plt.figure(1)
#     plt.plot([best], 'g-')
#     plt.xlabel('number of evalution')
#     plt.ylabel('best fitness value')
#     plt.title('进化曲线')
#     plt.pause(0.8)
#     # plt.ioff()
#     # plt.close()
#     plt.show()



if __name__ == '__main__':
    globalSearch()



























