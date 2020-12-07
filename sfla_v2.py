# -*- coding: utf-8 -*-
#  V2 change point:将python的列表操作全部替换为np的数组操作
#####################################################
#  混合蛙跳算法解决Fn5
#  Fn5: DeJong's F5 function
#  -66 <= x1, x2 <= 66
#####################################################
import numpy as np
import random
import matplotlib.pyplot as plt

from optim_func import fn5

#根据实际问题和经验确定相关参数
#社群的个数
memeplexes_number = 100
#每个社群中青蛙的个数
frog_n = 30
#整个蛙群的数量
total = memeplexes_number * frog_n
#进化的次数
circulation_N = 30
#子群的大小
submemep_q = 20
#最大步长                              #增加一个维度变量增加抽象
step_max = [1,1]   #未用
#全局最大循环次数
total_eval = 100


def globalSearch(func, dimension=2, low_limit=-66, up_limit=66, accuracy=0.0001):
    '''进行全局搜索

    Argument:
        func        {obj}   -- 目标函数
        dimension   {int}   -- 决策变量的个数
        low_limit   {int}   -- 可行域下界 
        up_limit    {int}   -- 可行域上界
        accuracy    {float} -- 收敛精度  
    Return:
    '''
    #初始化青蛙群
    populations = initialPopula(dimension, low_limit, up_limit) 
    #初始化社群  例：memplexes[0]代表第一个社群
    memeplexes = np.zeros((memeplexes_number, frog_n, dimension))
    #初始化社群适应度
    mem_fitness = np.zeros((memeplexes_number, frog_n))
    
    # while 收敛精度满足 或达到最大循环次数 stop , 否则循环
    evalution = 0
    show_fitness = []
    while(evalution < total_eval):
        #适应度数组  适应度越小越好
        fitness = calculateFitness(fn5, populations) 
        #将蛙群以适应度降序，并记录全局最好青蛙的位置
        populations =  populations[np.argsort(fitness)]
        fitness.sort()

        #全局最好青蛙的位置
        frog_g = populations[0]
        print("The best frog's position and fitness: ", frog_g, fitness[0])
        #画图
        show_fitness.append(fitness[0]) 
        plot(populations, show_fitness)
        # print(global_best_position)
        
        #用花式索引解决会更优雅
        #划分社群
        for j in range(frog_n):
            for k in range(memeplexes_number):
                    memeplexes[k][j] = (populations[k + memeplexes_number * j])
                    mem_fitness[k][j] = (fitness[k + memeplexes_number * j])

        #调用局部搜索函数, 返回更新后的社群
        #可以使用多线程进行加速
        memeplexes = localSearch(memeplexes, mem_fitness, frog_g)
        #社群混洗  
        populations = memeplexes.reshape(total, dimension)

        evalution += 1

def initialPopula(dim, low, up, population=True):
    '''初始化群体,产生的都是整数

    Arguemnt:
        dim     {int}   --  决策变量的个数
        low     {int}   --  可行域的下界
        up      {int}   --  可行域的上界
        population {bool} -- 产生个体or群体
    Return:
                {ndarray}   -- 随机初始化后的群体or个体
    '''
    if population:
        return np.random.randint(low, up+1, size=(total, dim))
    else:
        return np.random.randint(low, up+1, size=(1, dim))


#这个函数需要重写
def calculateFitness(func, param, dim=2):
    '''计算适应度

    Argument:
        func    {callable}   -- 适应度函数
        param   {ndarray}   -- 用于计算的值
        dim     {int}   -- 决策变量的数
    Return:
                {ndarray} -- 各个青蛙的适应度
    '''
    fit_func = np.frompyfunc(func, dim, 1)
    if param.ndim == 1:
        return fit_func(param[0], param[1])
    if param.ndim == 2:
        return fit_func(param[:,0],param[:,1])


def localSearch(memeplexes, mem_fitness, global_best):
    '''对划分好的社群执行局部搜索
    Argument:
        memeplexes  {ndarray}   -- 
        mem_fitness {ndarray}   -- 
        global_best {ndarray}   -- 
    Return:
        memeplexes  {ndarray}   -- 更新完的所有社群
    '''
    #当前社群编号
    im = 0
    while im < memeplexes_number:
        #当前进化次数
        iN = 0
        while iN < circulation_N:
            submemep = constructSubmemep(memeplexes[im])
            #子群的适应度     
            sub_fitness = calculateFitness(fn5, submemep)
            #将子群以适应度降序，并记录子群最好\最坏青蛙的位置
            submemep = submemep[np.argsort(sub_fitness)]
            sub_fitness.sort()
            sub_best = submemep[0]
            sub_worst = submemep[-1] 
            new_position = updateWorst(sub_best, sub_worst, global_best)
            #最坏青蛙在相应社群的位置
            # index =  mem_fitness[im].index(sub_fitness[-1])
            index = np.where(mem_fitness[im] == sub_fitness[-1])
            #更新社群中青蛙及适应度
            memeplexes[im][index] = new_position 
            mem_fitness[im][index] = calculateFitness(fn5, new_position) 

            iN += 1
        im += 1

    return memeplexes

def constructSubmemep(current_memep):
    '''用轮盘赌的方式构造当前社群的子群

    Argument:
        current_memep   {ndarray}   -- 社群

    Return:
            {ndarray}   -- 子群（根据当前社群构造的）
    '''
    #转盘赌选择submemep_q个个体 概率和为1 
    select_prob = [2*(frog_n - j) / (frog_n * (frog_n + 1)) for j in range(frog_n)]
    wheel_select = np.cumsum(select_prob)
    submemep_index = []   #用集合实现呢
    i = 0
    while i < submemep_q:
        rand = random.random()
        for j in range(frog_n):
            if rand < wheel_select[j]:
                seleted = j 
                break
        #个体不能重复
        if seleted not in submemep_index:
            submemep_index.append(j)
        else:
            i -= 1
        i += 1
    return current_memep[submemep_index]

def updateWorst(local_best, local_wrost, global_best, low_limit=-66, up_limit=66):
    '''更新最差青蛙的位置
    先通过局部最优来提高最差青蛙的位置，还不行，则通过全局最优来提高

    Arguments:
        local_best {ndarray} -- 子群最好青蛙的位置
        local_wrost {ndarray}   -- 子群最差青蛙的位置
        global_best {ndarray}   -- 全局最好青蛙的位置

    Return:
        new {ndarray} -- 更新后的位置
    '''
    #局部最优来更新worst
    #暂时忽略step_max, 直接求
    step_size = random.random() * (local_best- local_wrost)
    new = (local_wrost + step_size)
    #如果不优或不在可行域中                      #这几个比较不够优雅，
    if int(new[0]) > up_limit or int(new[0]) < low_limit \
        or int(new[1]) > up_limit or int(new[1]) < low_limit \
            or calculateFitness(fn5, new) < calculateFitness(fn5, local_wrost):
        step_size = random.random() * (global_best - local_wrost)
        new = (local_wrost + step_size)
        #还不是最优的，则产生一个新的青蛙
        if int(new[0]) > up_limit or int(new[0]) < low_limit \
            or int(new[1]) > up_limit or int(new[1]) < low_limit \
                or calculateFitness(fn5, new) < calculateFitness(fn5, local_wrost):
            new = initialPopula(2, low_limit, up_limit, False)
    return new

def plot(populations, best_fitness):
    '''画图-进化动态以及进化曲线
    '''
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

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


if __name__ == '__main__':
    globalSearch(fn5,2,-66,66,0.0000000001)
