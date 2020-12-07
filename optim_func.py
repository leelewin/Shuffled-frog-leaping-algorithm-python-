# -*- coding: utf-8 -*-
#####################################################
#  Fn5: DeJong's F5 function
#  论文附录F5函数定义错误,见图片
#
#####################################################
import math

def fn5(x1, x2):
    '''this two-dimensional function contain 25 foxholes of various depths surrounded
    by a relatively flat surface.

    >>>fn5(-32,-32)
    0.998
    '''
    base = [-32.0, -16.0, 0.0, 16.0, 32.0]
    a = [[base[i % 5] for i in range(25)], 
         [base[j // 5] for j in range(25)]]
    sum = 0
    for x in range(25):
        sum += 1 / (x+1 + math.pow((x1-a[0][x]), 6) + math.pow((x2-a[1][x]), 6))
    
    return 1 / (0.002 + sum)

#########################################################
#      Gear problem (Deb and Goyal 1997)
#       
#       -12 <= x1,x2,x3,x4 <= 60
# 
#########################################################        

def gear(x1, x2, x3, x4):
    pass




def nestedFn5(parameter):
    return fn5(parameter[0], parameter[1])












