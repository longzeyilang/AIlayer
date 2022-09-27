
import numpy as np
import math


def gene(APY_STEP, APY_COEF) :
    # 用于产生系数
    # APY_COEF 用于调整精度，数据越大，计算的误差越小
    coef = [0] *20
    coef [0]  = round(math.exp(int("0x58b91", 16)/APY_STEP)*APY_COEF)
    coef [1]  = round(math.exp(int("0x2c5c8", 16)/APY_STEP)*APY_COEF)
    coef [2]  = round(math.exp(int("0x162e4", 16)/APY_STEP)*APY_COEF)
    coef [3]  = round(math.exp(int("0x0b172", 16)/APY_STEP)*APY_COEF)
    coef [4]  = round(math.exp(int("0x067cd", 16)/APY_STEP)*APY_COEF)
    coef [5]  = round(math.exp(int("0x03920", 16)/APY_STEP)*APY_COEF)
    coef [6]  = round(math.exp(int("0x01e27", 16)/APY_STEP)*APY_COEF)
    coef [7]  = round(math.exp(int("0x00f85", 16)/APY_STEP)*APY_COEF)
    coef [8]  = round(math.exp(int("0x007e1", 16)/APY_STEP)*APY_COEF)
    coef [9]  = round(math.exp(int("0x003f8", 16)/APY_STEP)*APY_COEF)
    coef [10] = round(math.exp(int("0x001fe", 16)/APY_STEP)*APY_COEF)
    coef [11] = round(math.exp( int("0x100", 16)/APY_STEP)*APY_COEF)
    coef [12] = round(math.exp( int("0x080", 16)/APY_STEP)*APY_COEF)
    coef [13] = round(math.exp( int("0x040", 16)/APY_STEP)*APY_COEF)
    coef [14] = round(math.exp( int("0x020", 16)/APY_STEP)*APY_COEF)
    coef [15] = round(math.exp( int("0x010", 16)/APY_STEP)*APY_COEF)
    coef [16] = round(math.exp( int("0x008", 16)/APY_STEP)*APY_COEF)
    coef [17] = round(math.exp( int("0x004", 16)/APY_STEP)*APY_COEF)
    coef [18] = round(math.exp( int("0x002", 16)/APY_STEP)*APY_COEF)
    coef [19] = round(math.exp( int("0x001", 16)/APY_STEP)*APY_COEF)
    # for i in coef :
    #     print (bin(i))
    return coef

def gene_step ( APY_STEP):
    step = []
    step.append(hex(round(math.log(256    )*APY_STEP)) )
    step.append(hex(round(math.log(16     )*APY_STEP)) )
    step.append(hex(round(math.log(4      )*APY_STEP)) )
    step.append(hex(round(math.log(2      )*APY_STEP)) )
    step.append(hex(round(math.log(3/2    )*APY_STEP)) )
    step.append(hex(round(math.log(5/4    )*APY_STEP)) )
    step.append(hex(round(math.log(9/8    )*APY_STEP)) )
    step.append(hex(round(math.log(17/16  )*APY_STEP)) )
    step.append(hex(round(math.log(33/32  )*APY_STEP)) )
    step.append(hex(round(math.log(65/64  )*APY_STEP)) )
    step.append(hex(round(math.log(129/128)*APY_STEP)) )
    for i in step :
        print (i)
    return step

def shift(x,y, step,k, CONST):
    t = x-step
    if (t>=0) :
        x=t
        y= math.floor(y*k/CONST)
    return x,y

def cal_exp(x, CONST1, coef, APY_COEF):
    # 近似计算流程，采用累加方式实现
    # x迭代放大了16倍，输入最大x为10左右
    # 步进“0x58b91”为放大16倍后的数据，因此输入数据需要保持16位小数位
    y= CONST1
    x,y = shift(x,y,0x58b91, coef [0]  , APY_COEF)
    print(x,y,0x58b91,coef[0])
    x,y = shift(x,y,0x2c5c8, coef [1]  , APY_COEF)
    print(x,y,0x2c5c8,coef[1])
    x,y = shift(x,y,0x162e4, coef [2]  , APY_COEF)
    print(x,y,0x162e4,coef[2])
    x,y = shift(x,y,0x0b172, coef [3]  , APY_COEF)
    print(x,y,0x0b172,coef[3])
    x,y = shift(x,y,0x067cd, coef [4]  , APY_COEF)
    print(x,y,0x067cd,coef[4])
    x,y = shift(x,y,0x03920, coef [5]  , APY_COEF)
    print(x,y,0x03920,coef[5])
    x,y = shift(x,y,0x01e27, coef [6]  , APY_COEF)
    print(x,y,0x01e27,coef[6])
    x,y = shift(x,y,0x00f85, coef [7]  , APY_COEF)
    print(x,y,0x00f85,coef[7])
    x,y = shift(x,y,0x007e1, coef [8]  , APY_COEF)
    print(x,y,0x007e1,coef[8])
    x,y = shift(x,y,0x003f8, coef [9]  , APY_COEF)
    print(x,y,0x003f8,coef[9])
    x,y = shift(x,y,0x001fe, coef [10] , APY_COEF)
    print(x,y,0x001fe,coef[10])
    if(x&0x100) : y=math.floor (y*coef [11]/APY_COEF)
    if(x&0x080) : y=math.floor (y*coef [12]/APY_COEF)
    if(x&0x040) : y=math.floor (y*coef [13]/APY_COEF)
    if(x&0x020) : y=math.floor (y*coef [14]/APY_COEF)
    if(x&0x010) : y=math.floor (y*coef [15]/APY_COEF)
    if(x&0x008) : y=math.floor (y*coef [16]/APY_COEF)
    if(x&0x004) : y=math.floor (y*coef [17]/APY_COEF)
    if(x&0x002) : y=math.floor (y*coef [18]/APY_COEF)
    if(x&0x001) : y=math.floor (y*coef [19]/APY_COEF)
    return y

def cal_sig(x, flag):
    if (flag>0):
        y = 1-(1/(x+1))
    else :
        y = 1/(1+x)
    return y

def test(CONST1, APY_STEP, coef, APY_COEF):
    # 测试数据，从
    a = [(i)*10 for i in range(APY_STEP)]
    length = len(a)
    res = [0] * length
    res1 = [0] * length
    diff = [0] * length
    diff_sig_a= [0] * length
    diff_sig_b= [0] * length
    sig_a= [0] * length
    sig_b= [0] * length
    print (len(a), len(res), len(res1))
    diff_max = 0
    fexp = open("fexp.txt", "w")
    fsigma = open("fsigma.txt", "w")
    fsigmb = open("fsigmb.txt", "w")
    for i in range(length) :
        # i = 6000
        res[i] = cal_exp (a[i] , CONST1, coef, APY_COEF )
        fexp.write("%d \n"% res[i])
        res1[i] = math.exp(a[i]/APY_STEP)*CONST1
        diff[i] = abs(res[i]- res1[i])
        sig_a [i]=  cal_sig(res[i]/CONST1, 1)*(CONST1)
        sig_b [i]=  cal_sig(res[i]/CONST1, -1)*(CONST1)
        fsigma.write("%d \n"% sig_a[i] )
        fsigmb.write("%d \n"% sig_b[i] )
        diff_sig_a [i]= abs( cal_sig(res[i]/CONST1, 1) - cal_sig(res1[i]/CONST1, 1) )
        diff_sig_b [i]= abs( cal_sig(res[i]/CONST1, -1) - cal_sig(res1[i]/CONST1, -1) )
        # if (diff[i] > diff_max) :
        #     diff_max = diff[i]
        #     print (i,a[i],diff[i], res[i], res1[i])

    print  ("diff      ", max(diff), diff.index(max(diff)) )
    print  ("diff_sig_a", max(diff_sig_a), max(diff_sig_a)*(CONST1) )
    print  ("diff_sig_b", max(diff_sig_b), max(diff_sig_b)*(CONST1) )
    print  ("sig_a", max(sig_a)*(CONST1), min(sig_a)*(CONST1) )
    print  ("sig_b", max(sig_b)*(CONST1), min(sig_b)*(CONST1) )
    print  ("res", max(res)/(CONST1))
    print  ("res1", max(res1)/(CONST1))

# CONST1 代码计算结果的精度
CONST1 = 1<<13

# 中间运算结果精度扩展
APY_COEF = 1<<12
APY_STEP = 1<<16
coef = gene(APY_STEP,APY_COEF)
# gene_step(APY_STEP)

# test(CONST1, APY_STEP, coef, APY_COEF)
a = cal_exp(5000,CONST1,coef,APY_COEF)
print(1/(1+ math.exp(-5000/APY_STEP))*CONST1)
print(cal_sig(a/CONST1, -1)*(CONST1))
print(cal_sig(a/CONST1, 1)*(CONST1))