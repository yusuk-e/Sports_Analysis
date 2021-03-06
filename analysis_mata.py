# -*- coding:utf-8 -*-

import pdb

from time import time
import datetime as dt
import numpy as np
from scipy.special import gammaln
import matplotlib
#matplotlib.use('Agg') #DISPLAYの設定
import matplotlib.pyplot as plt
import resource
import codecs
import random
from collections import defaultdict
from collections import namedtuple

#variable---------------------------------------------------------------------------------
action_dic = []
t_dic = []
elem = namedtuple("elem", "t, a, x, y")
#サンプルNo, アクションID, x座標，y座標

xmax = -10 ** 14
xmin = 10 ** 14
ymax = -10 ** 14
ymin = 10 ** 14

new_xmax = float(2 * xmax) + 1 #x=xmaxのときint()が30を超えてしまうので+1しておく
new_ymax = float(2 * ymax) + 1
Dx = 30 #x方向のメッシュ分割数
Dy = 20 #y方向のメッシュ分割数
M_Team0 = np.zeros([Dy, Dx])
M_Team1 = np.zeros([Dy, Dx])
#-----------------------------------------------------------------------------------------


#Input------------------------------------------------------------------------------------
counter = 0
t0 = time()
filename = "processed_metadata.csv"
D = defaultdict(int)
#アクションID付きボール位置データ D[t]

fin = open(filename)
counter = 0
for row in fin:
    temp = row.rstrip("\r\n").split(",")
    t = int(temp[0])

    if t not in t_dic:
        t_dic.append(t)

    action = int(temp[2])
    if action not in action_dic:
        action_dic.append(action)

    x = float(temp[4])
    if xmin > x:
        xmin = x
    if xmax < x:
        xmax = x

    y = float(temp[5])
    if ymin > y:
        ymin = y
    if ymax < y:
        ymax = y

    f = elem(t, action, x, y)
    D[counter] = f
    counter += 1

fin.close()
print "time:%f" % (time()-t0)
#-----------------------------------------------------------------------------------------


#-----とりあえず前半
#N = 300
N = counter - 1
#-----


#ボール軌跡-------------------------------------------------------------------------------
t0 = time()
XY = np.zeros([N,2])
for n in range(N):
    XY[n][0] = D[n].x
    XY[n][1] = D[n].y
    if D[n].a == 

#ボール全軌跡----------------------
#fig = plt.figure()
#plt.plot(XY[:,0], XY[:,1])
#plt.axis([-155, 155, -100, 100])
#plt.show()
#----------------------------------


print 'time:%f' % (time()-t0)
#-----------------------------------------------------------------------------------------

'''
    for i in range(len(temp1) - 1):
        temp2 = temp1[i].rstrip("\r\n").split(",")
        Team_id = int(temp2[0])
        sys_id = int(temp2[1])
        Player_id = int(temp2[2])
        for j in range(len(temp2)):
            if j == 1:
                if sys_id not in sys_id_dic:
                    sys_id_dic.append(sys_id)
            elif j == 2 and Team_id == 0:
                if Player_id not in Player_Team0_dic:
                    Player_Team0_dic.append(Player_id)
            elif j == 2 and Team_id == 1:
                if Player_id not in Player_Team1_dic:
                    Player_Team1_dic.append(Player_id)

        new_x = float(temp2[3])-xmin #原点をコートの左下に
        new_y = float(temp2[4])-ymin

        Mx_id = int( Dx * new_x / new_xmax ) #メッシュidを計算
        #if Mx_id > 29 or Mx_id < 0:
        #    pdb.set_trace()
        My_id = Dy - 1 - int( Dy * new_y / new_ymax ) #左上がメッシュid=0になるように反転
        #if My_id < 0 or My_id > 20:
        #    pdb.set_trace()
        M_id = My_id * Dx + Mx_id

        f = Foo_p(new_x, new_y, float(temp2[5]), Mx_id, My_id, M_id)
        D[frame_id][Team_id][Player_id] = f

        if f.x < 0 or f.y < 0:
            print err
        #if f.x > xmax:
        #    xmax = f.x
        #if f.x < xmin:
        #    xmin = f.x
        #if f.y > ymax:
        #    ymax = f.y
        #if f.y < ymin:
        #    ymin = f.y

    temp3 = temp0[2].rstrip("\r\n").split(";")
    temp3 = temp3[0].rstrip("\r\n").split(",")

    for j in range(len(temp3) - 1):
        
        if j == 4:
            if temp3[j] == "A":
                frag = 0
            elif temp3[j] == "H":
                frag = 1
            else:
                print "err"
            temp3[j] = frag

    if len(temp3) == 6:
        f = Foo_b(float(temp3[0]), float(temp3[1]), float(temp3[2]), float(temp3[3]), int(temp3[4]), temp3[5], "")
    elif len(temp3) == 7:
        f = Foo_b(float(temp3[0]), float(temp3[1]), float(temp3[2]), float(temp3[3]), int(temp3[4]), temp3[5], temp3[6])
        info = temp3[6]
        if info not in info_dic:
            info_dic.append(info)
    else:
        print "err"

    B[frame_id] = f

    #if f.x > xmax:
    #    xmax = f.x
    #if f.x < xmin:
    #    xmin = f.x
    #if f.y > ymax:
    #    ymax = f.y
    #if f.y < ymin:
    #    ymin = f.y

fin.close()

print "time:%f" % (time()-t0)
#-----------------------------------------------------------------------------------------


#pdb.set_trace()
#今だけ--------------
frame_id_max = int(frame_id_max - (frame_id_max - frame_id_min + 1) / 2) - 2000 #適当に前半だけ分析
#--------------


#プレイヤー毎のヒートマップ作成-----------------------------------------------
t0 = time()

D_Team0 = defaultdict(int) #選手の位置情報 D[Player_id]
D_Team1 = defaultdict(int) #選手の位置情報 D[Player_id]
N = frame_id_max - frame_id_min + 1
for i in range(N):
    n = i + frame_id_min
    x_Team0 = D[n][0]
    N_Player_Team0 = len(Player_Team0_dic)

    for j in range(N_Player_Team0):
    #for j in range(2):
        Player_Team0_id = Player_Team0_dic[j]
        x = x_Team0[Player_Team0_id]

        if np.size(D_Team0[Player_Team0_id]) == 1: #D_Team0にまだデータがない選手の場合
            if np.size(x) == 6: #xが構造体を持つ場合
                f = np.array([n, x.x, x.y, x.v, x.mx, x.my, x.m])
                D_Team0[Player_Team0_id] = f

        else:
            if np.size(x) == 6:#xが構造体を持つ場合
                f = np.array([n, x.x, x.y, x.v, x.mx, x.my, x.m])
                D_Team0[Player_Team0_id] = np.vstack([D_Team0[Player_Team0_id],f])


for j in range(N_Player_Team0):
#for j in range(2):
    Player_Team0_id = Player_Team0_dic[j]
    if np.size(D_Team0[Player_Team0_id]) != 1:
        X = D_Team0[Player_Team0_id][:,1]
        Y = D_Team0[Player_Team0_id][:,2]
        fig = plt.figure()
        plt.plot(X,Y)
        plt.axis([0.0,new_xmax,0.0,new_ymax])
        plt.savefig('Tracking/Tracking_Team'+str(0)+'_Player'+str(Player_Team0_id)+'.png')
        plt.close()

        Mx = D_Team0[Player_Team0_id][:,4]
        My = D_Team0[Player_Team0_id][:,5]
        M = np.zeros([Dy, Dx])
        for k in range(len(Mx)):
            M[My[k],Mx[k]] += 1

        fig = plt.figure()
        plt.imshow(M)
        plt.savefig('Mesh/M_Team0'+'_Player'+str(Player_Team0_id)+'.png')
        plt.close()


print "time:%f" % (time()-t0)
#-----------------------------------------------------------------------------

    
#{ユーザ，場所}*時間の行列作成-----------------------------------------------
t0 = time()



print "time:%f" % (time()-t0)
#-----------------------------------------------------------------------------
'''


pdb.set_trace()
