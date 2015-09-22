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
team_dic = []
player1_dic = []
player2_dic = []
t_dic = []
elem = namedtuple("elem", "t, re_t, team, p, a, x, y")
#絶対時刻, ハーフ相対時間，チームID，アクションID, x座標，y座標

xmax = -10 ** 14
xmin = 10 ** 14
ymax = -10 ** 14
ymin = 10 ** 14
tmin = 0
tmax = 0

first_half_start = 0
first_half_end = 0
last_half_start = 0
last_half_end = 0

first_half_start_re_t = 0
first_half_end_re_t = 0
last_half_start_re_t = 0
last_half_end_re_t = 0

D = defaultdict(int)#アクションID付きボール位置データ D[counter]

Seq_Team1_shots = defaultdict(int)
Seq_Team2_shots = defaultdict(int)
N_Team1_shots = 0
N_Team2_shots = 0

Action_Team1_VitalArea = {}
Action_Team2_VitalArea = {}
N_Action_Team1_VitalArea = 0
N_Action_Team2_VitalArea = 0

new_xmax = float(2 * xmax) + 1 #x=xmaxのときint()が30を超えてしまうので+1しておく
new_ymax = float(2 * ymax) + 1
Dx = 30 #x方向のメッシュ分割数
Dy = 20 #y方向のメッシュ分割数
M_Team0 = np.zeros([Dy, Dx])
M_Team1 = np.zeros([Dy, Dx])
#-----------------------------------------------------------------------------------------


#Input------------------------------------------------------------------------------------
def input():
    global xmax, xmin, ymax, ymin
    global N
    global first_half_start, first_half_end, last_half_start, last_half_end
    counter = 0
    t0 = time()
    filename = "processed_metadata.csv"
    fin = open(filename)

    first_half_start = int(fin.readline().rstrip("\r\n"))
    first_half_end = int(fin.readline().rstrip("\r\n"))
    last_half_start = int(fin.readline().rstrip("\r\n"))
    last_half_end = int(fin.readline().rstrip("\r\n"))

    counter = 0
    for row in fin:
        temp = row.rstrip("\r\n").split(",")

        t = int(temp[0])
        if t not in t_dic:
            t_dic.append(t)

        re_t = int(temp[1])

        if t == first_half_start:
            pdb.set_trace()
            first_half_start_re_t = re_t
        if t == first_half_end:
            first_half_end_re_t = re_t
        if t == last_half_start:
            last_half_start_re_t = re_t
        if t == last_half_end:
            last_half_end_re_t = re_t

        team = int(temp[2])
        if team not in team_dic:
            team_dic.append(team)

        action = int(temp[3])
        if action not in action_dic:
            action_dic.append(action)

        player = int(temp[4])
        if team == 1:
            if player not in player1_dic:
                player1_dic.append(player)
        elif team == 2:
            if player not in player2_dic:
                player2_dic.append(player)
        else:
            print "err"

        x = float(temp[5])
        if xmin > x:
            xmin = x
        if xmax < x:
            xmax = x

        y = float(temp[6])
        if ymin > y:
            ymin = y
        if ymax < y:
            ymax = y

        f = elem(t, re_t, team, action, player, x, y)
        D[counter] = f
        counter += 1
    pdb.set_trace()
    fin.close()
    N = counter - 1
    print "time:%f" % (time()-t0)
#-----------------------------------------------------------------------------------------

#shots ボール軌跡-----------------------------------------------------------------------------
def Seq_Team_shots():
    global N_Team1_shots, N_Team2_shots
    t0 = time()
    for n in range(N):

        if D[n].a == 15:
            k = n
            team = D[n].team
            if team == 1:
                while k >= 0:
                    if D[k].team == 2:
                        N_Team1_shots += 1
                        break

                    x = D[k]
                    if np.size(Seq_Team1_shots[N_Team1_shots]) == 1: 
                        #Seq_Team1_shotsにまだデータがない場合
                        f = np.array([x.t, x.team, x.p, x.a, x.x, x.y])
                        Seq_Team1_shots[N_Team1_shots] = f
                    else:
                        if D[k+1].t - D[k].t > 10:
                            N_Team1_shots += 1
                            break

                        f = np.array([x.t, x.team, x.p, x.a, x.x, x.y])
                        Seq_Team1_shots[N_Team1_shots] = np.vstack([f, Seq_Team1_shots[N_Team1_shots]])
                    k -= 1
            
            elif team == 2:
                while k >= 0:
                    if D[k].team == 1:
                        N_Team2_shots += 1
                        break

                    x = D[k]
                    if np.size(Seq_Team2_shots[N_Team2_shots]) == 1: 
                        #Seq_Team2_shotsにまだデータがない場合
                        f = np.array([x.t, x.team, x.p, x.a, x.x, x.y])
                        Seq_Team2_shots[N_Team2_shots] = f
                    else:
                        if D[k+1].t - D[k].t > 10:
                            N_Team2_shots += 1
                            break

                        f = np.array([x.t, x.team, x.p, x.a, x.x, x.y])
                        Seq_Team2_shots[N_Team2_shots] = np.vstack([f, Seq_Team2_shots[N_Team2_shots]])
                    k -= 1
            else:
                print "err"


    for s in range(N_Team1_shots):
        S = Seq_Team1_shots[s]
        if np.size(S) > 6:
            Start_time = int(S[0,0])
            X = S[:,4]
            Y = S[:,5]
            fig = plt.figure()
            plt.plot(X,Y)
            plt.scatter(X,Y)
            plt.axis([-155, 155, -100, 100])
            if Start_time < first_half_end:#右方向に攻撃
                x = np.arange(15.0, 85.0, 0.005)
                y0 = np.ones(len(x)) * -60
                y1 = np.ones(len(x)) * 60
            elif Start_time > last_half_start:#左方向に攻撃
                x = np.arange(-85.0, -15.0, 0.005)
                y0 = np.ones(len(x)) * -60
                y1 = np.ones(len(x)) * 60
            plt.fill_between(x, y0, y1, edgecolor = 'none', facecolor = 'blue', alpha = 0.2)
            plt.savefig('Seq_Team1_shots/Seq_Team1_shots'+'_no'+str(s)+'.png')
            plt.close()


    for s in range(N_Team2_shots):
        S = Seq_Team2_shots[s]
        if np.size(S) > 6:
            Start_time = int(S[0,0])
            X = S[:,4]
            Y = S[:,5]
            fig = plt.figure()
            plt.plot(X,Y)
            plt.scatter(X,Y)
            plt.axis([-155, 155, -100, 100])
            if Start_time < first_half_end:
                x = np.arange(-85.0, -15.0, 0.005)#左方向に攻撃
                y0 = np.ones(len(x)) * -60
                y1 = np.ones(len(x)) * 60
            elif Start_time > last_half_start:
                x = np.arange(15.0, 85.0, 0.005)#右方向に攻撃
                y0 = np.ones(len(x)) * -60
                y1 = np.ones(len(x)) * 60
            plt.fill_between(x, y0, y1, edgecolor = 'none', facecolor = 'blue', alpha = 0.2)
            plt.savefig('Seq_Team2_shots/Seq_Team2_shots'+'_no'+str(s)+'.png')
            plt.close()

    print 'time:%f' % (time()-t0)
#-----------------------------------------------------------------------------------------

#VitalArea 行動-----------------------------------------------------------------------------
def Action_Plot_VitalArea():
    global Action_Team1_VitalArea, Action_Team2_VitalArea

    t0 = time()
    for n in range(N):
        x = D[n]
        t = x.t
        team = x.team
        X = x.x
        Y = x.y

        if team == 1:
            if t < first_half_end:#右方向に攻撃
                if 15.0 < X < 85.0 and -60 < Y < 60:
                    if np.size(Action_Team1_VitalArea) == 1: 
                        #Action_Team1_VitalAreaにまだデータがない場合
                        f = np.array([x.t, x.team, x.p, x.a, x.x, x.y])
                        Action_Team1_VitalArea = f
                    else:
                        f = np.array([x.t, x.team, x.p, x.a, x.x, x.y])
                        Action_Team1_VitalArea = np.vstack([Action_Team1_VitalArea,f])

            elif t > last_half_start:#左方向に攻撃            
                if -85.0 < X < -15.0 and -60 < Y < 60:
                    if np.size(Action_Team1_VitalArea) == 1: 
                        #Action_Team1_VitalAreaにまだデータがない場合
                        f = np.array([x.t, x.team, x.p, x.a, x.x, x.y])
                        Action_Team1_VitalArea = f
                    else:
                        f = np.array([x.t, x.team, x.p, x.a, x.x, x.y])
                        Action_Team1_VitalArea = np.vstack([Action_Team1_VitalArea,f])

        if team == 2:
            if t < first_half_end:#左方向に攻撃
                if -85.0 < X < -15.0 and -60 < Y < 60:
                    if np.size(Action_Team2_VitalArea) == 1: 
                        #Action_Team2_VitalAreaにまだデータがない場合
                        f = np.array([x.t, x.team, x.p, x.a, x.x, x.y])
                        Action_Team2_VitalArea = f
                    else:
                        f = np.array([x.t, x.team, x.p, x.a, x.x, x.y])
                        Action_Team2_VitalArea = np.vstack([Action_Team2_VitalArea,f])

            elif t > last_half_start:#右方向に攻撃            
                if 15.0 < X < 85.0 and -60 < Y < 60:
                    if np.size(Action_Team2_VitalArea) == 1: 
                        #Action_Team2_VitalAreaにまだデータがない場合
                        f = np.array([x.t, x.team, x.p, x.a, x.x, x.y])
                        Action_Team2_VitalArea = f
                    else:
                        f = np.array([x.t, x.team, x.p, x.a, x.x, x.y])
                        Action_Team2_VitalArea = np.vstack([Action_Team2_VitalArea,f])


    Action_Team1_id_first_half = np.where(Action_Team1_VitalArea[:,0] < first_half_end)[0]
    Action_Team1_id_last_half = np.where(Action_Team1_VitalArea[:,0] > last_half_start)[0]

    Action_Team2_id_first_half = np.where(Action_Team2_VitalArea[:,0] < first_half_end)[0]
    Action_Team2_id_last_half = np.where(Action_Team2_VitalArea[:,0] > last_half_start)[0]

    fig = plt.figure()
    X = Action_Team1_VitalArea[Action_Team1_id_first_half,4]
    Y = Action_Team1_VitalArea[Action_Team1_id_first_half,5]
    plt.scatter(X,Y)
    plt.axis([-155, 155, -100, 100])
    x = np.arange(15.0, 85.0, 0.005)
    y0 = np.ones(len(x)) * -60
    y1 = np.ones(len(x)) * 60
    plt.fill_between(x, y0, y1, edgecolor = 'none', facecolor = 'blue', alpha = 0.1)
    plt.savefig('Action_Team1_VitalArea/Action_Team1_first_half_VitalArea.png')
    plt.close()

    fig = plt.figure()
    X = Action_Team1_VitalArea[Action_Team1_id_last_half,4]
    Y = Action_Team1_VitalArea[Action_Team1_id_last_half,5]
    plt.scatter(X,Y)
    plt.axis([-155, 155, -100, 100])
    x = np.arange(-85.0, -15.0, 0.005)
    y0 = np.ones(len(x)) * -60
    y1 = np.ones(len(x)) * 60
    plt.fill_between(x, y0, y1, edgecolor = 'none', facecolor = 'blue', alpha = 0.1)
    plt.savefig('Action_Team1_VitalArea/Action_Team1_last_half_VitalArea.png')
    plt.close()

    fig = plt.figure()
    X = Action_Team2_VitalArea[Action_Team2_id_first_half,4]
    Y = Action_Team2_VitalArea[Action_Team2_id_first_half,5]
    plt.scatter(X,Y)
    plt.axis([-155, 155, -100, 100])
    x = np.arange(-85.0, -15.0, 0.005)#左方向に攻撃
    y0 = np.ones(len(x)) * -60
    y1 = np.ones(len(x)) * 60
    plt.fill_between(x, y0, y1, edgecolor = 'none', facecolor = 'blue', alpha = 0.1)
    plt.savefig('Action_Team2_VitalArea/Action_Team2_first_half_VitalArea.png')
    plt.close()

    fig = plt.figure()
    X = Action_Team2_VitalArea[Action_Team2_id_last_half,4]
    Y = Action_Team2_VitalArea[Action_Team2_id_last_half,5]
    plt.scatter(X,Y)
    plt.axis([-155, 155, -100, 100])
    x = np.arange(15.0, 85.0, 0.005)#右方向に攻撃
    y0 = np.ones(len(x)) * -60
    y1 = np.ones(len(x)) * 60
    plt.fill_between(x, y0, y1, edgecolor = 'none', facecolor = 'blue', alpha = 0.1)
    plt.savefig('Action_Team2_VitalArea/Action_Team2_last_half_VitalArea.png')
    plt.close()
    
    print 'time:%f' % (time()-t0)
#-----------------------------------------------------------------------------------------


#Action Timing----------------------------------------------------------------------------
def Action_Timing():
    global tmin, tmax
    global N_Action_Team1_VitalArea, N_Action_Team2_VitalArea

    tmin = 0
    tmax = t_dic[len(t_dic) - 1] - t_dic[0]
    x = np.arange(tmax + 1)
    y0 = np.zeros(tmax + 1)



    pdb.set_trace()
    #Y_HalfTime = np.copy(y0)
    #HalfTime = last_half_start - first_half_end + 1
    #for i in range(HalfTime):
    #    Y_HalfTime[first_half_end - t_dic[0] + i] = 1.0

    Y_Team1_shots = np.copy(y0)
    for i in range(N_Team1_shots):
        S = Seq_Team1_shots[i]
        if np.size(S) > 6:
            shot_event_no = np.shape(S)[0] - 1
            shot_event_time = S[shot_event_no,0] - t_dic[0]
            Y_Team1_shots[shot_event_time] = 0.8

    Y_Team1_Action_VitalArea = np.copy(y0)
    N_Action_Team1_VitalArea = len(Action_Team1_VitalArea)
    for i in range(N_Action_Team1_VitalArea):
        Action_event_VitalArea = Action_Team1_VitalArea[i]
        Action_time_VitalArea = Action_event_VitalArea[0] - t_dic[0]
        Y_Team1_Action_VitalArea[Action_time_VitalArea] = 0.8

    fig = plt.figure(figsize=(8,4))
    plt.subplots_adjust(hspace=0.4)
    plt.subplot(2, 1, 1)

    plt.fill_between(x, y0, Y_Team1_shots, edgecolor = 'blue', facecolor = 'blue', alpha = 0.8)
    plt.fill_between(x, y0, Y_HalfTime, edgecolor = 'green', facecolor = 'green', alpha = 0.4)
    plt.axis([tmin, tmax, 0, 1])
    plt.yticks([])
    plt.title('Team1_shot_time')

    plt.subplot(2, 1, 2)
    plt.fill_between(x, y0, Y_Team1_Action_VitalArea, edgecolor = 'blue', facecolor = 'blue', alpha = 0.8)
    plt.axis([tmin, tmax, 0, 1])
    plt.yticks([])
    plt.title('Team1_Action_time_VitalArea')

    plt.show()
    pdb.set_trace()

    

#-----------------------------------------------------------------------------------------


input()#データ読み込み
Seq_Team_shots()#シュートに至った攻撃機会のボールの軌跡
Action_Plot_VitalArea()#バイタルエリア内での行動の回数

Action_Timing()
pdb.set_trace()

pdb.set_trace()
