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

Seq_Team1_of = defaultdict(int)
Seq_Team2_of = defaultdict(int)
N_Team1_of = 0
N_Team2_of = 0

Seq_Team1_shots = defaultdict(int)
Seq_Team2_shots = defaultdict(int)
N_Team1_shots = 0
N_Team2_shots = 0

shot_first_half_Team1_re_t = []
shot_last_half_Team1_re_t = []
shot_first_half_Team2_re_t = []
shot_last_half_Team2_re_t = []

Vital_time_Team1_first_half = []
Vital_time_Team1_last_half = []
Vital_time_Team2_first_half = []
Vital_time_Team2_last_half = []

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
    global first_half_start_re_t, first_half_end_re_t, last_half_start_re_t, last_half_end_re_t

    counter = 0
    t0 = time()
    filename = "processed_metadata.csv"
    fin = open(filename)

    temp = fin.readline().rstrip("\r\n").split(",")
    first_half_start = int(temp[0])
    first_half_start_re_t = int(temp[1])
    temp = fin.readline().rstrip("\r\n").split(",")
    first_half_end = int(temp[0])
    first_half_end_re_t = int(temp[1])
    temp = fin.readline().rstrip("\r\n").split(",")
    last_half_start = int(temp[0])
    last_half_start_re_t = int(temp[1])
    temp = fin.readline().rstrip("\r\n").split(",")
    last_half_end = int(temp[0])
    last_half_end_re_t = int(temp[1])

    counter = 0
    for row in fin:
        temp = row.rstrip("\r\n").split(",")

        t = int(temp[0])
        if t not in t_dic:
            t_dic.append(t)

        re_t = int(temp[1])

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

    fin.close()
    N = counter - 1
    print "time:%f" % (time()-t0)
#-----------------------------------------------------------------------------------------


#of ボール軌跡-----------------------------------------------------------------------------
def Seq_Team_of():
    global N_Team1_of, N_Team2_of
    t0 = time()
    
    n = 0
    team = D[n].team
    while n < N:

        if team == 1:
            #if D[n].team == 2:
                #N_Team1_of += 1
                #team = D[n].team
            
            if D[n].team == 1 and D[n].a != 15:
                if n == 0:
                    prev_t = D[n].t
                    prev_team = D[n].team
                    prev_a = D[n].a
                else:
                    prev_t = D[n - 1].t
                    prev_team = D[n - 1].team
                    prev_a = D[n - 1].a

                if D[n].t - prev_t > 10 and prev_team == 1 and prev_a !=15:
                    N_Team1_of += 1

                    x = D[n]
                    if np.size(Seq_Team1_of[N_Team1_of]) == 1: 
                        #Seq_Team1_ofにまだデータがない場合
                        f = np.array([x.t,x.re_t, x.team, x.p, x.a, x.x, x.y])
                        Seq_Team1_of[N_Team1_of] = f

                    else:
                        f = np.array([x.t, x.re_t, x.team, x.p, x.a, x.x, x.y])
                        Seq_Team1_of[N_Team1_of] = np.vstack([Seq_Team1_of[N_Team1_of],f])
                        
                #n += 1
                #team = D[n].team
                    
                #elif D[n].t - prev_t <= 10 and prev_team != 1:
                else:
                    x = D[n]
                    if np.size(Seq_Team1_of[N_Team1_of]) == 1: 
                        #Seq_Team1_ofにまだデータがない場合
                        f = np.array([x.t,x.re_t, x.team, x.p, x.a, x.x, x.y])
                        Seq_Team1_of[N_Team1_of] = f

                    else:
                        f = np.array([x.t, x.re_t, x.team, x.p, x.a, x.x, x.y])
                        Seq_Team1_of[N_Team1_of] = np.vstack([Seq_Team1_of[N_Team1_of],f])
                        
                n += 1
                team = D[n].team
                if team == 2:
                    N_Team1_of += 1

            elif D[n].team == 1 and D[n].a == 15:
                x = D[n]
                if np.size(Seq_Team1_of[N_Team1_of]) == 1: 
                    #Seq_Team1_ofにまだデータがない場合
                    f = np.array([x.t,x.re_t, x.team, x.p, x.a, x.x, x.y])
                    Seq_Team1_of[N_Team1_of] = f

                else:
                    f = np.array([x.t, x.re_t, x.team, x.p, x.a, x.x, x.y])
                    Seq_Team1_of[N_Team1_of] = np.vstack([Seq_Team1_of[N_Team1_of],f])

                N_Team1_of += 1
                n += 1
                team = D[n].team
                
        elif team == 2:
            #if D[n].team == 1:
                #N_Team2_of += 1
                #team = D[n].team

            if D[n].team == 2 and D[n].a != 15:
                if n == 0:
                    prev_t = D[n].t
                    prev_team = D[n].team
                    prev_a = D[n].a
                else:
                    prev_t = D[n - 1].t
                    prev_team = D[n - 1].team
                    prev_a = D[n - 1].a

                if D[n].t - prev_t > 10 and prev_team == 2 and prev_a != 15:
                    N_Team2_of += 1

                    x = D[n]
                    if np.size(Seq_Team2_of[N_Team2_of]) == 1: 
                        #Seq_Team2_ofにまだデータがない場合
                        f = np.array([x.t,x.re_t, x.team, x.p, x.a, x.x, x.y])
                        Seq_Team2_of[N_Team2_of] = f

                    else:
                        f = np.array([x.t, x.re_t, x.team, x.p, x.a, x.x, x.y])
                        Seq_Team2_of[N_Team2_of] = np.vstack([Seq_Team2_of[N_Team2_of],f])

                    #n += 1
                    
                #elif D[n].t - prev_t <= 10 and prev_team != 2:
                else:
                    x = D[n]
                    if np.size(Seq_Team2_of[N_Team2_of]) == 1: 
                        #Seq_Team2_ofにまだデータがない場合
                        f = np.array([x.t,x.re_t, x.team, x.p, x.a, x.x, x.y])
                        Seq_Team2_of[N_Team2_of] = f

                    else:
                        f = np.array([x.t, x.re_t, x.team, x.p, x.a, x.x, x.y])
                        Seq_Team2_of[N_Team2_of] = np.vstack([Seq_Team2_of[N_Team2_of],f])

                n += 1
                team = D[n].team
                if team == 1:
                    N_Team2_of += 1


            elif D[n].team == 2 and D[n].a == 15:
                x = D[n]
                if np.size(Seq_Team2_of[N_Team2_of]) == 1: 
                    #Seq_Team2_ofにまだデータがない場合
                    f = np.array([x.t,x.re_t, x.team, x.p, x.a, x.x, x.y])
                    Seq_Team2_of[N_Team2_of] = f

                else:
                    f = np.array([x.t, x.re_t, x.team, x.p, x.a, x.x, x.y])
                    Seq_Team2_of[N_Team2_of] = np.vstack([Seq_Team2_of[N_Team2_of],f])

                N_Team2_of += 1
                n += 1
                team = D[n].team

        else:
            print 'err'

    N_Team1_of = len(Seq_Team1_of)
    N_Team2_of = len(Seq_Team2_of)

    for s in range(N_Team1_of):
        S = Seq_Team1_of[s]
        if np.size(S) > 7:
            Start_time = int(S[0,0])
            X = S[:,5]
            Y = S[:,6]
            T = S[:,1]
            timing = int(T[len(T)-1])
            if Start_time < first_half_end:#右方向に攻撃
                x = np.arange(15.0, 85.0, 0.005)
                y0 = np.ones(len(x)) * -60
                y1 = np.ones(len(x)) * 60

                flag = 0
                for i in range(len(X)):
                    tempX = X[i]
                    tempY = Y[i]
                    if 15.0 < tempX < 85.0 and -60 < tempY < 60:
                        flag = 1
                if flag == 1:
                    Vital_time_Team1_first_half.append(timing)

                    fig = plt.figure()
                    plt.quiver(X[:-1], Y[:-1], X[1:]-X[:-1], Y[1:]-Y[:-1], width=0.003, scale_units='xy', angles='xy', scale=1, color='darkcyan')
                    plt.axis([-155, 155, -100, 100])
                    plt.fill_between(x, y0, y1, edgecolor = 'none', facecolor = 'dimgray', alpha = 0.2)
                    plt.savefig('Seq_Team1_VitalArea/Seq_Team1_of'+'_no'+str(s)+'_t'+str(timing)+'.png')
                    plt.close()

                if flag != 1:
                    fig = plt.figure()
                    plt.quiver(X[:-1], Y[:-1], X[1:]-X[:-1], Y[1:]-Y[:-1], width=0.003, scale_units='xy', angles='xy', scale=1, color='darkcyan')
                    plt.axis([-155, 155, -100, 100])
                    plt.fill_between(x, y0, y1, edgecolor = 'none', facecolor = 'dimgray', alpha = 0.2)
                    plt.savefig('Seq_Team1_not_VitalArea/Seq_Team1_of'+'_no'+str(s)+'_t'+str(timing)+'.png')
                    plt.close()
                    
                if int(S[len(S) - 1][4]) == 15:
                    fig = plt.figure()
                    plt.quiver(X[:-1], Y[:-1], X[1:]-X[:-1], Y[1:]-Y[:-1], width=0.003, scale_units='xy', angles='xy', scale=1, color='darkcyan')
                    plt.axis([-155, 155, -100, 100])
                    plt.fill_between(x, y0, y1, edgecolor = 'none', facecolor = 'dimgray', alpha = 0.2)
                    plt.savefig('Seq_Team1_shot/Seq_Team1_of'+'_no'+str(s)+'_t'+str(timing)+'.png')
                    plt.close()
                       
                if flag == 1 and int(S[len(S) - 1][4]) == 15:
                    fig = plt.figure()
                    plt.quiver(X[:-1], Y[:-1], X[1:]-X[:-1], Y[1:]-Y[:-1], width=0.003, scale_units='xy', angles='xy', scale=1, color='darkcyan')
                    plt.axis([-155, 155, -100, 100])
                    plt.fill_between(x, y0, y1, edgecolor = 'none', facecolor = 'dimgray', alpha = 0.2)
                    plt.savefig('Seq_Team1_VitalArea_shot/Seq_Team1_of'+'_no'+str(s)+'_t'+str(timing)+'.png')
                    plt.close()


            elif Start_time > last_half_start:#左方向に攻撃
                x = np.arange(-85.0, -15.0, 0.005)
                y0 = np.ones(len(x)) * -60
                y1 = np.ones(len(x)) * 60

                flag = 0
                for i in range(len(X)):
                    tempX = X[i]
                    tempY = Y[i]
                    if -85.0 < tempX < -15.0 and -60 < tempY < 60:
                        flag = 1
                if flag == 1:
                    Vital_time_Team1_last_half.append(T[len(T)-1])

                    fig = plt.figure()
                    plt.quiver(X[:-1], Y[:-1], X[1:]-X[:-1], Y[1:]-Y[:-1], width=0.003, scale_units='xy', angles='xy', scale=1, color='darkcyan')
                    plt.axis([-155, 155, -100, 100])
                    plt.fill_between(x, y0, y1, edgecolor = 'none', facecolor = 'dimgray', alpha = 0.2)
                    plt.savefig('Seq_Team1_VitalArea/Seq_Team1_of'+'_no'+str(s)+'_t'+str(timing)+'.png')
                    plt.close()

                if flag != 1:
                    fig = plt.figure()
                    plt.quiver(X[:-1], Y[:-1], X[1:]-X[:-1], Y[1:]-Y[:-1], width=0.003, scale_units='xy', angles='xy', scale=1, color='darkcyan')
                    plt.axis([-155, 155, -100, 100])
                    plt.fill_between(x, y0, y1, edgecolor = 'none', facecolor = 'dimgray', alpha = 0.2)
                    plt.savefig('Seq_Team1_not_VitalArea/Seq_Team1_of'+'_no'+str(s)+'_t'+str(timing)+'.png')
                    plt.close()
                    
                if int(S[len(S) - 1][4]) == 15:
                    fig = plt.figure()
                    plt.quiver(X[:-1], Y[:-1], X[1:]-X[:-1], Y[1:]-Y[:-1], width=0.003, scale_units='xy', angles='xy', scale=1, color='darkcyan')
                    plt.axis([-155, 155, -100, 100])
                    plt.fill_between(x, y0, y1, edgecolor = 'none', facecolor = 'dimgray', alpha = 0.2)
                    plt.savefig('Seq_Team1_shot/Seq_Team1_of'+'_no'+str(s)+'_t'+str(timing)+'.png')
                    plt.close()
                       
                if flag == 1 and int(S[len(S) - 1][4]) == 15:
                    fig = plt.figure()
                    plt.quiver(X[:-1], Y[:-1], X[1:]-X[:-1], Y[1:]-Y[:-1], width=0.003, scale_units='xy', angles='xy', scale=1, color='darkcyan')
                    plt.axis([-155, 155, -100, 100])
                    plt.fill_between(x, y0, y1, edgecolor = 'none', facecolor = 'dimgray', alpha = 0.2)
                    plt.savefig('Seq_Team1_VitalArea_shot/Seq_Team1_of'+'_no'+str(s)+'_t'+str(timing)+'.png')
                    plt.close()


            fig = plt.figure()
            plt.quiver(X[:-1], Y[:-1], X[1:]-X[:-1], Y[1:]-Y[:-1], width=0.003, scale_units='xy', angles='xy', scale=1, color='darkcyan')
            plt.axis([-155, 155, -100, 100])
            plt.fill_between(x, y0, y1, edgecolor = 'none', facecolor = 'dimgray', alpha = 0.2)
            plt.savefig('Seq_Team1_of/Seq_Team1_of'+'_no'+str(s)+'_t'+str(timing)+'.png')
            plt.close()

    for s in range(N_Team2_of):
        S = Seq_Team2_of[s]
        if np.size(S) > 7:
            Start_time = int(S[0,0])
            X = S[:,5]
            Y = S[:,6]
            T = S[:,1]
            timing = int(T[len(T)-1])
            if Start_time < first_half_end:
                x = np.arange(-85.0, -15.0, 0.005)#左方向に攻撃
                y0 = np.ones(len(x)) * -60
                y1 = np.ones(len(x)) * 60

                flag = 0
                for i in range(len(X)):
                    tempX = X[i]
                    tempY = Y[i]
                    if -85.0 < tempX < -15.0 and -60 < tempY < 60:
                        flag = 1
                if flag == 1:
                    Vital_time_Team2_first_half.append(T[len(T)-1])

                    fig = plt.figure()
                    plt.quiver(X[:-1], Y[:-1], X[1:]-X[:-1], Y[1:]-Y[:-1], width=0.003, scale_units='xy', angles='xy', scale=1, color='darkcyan')
                    plt.axis([-155, 155, -100, 100])
                    plt.fill_between(x, y0, y1, edgecolor = 'none', facecolor = 'dimgray', alpha = 0.2)
                    plt.savefig('Seq_Team2_VitalArea/Seq_Team2_of'+'_no'+str(s)+'_t'+str(timing)+'.png')
                    plt.close()

                if flag != 1:
                    fig = plt.figure()
                    plt.quiver(X[:-1], Y[:-1], X[1:]-X[:-1], Y[1:]-Y[:-1], width=0.003, scale_units='xy', angles='xy', scale=1, color='darkcyan')
                    plt.axis([-155, 155, -100, 100])
                    plt.fill_between(x, y0, y1, edgecolor = 'none', facecolor = 'dimgray', alpha = 0.2)
                    plt.savefig('Seq_Team2_not_VitalArea/Seq_Team2_of'+'_no'+str(s)+'_t'+str(timing)+'.png')
                    plt.close()
                    
                if int(S[len(S) - 1][4]) == 15:
                    fig = plt.figure()
                    plt.quiver(X[:-1], Y[:-1], X[1:]-X[:-1], Y[1:]-Y[:-1], width=0.003, scale_units='xy', angles='xy', scale=1, color='darkcyan')
                    plt.axis([-155, 155, -100, 100])
                    plt.fill_between(x, y0, y1, edgecolor = 'none', facecolor = 'dimgray', alpha = 0.2)
                    plt.savefig('Seq_Team2_shot/Seq_Team2_of'+'_no'+str(s)+'_t'+str(timing)+'.png')
                    plt.close()
                       
                if flag == 1 and int(S[len(S) - 1][4]) == 15:
                    fig = plt.figure()
                    plt.quiver(X[:-1], Y[:-1], X[1:]-X[:-1], Y[1:]-Y[:-1], width=0.003, scale_units='xy', angles='xy', scale=1, color='darkcyan')
                    plt.axis([-155, 155, -100, 100])
                    plt.fill_between(x, y0, y1, edgecolor = 'none', facecolor = 'dimgray', alpha = 0.2)
                    plt.savefig('Seq_Team2_VitalArea_shot/Seq_Team2_of'+'_no'+str(s)+'_t'+str(timing)+'.png')
                    plt.close()


            elif Start_time > last_half_start:
                x = np.arange(15.0, 85.0, 0.005)#右方向に攻撃
                y0 = np.ones(len(x)) * -60
                y1 = np.ones(len(x)) * 60

                flag = 0
                for i in range(len(X)):
                    tempX = X[i]
                    tempY = Y[i]
                    if 15.0 < tempX < 85.0 and -60 < tempY < 60:
                        flag = 1
                if flag == 1:
                    Vital_time_Team2_last_half.append(T[len(T)-1])

                    fig = plt.figure()
                    plt.quiver(X[:-1], Y[:-1], X[1:]-X[:-1], Y[1:]-Y[:-1], width=0.003, scale_units='xy', angles='xy', scale=1, color='darkcyan')
                    plt.axis([-155, 155, -100, 100])
                    plt.fill_between(x, y0, y1, edgecolor = 'none', facecolor = 'dimgray', alpha = 0.2)
                    plt.savefig('Seq_Team2_VitalArea/Seq_Team2_of'+'_no'+str(s)+'_t'+str(timing)+'.png')
                    plt.close()

                if flag != 1:
                    fig = plt.figure()
                    plt.quiver(X[:-1], Y[:-1], X[1:]-X[:-1], Y[1:]-Y[:-1], width=0.003, scale_units='xy', angles='xy', scale=1, color='darkcyan')
                    plt.axis([-155, 155, -100, 100])
                    plt.fill_between(x, y0, y1, edgecolor = 'none', facecolor = 'dimgray', alpha = 0.2)
                    plt.savefig('Seq_Team2_not_VitalArea/Seq_Team2_of'+'_no'+str(s)+'_t'+str(timing)+'.png')
                    plt.close()
                    
                if int(S[len(S) - 1][4]) == 15:
                    fig = plt.figure()
                    plt.quiver(X[:-1], Y[:-1], X[1:]-X[:-1], Y[1:]-Y[:-1], width=0.003, scale_units='xy', angles='xy', scale=1, color='darkcyan')
                    plt.axis([-155, 155, -100, 100])
                    plt.fill_between(x, y0, y1, edgecolor = 'none', facecolor = 'dimgray', alpha = 0.2)
                    plt.savefig('Seq_Team2_shot/Seq_Team2_of'+'_no'+str(s)+'_t'+str(timing)+'.png')
                    plt.close()
                       
                if flag == 1 and int(S[len(S) - 1][4]) == 15:
                    fig = plt.figure()
                    plt.quiver(X[:-1], Y[:-1], X[1:]-X[:-1], Y[1:]-Y[:-1], width=0.003, scale_units='xy', angles='xy', scale=1, color='darkcyan')
                    plt.axis([-155, 155, -100, 100])
                    plt.fill_between(x, y0, y1, edgecolor = 'none', facecolor = 'dimgray', alpha = 0.2)
                    plt.savefig('Seq_Team2_VitalArea_shot/Seq_Team2_of'+'_no'+str(s)+'_t'+str(timing)+'.png')
                    plt.close()



            fig = plt.figure()
            plt.quiver(X[:-1], Y[:-1], X[1:]-X[:-1], Y[1:]-Y[:-1], width=0.003, scale_units='xy', angles='xy', scale=1, color='darkcyan')
            plt.fill_between(x, y0, y1, edgecolor = 'none', facecolor = 'dimgray', alpha = 0.2)
            plt.axis([-155, 155, -100, 100])
            plt.savefig('Seq_Team2_of/Seq_Team2_of'+'_no'+str(s)+'_t'+str(timing)+'.png')
            plt.close()

    print 'time:%f' % (time()-t0)
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
                        f = np.array([x.t,x.re_t, x.team, x.p, x.a, x.x, x.y])
                        Seq_Team1_shots[N_Team1_shots] = f
                    else:
                        if D[k+1].t - D[k].t > 10:
                            N_Team1_shots += 1
                            break

                        f = np.array([x.t, x.re_t, x.team, x.p, x.a, x.x, x.y])
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
                        f = np.array([x.t,x.re_t, x.team, x.p, x.a, x.x, x.y])
                        Seq_Team2_shots[N_Team2_shots] = f
                    else:
                        if D[k+1].t - D[k].t > 10:
                            N_Team2_shots += 1
                            break

                        f = np.array([x.t,x.re_t, x.team, x.p, x.a, x.x, x.y])
                        Seq_Team2_shots[N_Team2_shots] = np.vstack([f, Seq_Team2_shots[N_Team2_shots]])
                    k -= 1
            else:
                print "err"

    N_Team1_shots = len(Seq_Team1_shots)
    N_Team2_shots = len(Seq_Team2_shots)

    for s in range(N_Team1_shots):
        S = Seq_Team1_shots[s]
        if np.size(S) > 7:
            Start_time = int(S[0,0])
            X = S[:,5]
            Y = S[:,6]
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
            plt.fill_between(x, y0, y1, edgecolor = 'none', facecolor = 'dimgray', alpha = 0.2)
#            plt.savefig('Seq_Team1_shots/Seq_Team1_shots'+'_no'+str(s)+'.png')
            plt.close()


    for s in range(N_Team2_shots):
        S = Seq_Team2_shots[s]
        if np.size(S) > 7:
            Start_time = int(S[0,0])
            X = S[:,5]
            Y = S[:,6]
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
            plt.fill_between(x, y0, y1, edgecolor = 'none', facecolor = 'dimgray', alpha = 0.2)
#            plt.savefig('Seq_Team2_shots/Seq_Team2_shots'+'_no'+str(s)+'.png')
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
                        f = np.array([x.t, x.re_t, x.team, x.p, x.a, x.x, x.y])
                        Action_Team1_VitalArea = f
                    else:
                        f = np.array([x.t,x.re_t, x.team, x.p, x.a, x.x, x.y])
                        Action_Team1_VitalArea = np.vstack([Action_Team1_VitalArea,f])

            elif t > last_half_start:#左方向に攻撃            
                if -85.0 < X < -15.0 and -60 < Y < 60:
                    if np.size(Action_Team1_VitalArea) == 1: 
                        #Action_Team1_VitalAreaにまだデータがない場合
                        f = np.array([x.t,x.re_t, x.team, x.p, x.a, x.x, x.y])
                        Action_Team1_VitalArea = f
                    else:
                        f = np.array([x.t,x.re_t, x.team, x.p, x.a, x.x, x.y])
                        Action_Team1_VitalArea = np.vstack([Action_Team1_VitalArea,f])

        if team == 2:
            if t < first_half_end:#左方向に攻撃
                if -85.0 < X < -15.0 and -60 < Y < 60:
                    if np.size(Action_Team2_VitalArea) == 1: 
                        #Action_Team2_VitalAreaにまだデータがない場合
                        f = np.array([x.t,x.re_t, x.team, x.p, x.a, x.x, x.y])
                        Action_Team2_VitalArea = f
                    else:
                        f = np.array([x.t,x.re_t, x.team, x.p, x.a, x.x, x.y])
                        Action_Team2_VitalArea = np.vstack([Action_Team2_VitalArea,f])

            elif t > last_half_start:#右方向に攻撃            
                if 15.0 < X < 85.0 and -60 < Y < 60:
                    if np.size(Action_Team2_VitalArea) == 1: 
                        #Action_Team2_VitalAreaにまだデータがない場合
                        f = np.array([x.t,x.re_t, x.team, x.p, x.a, x.x, x.y])
                        Action_Team2_VitalArea = f
                    else:
                        f = np.array([x.t,x.re_t, x.team, x.p, x.a, x.x, x.y])
                        Action_Team2_VitalArea = np.vstack([Action_Team2_VitalArea,f])




    Action_Team1_id_first_half = np.where(Action_Team1_VitalArea[:,0] < first_half_end)[0]
    Action_Team1_id_last_half = np.where(Action_Team1_VitalArea[:,0] > last_half_start)[0]

    Action_Team2_id_first_half = np.where(Action_Team2_VitalArea[:,0] < first_half_end)[0]
    Action_Team2_id_last_half = np.where(Action_Team2_VitalArea[:,0] > last_half_start)[0]


    fig = plt.figure()
    X = Action_Team1_VitalArea[Action_Team1_id_first_half,5]
    Y = Action_Team1_VitalArea[Action_Team1_id_first_half,6]
    plt.scatter(X,Y)
    plt.axis([-155, 155, -100, 100])
    x = np.arange(15.0, 85.0, 0.005)
    y0 = np.ones(len(x)) * -60
    y1 = np.ones(len(x)) * 60
    plt.fill_between(x, y0, y1, edgecolor = 'none', facecolor = 'dimgray', alpha = 0.1)
    plt.savefig('Action_Team1_VitalArea/Action_Team1_first_half_VitalArea.png')
    plt.close()

    fig = plt.figure()
    X = Action_Team1_VitalArea[Action_Team1_id_last_half,5]
    Y = Action_Team1_VitalArea[Action_Team1_id_last_half,6]
    plt.scatter(X,Y)
    plt.axis([-155, 155, -100, 100])
    x = np.arange(-85.0, -15.0, 0.005)
    y0 = np.ones(len(x)) * -60
    y1 = np.ones(len(x)) * 60
    plt.fill_between(x, y0, y1, edgecolor = 'none', facecolor = 'dimgray', alpha = 0.1)
    plt.savefig('Action_Team1_VitalArea/Action_Team1_last_half_VitalArea.png')
    plt.close()

    fig = plt.figure()
    X = Action_Team2_VitalArea[Action_Team2_id_first_half,5]
    Y = Action_Team2_VitalArea[Action_Team2_id_first_half,6]
    plt.scatter(X,Y)
    plt.axis([-155, 155, -100, 100])
    x = np.arange(-85.0, -15.0, 0.005)#左方向に攻撃
    y0 = np.ones(len(x)) * -60
    y1 = np.ones(len(x)) * 60
    plt.fill_between(x, y0, y1, edgecolor = 'none', facecolor = 'dimgray', alpha = 0.1)
    plt.savefig('Action_Team2_VitalArea/Action_Team2_first_half_VitalArea.png')
    plt.close()

    fig = plt.figure()
    X = Action_Team2_VitalArea[Action_Team2_id_last_half,5]
    Y = Action_Team2_VitalArea[Action_Team2_id_last_half,6]
    plt.scatter(X,Y)
    plt.axis([-155, 155, -100, 100])
    x = np.arange(15.0, 85.0, 0.005)#右方向に攻撃
    y0 = np.ones(len(x)) * -60
    y1 = np.ones(len(x)) * 60
    plt.fill_between(x, y0, y1, edgecolor = 'none', facecolor = 'dimgray', alpha = 0.1)
    plt.savefig('Action_Team2_VitalArea/Action_Team2_last_half_VitalArea.png')
    plt.close()
    
    print 'time:%f' % (time()-t0)
#-----------------------------------------------------------------------------------------


#Action Timing----------------------------------------------------------------------------
def Action_Timing():
    global tmin, tmax
    global N_Action_Team1_VitalArea, N_Action_Team2_VitalArea
    global first_half_start_re_t, first_half_end_re_t, last_half_start_re_t, last_half_end_re_t

    x_first_half = np.arange(first_half_end_re_t + 1)
    y0_first_half = np.zeros(first_half_end_re_t + 1)

    x_last_half = np.arange(last_half_end_re_t + 1)
    y0_last_half = np.zeros(last_half_end_re_t + 1)

    Y_first_half_Team1_shots = np.copy(y0_first_half)
    Y_last_half_Team1_shots = np.copy(y0_last_half)
    for i in range(N_Team1_shots):
        S = Seq_Team1_shots[i]
        if np.size(S) == 7:
            shot_event_t = S[0]
            shot_event_re_t = S[1]
            if shot_event_t < first_half_end:
                shot_first_half_Team1_re_t.append(shot_event_re_t)
            elif shot_event_t > last_half_start:
                shot_last_half_Team1_re_t.append(shot_event_re_t)

        if np.size(S) > 7:
            shot_event_no = np.shape(S)[0] - 1
            shot_event_t = S[shot_event_no,0]
            shot_event_re_t = S[shot_event_no,1]
            if shot_event_t < first_half_end:
                shot_first_half_Team1_re_t.append(shot_event_re_t)
                Y_first_half_Team1_shots[shot_event_re_t] = 0.8
            elif shot_event_t > last_half_start:
                shot_last_half_Team1_re_t.append(shot_event_re_t)
                Y_last_half_Team1_shots[shot_event_re_t] = 0.8

    for i in range(N_Team2_shots):
        S = Seq_Team2_shots[i]
        if np.size(S) == 7:
            shot_event_t = S[0]
            shot_event_re_t = S[1]
            if shot_event_t < first_half_end:
                shot_first_half_Team2_re_t.append(shot_event_re_t)
            elif shot_event_t > last_half_start:
                shot_last_half_Team2_re_t.append(shot_event_re_t)

        if np.size(S) > 7:
            shot_event_no = np.shape(S)[0] - 1
            shot_event_t = S[shot_event_no,0]
            shot_event_re_t = S[shot_event_no,1]
            if shot_event_t < first_half_end:
                shot_first_half_Team2_re_t.append(shot_event_re_t)
            elif shot_event_t > last_half_start:
                shot_last_half_Team2_re_t.append(shot_event_re_t)

    Y_first_half_Team1_Action_VitalArea = np.copy(y0_first_half)
    Y_last_half_Team1_Action_VitalArea = np.copy(y0_last_half)
    N_Action_Team1_VitalArea = len(Action_Team1_VitalArea)
    for i in range(N_Action_Team1_VitalArea):
        Action_event_VitalArea = Action_Team1_VitalArea[i]
        Action_t_VitalArea = Action_event_VitalArea[0]
        Action_re_t_VitalArea = Action_event_VitalArea[1]
        if Action_t_VitalArea < first_half_end:
            Y_first_half_Team1_Action_VitalArea[Action_re_t_VitalArea] = 0.8
        elif Action_t_VitalArea > last_half_start:
            Y_last_half_Team1_Action_VitalArea[Action_re_t_VitalArea] = 0.8


    fig = plt.figure(figsize=(16,4))
    plt.subplots_adjust(hspace=0.4)

    plt.subplot(2, 2, 1)
    plt.fill_between(x_first_half, y0_first_half, Y_first_half_Team1_shots, edgecolor = 'dimgray', facecolor = 'dimgray', alpha = 0.8)
    plt.axis([first_half_start_re_t, first_half_end_re_t, 0, 1])
    plt.yticks([])
    plt.title('Team1_first_half_shot')

    plt.subplot(2, 2, 2)
    plt.fill_between(x_last_half, y0_last_half, Y_last_half_Team1_shots, edgecolor = 'dimgray', facecolor = 'dimgray', alpha = 0.8)
    plt.axis([last_half_start_re_t, last_half_end_re_t, 0, 1])
    plt.yticks([])
    plt.title('Team1_last_half_shot')

    plt.subplot(2, 2, 3)
    plt.fill_between(x_first_half, y0_first_half, Y_first_half_Team1_Action_VitalArea, edgecolor = 'dimgray', facecolor = 'dimgray', alpha = 0.8)
    plt.axis([first_half_start_re_t, first_half_end_re_t, 0, 1])
    plt.yticks([])
    plt.title('Team1_first_half_Action_VitalArea')

    plt.subplot(2, 2, 4)
    plt.fill_between(x_last_half, y0_last_half, Y_last_half_Team1_Action_VitalArea, edgecolor = 'dimgray', facecolor = 'dimgray', alpha = 0.8)
    plt.axis([last_half_start_re_t, last_half_end_re_t, 0, 1])
    plt.yticks([])
    plt.title('Team1_last_half_Action_VitalArea')


    plt.close()
#    plt.show()
#    pdb.set_trace()
#-----------------------------------------------------------------------------------------


#Possession----------------------------------------------------------------------------
def Possession():

    x_first_half = np.arange(first_half_end_re_t + 1)
    y0_first_half = np.zeros(first_half_end_re_t + 1)

    x_last_half = np.arange(last_half_end_re_t + 1)
    y0_last_half = np.zeros(last_half_end_re_t + 1)


    Y_first_half_Team1_of = np.copy(y0_first_half)
    Y_last_half_Team1_of = np.copy(y0_last_half)
    for i in range(N_Team1_of):
        S = Seq_Team1_of[i]
        if np.size(S) == 7:
            event_t = S[0]
            event_re_t = S[1]
            if event_t < first_half_end:
                Y_first_half_Team1_of[event_re_t] = 1.0
            elif event_t > last_half_start:
                Y_last_half_Team1_of[event_re_t] = 1.0

        if np.size(S) > 7:
            start_t = S[0,0]
            start_re_t = S[0,1]
            end_re_t = S[len(S) - 1,1]
            period = int(end_re_t - start_re_t)
            if start_t < first_half_end:
                for j in range(period):
                    Y_first_half_Team1_of[start_re_t + j] = 1.0
            elif start_t > last_half_start:
                for j in range(period):
                    Y_last_half_Team1_of[start_re_t + j] = 1.0
    
    fig = plt.figure(figsize=(16,4))
    plt.subplots_adjust(hspace=0.4)

    plt.subplot(2, 1, 1)
    plt.fill_between(x_first_half, y0_first_half, Y_first_half_Team1_of, edgecolor = 'mediumaquamarine', facecolor = 'mediumaquamarine')#, alpha = 0.4)
    tempX = np.array(shot_first_half_Team1_re_t)
    tempY = np.ones(len(shot_first_half_Team1_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=20, edgecolor = 'black', facecolor = 'black')
    tempX = np.array(Vital_time_Team1_first_half)
    tempY = np.ones(len(Vital_time_Team1_first_half)) * 0.5
    plt.scatter(tempX, tempY, s=70, edgecolor = 'black', facecolor = 'none')
    plt.axis([first_half_start_re_t, first_half_end_re_t, 0, 1])
    plt.yticks([])
    plt.title('Team1_first_half_offense')

    plt.subplot(2, 1, 2)
    plt.fill_between(x_last_half, y0_last_half, Y_last_half_Team1_of, edgecolor = 'mediumaquamarine', facecolor = 'mediumaquamarine')#, alpha = 0.4)
    tempX = np.array(shot_last_half_Team1_re_t)
    tempY = np.ones(len(shot_last_half_Team1_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=20, edgecolor = 'black', facecolor = 'black')
    tempX = np.array(Vital_time_Team1_last_half)
    tempY = np.ones(len(Vital_time_Team1_last_half)) * 0.5
    plt.scatter(tempX, tempY, s=80, edgecolor = 'black', facecolor = 'none')
    plt.axis([last_half_start_re_t, last_half_end_re_t, 0, 1])
    plt.yticks([])
    plt.title('Team1_last_half_offense')

    plt.savefig('Team1_offense/Team1_offense.png')
    #plt.show()
    plt.close()



    Y_first_half_Team2_of = np.copy(y0_first_half)
    Y_last_half_Team2_of = np.copy(y0_last_half)
    for i in range(N_Team2_of):
        S = Seq_Team2_of[i]
        if np.size(S) == 7:
            event_t = S[0]
            event_re_t = S[1]
            if event_t < first_half_end:
                Y_first_half_Team2_of[event_re_t] = 1.0
            elif event_t > last_half_start:
                Y_last_half_Team2_of[event_re_t] = 1.0

        if np.size(S) > 7:
            start_t = S[0,0]
            start_re_t = S[0,1]
            end_re_t = S[len(S) - 1,1]
            period = int(end_re_t - start_re_t)
            if start_t < first_half_end:
                for j in range(period):
                    Y_first_half_Team2_of[start_re_t + j] = 1.0
            elif start_t > last_half_start:
                for j in range(period):
                    Y_last_half_Team2_of[start_re_t + j] = 1.0
    
    fig = plt.figure(figsize=(16,4))
    plt.subplots_adjust(hspace=0.4)

    plt.subplot(2, 1, 1)
    plt.fill_between(x_first_half, y0_first_half, Y_first_half_Team2_of, edgecolor = 'orange', facecolor = 'orange')#, alpha = 0.4)
    tempX = np.array(shot_first_half_Team2_re_t)
    tempY = np.ones(len(shot_first_half_Team2_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=20, edgecolor = 'black', facecolor = 'black')
    tempX = np.array(Vital_time_Team2_first_half)
    tempY = np.ones(len(Vital_time_Team2_first_half)) * 0.5
    plt.scatter(tempX, tempY, s=70, edgecolor = 'black', facecolor = 'none')
    plt.axis([first_half_start_re_t, first_half_end_re_t, 0, 1])
    plt.yticks([])
    plt.title('Team2_first_half_offense')

    plt.subplot(2, 1, 2)
    plt.fill_between(x_last_half, y0_last_half, Y_last_half_Team2_of, edgecolor = 'orange', facecolor = 'orange')#, alpha = 0.4)
    tempX = np.array(shot_last_half_Team2_re_t)
    tempY = np.ones(len(shot_last_half_Team2_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=20, edgecolor = 'black', facecolor = 'black')
    tempX = np.array(Vital_time_Team2_last_half)
    tempY = np.ones(len(Vital_time_Team2_last_half)) * 0.5
    plt.scatter(tempX, tempY, s=80, edgecolor = 'black', facecolor = 'none')
    plt.axis([last_half_start_re_t, last_half_end_re_t, 0, 1])
    plt.yticks([])
    plt.title('Team2_last_half_offense')

    plt.savefig('Team2_offense/Team2_offense.png')
    #plt.show()
    plt.close()




    pdb.set_trace()

#-----------------------------------------------------------------------------------------


input()#データ読み込み
Seq_Team_of()#オフェンス時のボールの軌跡
Seq_Team_shots()#シュートに至った攻撃機会のボールの軌跡
Action_Plot_VitalArea()#バイタルエリア内での行動の回数
Action_Timing()
Possession()

pdb.set_trace()

pdb.set_trace()
