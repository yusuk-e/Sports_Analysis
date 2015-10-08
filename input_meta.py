# -*- coding:utf-8 -*-

import pdb
from time import time
import datetime as dt
import numpy as np
from scipy.special import gammaln
import matplotlib
#matplotlib.use('Agg') #DIPLAYの設定
import matplotlib.pyplot as plt
import resource
import codecs
import random
from collections import defaultdict
from collections import namedtuple
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde


#--variable--
action_dic = {}
team_dic = {}
player1_dic = {}
player2_dic = {}
t_dic = []
elem = namedtuple("elem", "t, re_t, team, p, a, x, y")
#絶対時刻, ハーフ相対時間，チームID，アクションID, x座標，y座標
Stoppage = [40, 27]

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
N = 0

Seq_Team1_of = defaultdict(int)
Seq_Team2_of = defaultdict(int)
N_Team1_of = -1
N_Team2_of = -1

shot_first_half_Team1_re_t = []
shot_last_half_Team1_re_t = []
shot_first_half_Team2_re_t = []
shot_last_half_Team2_re_t = []

#Dx = 9#x方向メッシュ分割数
#Dy = 6#y方向メッシュ分割数
Dx = 6
Dy = 4

K = 10
#C = ['blue', 'red', 'green', 'black', '']
C = ['#ff7f7f', '#ff7fbf', '#ff7fff', '#bf7fff', '#7f7fff', '#7fbfff', '#7fffff', '#7fffbf', \
     '#7fff7f', '#bfff7f', '#fff7f', '#ffb7f']

#------------


def input():
#--Input--

    global xmax, xmin, ymax, ymin
    global first_half_start, first_half_end, last_half_start, last_half_end
    global first_half_start_re_t, first_half_end_re_t, last_half_start_re_t, last_half_end_re_t
    global N

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
            team_dic[team] = len(team_dic)
            #team_dic.append(team)

        player = int(temp[3])
        if team == 1:
            if player not in player1_dic:
                player1_dic[player] = len(player1_dic)
                #player1_dic.append(player)
        elif team == 2:
            if player not in player2_dic:
                player2_dic[player] = len(player2_dic)
                #player2_dic.append(player)
        else:
            print "err"

        action = int(temp[4])
        if action not in action_dic:
            action_dic[action] = len(action_dic)
            #action_dic.append(action)

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

        f = elem(t, re_t, team, player, action, x, y)
        D[counter] = f
        counter += 1

    fin.close()
    N = counter - 1
    Reverse_Seq()#後半の攻撃を反転
    print "time:%f" % (time()-t0)



def Reverse_Seq():
#--後半反転--
    for n in range(N):
        x = D[n]
        t = x.t

        if t > last_half_start:#左方向に攻撃            
            f = elem(t, x.re_t, x.team, x.p, x.a, -x.x, -x.y)
            D[n] = f


def Seq_Team_of():
#--offense ボール軌跡--
    global N_Team1_of, N_Team2_of
    t0 = time()
    
    n = 0
    pre_team = 0
    flag = 0
    while n < N:

        team = D[n].team
        action = D[n].a

        if action in Stoppage:
            flag = 1
            n += 1
            #pdb.set_trace()

        else:
            #pdb.set_trace()
            if pre_team == team:
                if team == 1:
                    if flag == 1:
                        N_Team1_of += 1
                        flag = 0

                    x = D[n]
                    if np.size(Seq_Team1_of[N_Team1_of]) == 1: 
                        #Seq_Team1_ofにまだデータがない場合
                        f = np.array([x.t,x.re_t, x.team, x.p, x.a, x.x, x.y])
                        Seq_Team1_of[N_Team1_of] = f
                    else:
                        f = np.array([x.t, x.re_t, x.team, x.p, x.a, x.x, x.y])
                        Seq_Team1_of[N_Team1_of] = np.vstack([Seq_Team1_of[N_Team1_of],f])

                    n += 1
                    pre_team = team

                elif team == 2:
                    if flag == 1:
                        N_Team2_of += 1
                        flag = 0

                    x = D[n]
                    if np.size(Seq_Team2_of[N_Team2_of]) == 1: 
                        #Seq_Team2_ofにまだデータがない場合
                        f = np.array([x.t,x.re_t, x.team, x.p, x.a, x.x, x.y])
                        Seq_Team2_of[N_Team2_of] = f
                    else:
                        f = np.array([x.t, x.re_t, x.team, x.p, x.a, x.x, x.y])
                        Seq_Team2_of[N_Team2_of] = np.vstack([Seq_Team2_of[N_Team2_of],f])

                    n += 1
                    pre_team = team

            else:
                #pdb.set_trace()
                if team == 1:
                    N_Team1_of += 1
                    x = D[n]
                    if np.size(Seq_Team1_of[N_Team1_of]) == 1: 
                        #Seq_Team1_ofにまだデータがない場合
                        f = np.array([x.t,x.re_t, x.team, x.p, x.a, x.x, x.y])
                        Seq_Team1_of[N_Team1_of] = f
                    else:
                        f = np.array([x.t, x.re_t, x.team, x.p, x.a, x.x, x.y])
                        Seq_Team1_of[N_Team1_of] = np.vstack([Seq_Team1_of[N_Team1_of],f])

                    n += 1
                    pre_team = team

                elif team == 2:
                    N_Team2_of += 1
                    x = D[n]
                    if np.size(Seq_Team2_of[N_Team2_of]) == 1: 
                        #Seq_Team2_ofにまだデータがない場合
                        f = np.array([x.t,x.re_t, x.team, x.p, x.a, x.x, x.y])
                        Seq_Team2_of[N_Team2_of] = f
                    else:
                        f = np.array([x.t, x.re_t, x.team, x.p, x.a, x.x, x.y])
                        Seq_Team2_of[N_Team2_of] = np.vstack([Seq_Team2_of[N_Team2_of],f])

                    n += 1
                    pre_team = team

    N_Team1_of = len(Seq_Team1_of)
    N_Team2_of = len(Seq_Team2_of)

    #pdb.set_trace()
    Remove_one()#一回しかアクションがない攻撃は消去
    Quantization()#位置情報をメッシュ状に量子化
    shot_timing()#シュートのタイミングを取得
    #Visualize_Seq()#オフェンス時のボール軌跡データ可視化
    print 'time:%f' % (time()-t0)


def Remove_one():
#--一回しかアクションがない攻撃は消去--
    global N_Team1_of, N_Team2_of, Seq_Team1_of, Seq_Team2_of

    tmp_Seq_Team1_of = defaultdict(int)
    tmp_Seq_Team2_of = defaultdict(int)    

    counter = 0
    for n in range(N_Team1_of):
        S = Seq_Team1_of[n]
        S_size = np.size(S)/7
        if S_size > 1:
            tmp_Seq_Team1_of[counter] = S
            counter += 1

    counter = 0
    for n in range(N_Team2_of):
        S = Seq_Team2_of[n]
        S_size = np.size(S)/7
        if S_size > 1:
            tmp_Seq_Team2_of[counter] = S
            counter += 1

    Seq_Team1_of = tmp_Seq_Team1_of
    Seq_Team2_of = tmp_Seq_Team2_of
    N_Team1_of = len(tmp_Seq_Team1_of)
    N_Team2_of = len(tmp_Seq_Team2_of)


def shot_timing():
#--シュートのタイミングを取得--

    for n in range(N_Team1_of):
        S = Seq_Team1_of[n]
        action = int(S[len(S)-1][4])
        shot_t = S[len(S)-1][0]
        shot_re_t = S[len(S)-1][1]
        if action == 15 and shot_t < first_half_end:
            shot_first_half_Team1_re_t.append(shot_re_t)
        elif action == 15 and shot_t > last_half_start:
            shot_last_half_Team1_re_t.append(shot_re_t)

    for n in range(N_Team2_of):
        S = Seq_Team2_of[n]
        action = int(S[len(S)-1][4])
        shot_t = S[len(S)-1][0]
        shot_re_t = S[len(S)-1][1]
        if action == 15 and shot_t < first_half_end:
            shot_first_half_Team2_re_t.append(shot_re_t)
        elif action == 15 and shot_t > last_half_start:
            shot_last_half_Team2_re_t.append(shot_re_t)


def Quantization():
#--位置情報をメッシュ状に量子化--

    tmp_xmax = float(2 * xmax) + 1
    tmp_ymax = float(2 * ymax) + 1

    t0 = time()
    for n in range(N_Team1_of):
        S = Seq_Team1_of[n]
        S_size = np.size(S)/7
        for s in range(S_size):
            line = S[s]
            x = line[5]
            y = line[6]

            tmp_x = float(x) - xmin#原点をコートの左下に
            tmp_y = float(y) - ymin

            Mx_id = int( Dx * tmp_x / tmp_xmax )#メッシュidを計算
            My_id = int( Dy * tmp_y / tmp_ymax )
            My_id = Dy - 1 - int( Dy * tmp_y / tmp_ymax )#左上がメッシュid=0になるように反転

            M_id = My_id * Dx + Mx_id
            if s == 0:
                M_id_set = np.array(M_id)
            else:
                M_id_set = np.vstack([M_id_set, M_id])

        Seq_Team1_of[n] = np.hstack([S, M_id_set])

    for n in range(N_Team2_of):
        S = Seq_Team2_of[n]
        S_size = np.size(S)/7
        for s in range(S_size):
            line = S[s]
            x = line[5]
            y = line[6]

            tmp_x = float(x) - xmin#原点をコートの左下に
            tmp_y = float(y) - ymin

            Mx_id = int( Dx * tmp_x / tmp_xmax )#メッシュidを計算
            My_id = int( Dy * tmp_y / tmp_ymax )
            My_id = Dy - 1 - int( Dy * tmp_y / tmp_ymax )#左上がメッシュid=0になるように反転

            M_id = My_id * Dx + Mx_id
            if s == 0:
                M_id_set = np.array(M_id)
            else:
                M_id_set = np.vstack([M_id_set, M_id])

        Seq_Team2_of[n] = np.hstack([S, M_id_set])

    print 'time:%f' % (time()-t0)

def make_BoF():
#--パス系列と量子化された位置情報を含むBag-of-Feature作成--

    t0 = time()

    flag = 0
    for n in range(N_Team1_of):
        N_player = len(player1_dic)
        M = np.zeros([N_player, N_player])
        L = np.zeros(Dx * Dy)
        S = Seq_Team1_of[n]
        Pass_Series = S[:,3]
        for i in range(len(Pass_Series) - 1):
            now_p_ind = Pass_Series[i]
            next_p_ind = Pass_Series[i+1]

            now_p = player1_dic[now_p_ind]
            next_p = player1_dic[next_p_ind]
            M[now_p, next_p] += 1


        #プレイヤーグラフ対称の場合-----------
        '''
        M2 = np.zeros([N_player, N_player])
        for i in range(N_player):
            for j in range(N_player):
                M2[i,j] = M[i,j] + M[j,i]

        pass_line = []
        for i in range(N_player):
            for j in range(N_player):
                if i < j:
                    pass_line.append(M2[i,j])
        pass_line = np.array(pass_line)
        '''
        #-------------------------------------

        #プレイヤーグラフ非対称の場合---------
        pass_line = []
        for i in range(N_player):
            for j in range(N_player):
                if i != j:
                    pass_line.append(M[i,j])
        pass_line = np.array(pass_line)
        #-------------------------------------

        Loc_Series = S[:,7]
        for i in range(len(Loc_Series)):
            Mesh = Loc_Series[i]
            L[Mesh] += 1
        
        line = np.hstack([pass_line, L])

        if flag == 0:
            BoF_Team1 = line
            flag = 1
        else:
            BoF_Team1 = np.vstack([BoF_Team1, line])


    flag = 0
    for n in range(N_Team2_of):
        N_player = len(player2_dic)
        M = np.zeros([N_player, N_player])
        L = np.zeros(Dx * Dy)
        S = Seq_Team2_of[n]
        Pass_Series = S[:,3]
        for i in range(len(Pass_Series) - 1):
            now_p_ind = Pass_Series[i]
            next_p_ind = Pass_Series[i+1]

            now_p = player2_dic[now_p_ind]
            next_p = player2_dic[next_p_ind]
            M[now_p, next_p] += 1

        #プレイヤーグラフ対称の場合-----------
        '''
        M2 = np.zeros([N_player, N_player])
        for i in range(N_player):
            for j in range(N_player):
                M2[i,j] = M[i,j] + M[j,i]

        pass_line = []
        for i in range(N_player):
            for j in range(N_player):
                if i < j:
                    pass_line.append(M2[i,j])
        pass_line = np.array(pass_line)
        '''
        #-------------------------------------

        #プレイヤーグラフ非対称の場合-----------                
        pass_line = []
        for i in range(N_player):
            for j in range(N_player):
                if i != j:
                    pass_line.append(M[i,j])
        pass_line = np.array(pass_line)
        #-------------------------------------

        Loc_Series = S[:,7]
        for i in range(len(Loc_Series)):
            Mesh = Loc_Series[i]
            L[Mesh] += 1
        
        line = np.hstack([pass_line, L])

        if flag == 0:
            BoF_Team2 = line
            flag = 1
        else:
            BoF_Team2 = np.vstack([BoF_Team2, line])

    BoF_Team1, BoF_Team2 = normalization(BoF_Team1, BoF_Team2)#平均0分散1に標準化
    print 'time:%f' % (time()-t0)
    return BoF_Team1, BoF_Team2


def normalization(BoF_Team1, BoF_Team2):
#--平均0分散1に標準化--

    dim = np.shape(BoF_Team1)[1]
    #del_dim_Team1 = []
    #del_dim_Team2 = []
    for i in range(dim):
        X = BoF_Team1[:,i]
        sumX = np.sum(X)
        if sumX != 0:
            average = np.mean(X)
            standard_dev = np.std(X)
            BoF_Team1[:,i] = (BoF_Team1[:,i] - average) / standard_dev
        #else:
        #    del_dim_Team1.append(i)

        X = BoF_Team2[:,i]
        sumX = np.sum(X)
        if sumX != 0:
            average = np.mean(BoF_Team2[:,i])
            standard_dev = np.std(BoF_Team2[:,i])
            BoF_Team2[:,i] = (BoF_Team2[:,i] - average) / standard_dev
        #else:
        #    del_dim_Team2.append(i)

    #BoF_Team1 = np.delete(BoF_Team1, del_dim_Team1, 1)
    #BoF_Team2 = np.delete(BoF_Team2, del_dim_Team2, 1)

    return BoF_Team1, BoF_Team2

def Visualize_Seq():
#--ボール軌跡可視化--
    t0 = time()
    for n in range(N_Team1_of):
        S = Seq_Team1_of[n]
        timing = int(S[0,1])
        team = int(S[0,2])
        action = int(S[np.shape(S)[0] - 1, 4])
        X = S[:,5]
        Y = S[:,6]
        fig = plt.figure()
        if action == 15:
            plt.quiver(X[:-1], Y[:-1], X[1:]-X[:-1], Y[1:]-Y[:-1], width=0.003, \
                       scale_units='xy', angles='xy', scale=1, color='red')
        else:
            plt.quiver(X[:-1], Y[:-1], X[1:]-X[:-1], Y[1:]-Y[:-1], width=0.003, \
                       scale_units='xy', angles='xy', scale=1, color='darkcyan')
        plt.axis([-155, 155, -100, 100])
        plt.savefig('Seq_Team' + str(team) + '/Seq_Team1_of'+'_no'+str(n)+'_t'+str(timing)+'.png')
        plt.close()

    for n in range(N_Team2_of):
        S = Seq_Team2_of[n]
        timing = int(S[0,1])
        team = int(S[0,2])
        action = int(S[np.shape(S)[0] - 1, 4])
        X = S[:,5]
        Y = S[:,6]
        fig = plt.figure()
        if action == 15:
            plt.quiver(X[:-1], Y[:-1], X[1:]-X[:-1], Y[1:]-Y[:-1], width=0.003, \
                       scale_units='xy', angles='xy', scale=1, color='red')
        else:
            plt.quiver(X[:-1], Y[:-1], X[1:]-X[:-1], Y[1:]-Y[:-1], width=0.003, \
                       scale_units='xy', angles='xy', scale=1, color='darkcyan')
        plt.axis([-155, 155, -100, 100])
        plt.savefig('Seq_Team' + str(team) + '/Seq_Team2_of'+'_no'+str(n)+'_t'+str(timing)+'.png')
        plt.close()

    print 'time:%f' % (time()-t0)


def Possession():
#--Possession--

    x_first_half = np.arange(first_half_end_re_t + 1)
    y0_first_half = np.zeros(first_half_end_re_t + 1)

    x_last_half = np.arange(last_half_end_re_t + 1)
    y0_last_half = np.zeros(last_half_end_re_t + 1)


    Y_first_half_Team1_of = np.copy(y0_first_half)
    Y_last_half_Team1_of = np.copy(y0_last_half)
    for i in range(N_Team1_of):
        S = Seq_Team1_of[i]
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
    plt.fill_between(x_first_half, y0_first_half, Y_first_half_Team1_of, \
                     edgecolor = 'mediumaquamarine', facecolor = 'mediumaquamarine')
    tempX = np.array(shot_first_half_Team1_re_t)
    tempY = np.ones(len(shot_first_half_Team1_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=20, edgecolor = 'black', facecolor = 'black')
    plt.axis([first_half_start_re_t, first_half_end_re_t, 0, 1])
    plt.yticks([])
    plt.title('Team1_first_half_offense')

    plt.subplot(2, 1, 2)
    plt.fill_between(x_last_half, y0_last_half, Y_last_half_Team1_of, edgecolor = 'mediumaquamarine', \
                     facecolor = 'mediumaquamarine')
    tempX = np.array(shot_last_half_Team1_re_t)
    tempY = np.ones(len(shot_last_half_Team1_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=20, edgecolor = 'black', facecolor = 'black')
    plt.axis([last_half_start_re_t, last_half_end_re_t, 0, 1])
    plt.yticks([])
    plt.title('Team1_last_half_offense')

    plt.savefig('Seq_Team1/Team1_offense.png')
    #plt.show()
    plt.close()



    Y_first_half_Team2_of = np.copy(y0_first_half)
    Y_last_half_Team2_of = np.copy(y0_last_half)
    for i in range(N_Team2_of):
        S = Seq_Team2_of[i]
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
    plt.fill_between(x_first_half, y0_first_half, Y_first_half_Team2_of, edgecolor = 'orange', \
                     facecolor = 'orange')
    tempX = np.array(shot_first_half_Team2_re_t)
    tempY = np.ones(len(shot_first_half_Team2_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=20, edgecolor = 'black', facecolor = 'black')
    plt.axis([first_half_start_re_t, first_half_end_re_t, 0, 1])
    plt.yticks([])
    plt.title('Team2_first_half_offense')

    plt.subplot(2, 1, 2)
    plt.fill_between(x_last_half, y0_last_half, Y_last_half_Team2_of, edgecolor = 'orange', \
                     facecolor = 'orange')
    tempX = np.array(shot_last_half_Team2_re_t)
    tempY = np.ones(len(shot_last_half_Team2_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=20, edgecolor = 'black', facecolor = 'black')
    plt.axis([last_half_start_re_t, last_half_end_re_t, 0, 1])
    plt.yticks([])
    plt.title('Team2_last_half_offense')

    plt.savefig('Seq_Team2/Team2_offense.png')
    #plt.show()
    plt.close()


def Clustering(BoF_Team1, BoF_Team2):
    t0 = time()

    PCA_threshold = 0.8

    #--Team1--
    dim = np.shape(BoF_Team1)[0]
    threshold_dim = 0
    for i in range(dim):
        pca = PCA(n_components = i)
        pca.fit(BoF_Team1)
        X = pca.transform(BoF_Team1)
        E = pca.explained_variance_ratio_
        if np.sum(E) > PCA_threshold:
            thereshold_dim = len(E)
            print 'Team1 dim:%d' % thereshold_dim
            break

    pca = PCA(n_components = thereshold_dim)
    pca.fit(BoF_Team1)
    X = pca.transform(BoF_Team1)

    min_score = 10000
    for i in range(200):
        model = KMeans(n_clusters=K, init='k-means++', max_iter=1000, tol=0.0001).fit(X)
        if min_score > model.score(X):
            labels_Team1 = model.labels_

    pca = PCA(n_components = 2)
    pca.fit(BoF_Team1)
    X = pca.transform(BoF_Team1)
    for k in range(K):
        labels_Team1_ind = np.where(labels_Team1 == k)[0]
        plt.scatter(X[labels_Team1_ind,0], X[labels_Team1_ind,1], color=C[k])

    plt.title('Team1_PCA_kmeans')
    #plt.legend()
    plt.savefig('Seq_Team1/Team1_PCA_kmeans.png')
    #plt.show()
    plt.close()
    np.savetxt('Seq_Team1/labels_Team1.csv', labels_Team1, delimiter=',')

    #--Team2--
    dim = np.shape(BoF_Team2)[0]
    threshold_dim = 0
    for i in range(dim):
        pca = PCA(n_components = i)
        pca.fit(BoF_Team2)
        X = pca.transform(BoF_Team2)
        E = pca.explained_variance_ratio_
        if np.sum(E) > PCA_threshold:
            thereshold_dim = len(E)
            print 'Team2 dim:%d' % thereshold_dim
            break

    min_score = 10000
    for i in range(200):
        model = KMeans(n_clusters=K, init='k-means++', max_iter=1000, tol=0.0001).fit(X)
        if min_score > model.score(X):
            labels_Team2 = model.labels_

    pca = PCA(n_components = 2)
    pca.fit(BoF_Team2)
    X = pca.transform(BoF_Team2)
    for k in range(K):
        labels_Team2_ind = np.where(labels_Team2 == k)[0]
        plt.scatter(X[labels_Team2_ind,0], X[labels_Team2_ind,1], color=C[k])

    plt.title('Team2_PCA_kmeans')
    plt.savefig('Seq_Team2/Team2_PCA_kmeans.png')
    #plt.show()
    plt.close()
    np.savetxt('Seq_Team2/labels_Team2.csv', labels_Team2, delimiter=',')

    print 'time:%f' % (time()-t0)
    return labels_Team1, labels_Team2


def Visualize_tactical_pattern():
#--kmeansで出力されたラベルに基づいて攻撃パターンを色塗り--

    labels_Team1 = np.loadtxt('Seq_Team1/labels_Team1.csv', delimiter=',')
    labels_Team2 = np.loadtxt('Seq_Team2/labels_Team2.csv', delimiter=',')

    x_first_half = np.arange(first_half_end_re_t + 1)
    y0_first_half = np.zeros(first_half_end_re_t + 1)

    x_last_half = np.arange(last_half_end_re_t + 1)
    y0_last_half = np.zeros(last_half_end_re_t + 1)

    #--Team1--
    fig = plt.figure(figsize=(16,4))
    plt.subplots_adjust(hspace=0.4)


    for k in range(K):
        Y_first_half_Team1_of = np.copy(y0_first_half)
        Y_last_half_Team1_of = np.copy(y0_last_half)

        for i in range(N_Team1_of):
            if labels_Team1[i] == k:
                S = Seq_Team1_of[i]
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

        plt.subplot(2, 1, 1)        
        plt.fill_between(x_first_half, y0_first_half, Y_first_half_Team1_of, \
                         edgecolor = C[k], facecolor = C[k])

        plt.subplot(2, 1, 2)
        plt.fill_between(x_last_half, y0_last_half, Y_last_half_Team1_of, \
                         edgecolor = C[k], facecolor = C[k])

    plt.subplot(2, 1, 1)        
    tempX = np.array(shot_first_half_Team1_re_t)
    tempY = np.ones(len(shot_first_half_Team1_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=20, edgecolor = 'black', facecolor = 'black')
    plt.axis([first_half_start_re_t, first_half_end_re_t, 0, 1])
    plt.yticks([])
    plt.title('Team1_first_half_offense')

    plt.subplot(2, 1, 2)
    tempX = np.array(shot_last_half_Team1_re_t)
    tempY = np.ones(len(shot_last_half_Team1_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=20, edgecolor = 'black', facecolor = 'black')
    plt.axis([last_half_start_re_t, last_half_end_re_t, 0, 1])
    plt.yticks([])
    plt.title('Team1_last_half_offense')

    plt.savefig('Seq_Team1/Vis_tactical_pattern_Team1.png')
    #plt.show()
    plt.close()


    #--Team2--
    fig = plt.figure(figsize=(16,4))
    plt.subplots_adjust(hspace=0.4)


    for k in range(K):
        Y_first_half_Team2_of = np.copy(y0_first_half)
        Y_last_half_Team2_of = np.copy(y0_last_half)

        for i in range(N_Team2_of):
            if labels_Team2[i] == k:
                S = Seq_Team2_of[i]
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

        plt.subplot(2, 1, 1)        
        plt.fill_between(x_first_half, y0_first_half, Y_first_half_Team2_of, \
                         edgecolor = C[k], facecolor = C[k])

        plt.subplot(2, 1, 2)
        plt.fill_between(x_last_half, y0_last_half, Y_last_half_Team2_of, \
                         edgecolor = C[k], facecolor = C[k])

    plt.subplot(2, 1, 1)        
    tempX = np.array(shot_first_half_Team2_re_t)
    tempY = np.ones(len(shot_first_half_Team2_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=20, edgecolor = 'black', facecolor = 'black')
    plt.axis([first_half_start_re_t, first_half_end_re_t, 0, 1])
    plt.yticks([])
    plt.title('Team2_first_half_offense')

    plt.subplot(2, 1, 2)
    tempX = np.array(shot_last_half_Team2_re_t)
    tempY = np.ones(len(shot_last_half_Team2_re_t)) * 0.5
    plt.scatter(tempX, tempY, s=20, edgecolor = 'black', facecolor = 'black')
    plt.axis([last_half_start_re_t, last_half_end_re_t, 0, 1])
    plt.yticks([])
    plt.title('Team2_last_half_offense')

    plt.savefig('Seq_Team2/Vis_tactical_pattern_Team2.png')
    #plt.show()
    plt.close()
    

def Cluster_analysis():
#--kmeansで出力されたクラスタの平均など分析--

    labels_Team1 = np.loadtxt('Seq_Team1/labels_Team1.csv', delimiter=',')
    labels_Team2 = np.loadtxt('Seq_Team2/labels_Team2.csv', delimiter=',')

    for k in range(K):
        index = np.where(labels_Team1 == k)[0]

        #--位置情報の可視化--
        for i in range(len(index)):
            n = index[i]
            S = Seq_Team1_of[n]
            x1 = S[:,5]
            x2 = S[:,6]
            d = np.vstack([x1,x2]).T
            if i == 0:
                D = d
            else:
                D = np.vstack([D,d])
                
        X, Y = np.mgrid[xmin:xmax:1000j, ymin:ymax:1000j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        kernel = gaussian_kde(D.T)
        Z = np.reshape(kernel(positions).T, X.shape)
        plt.imshow(np.rot90(Z), extent=[xmin, xmax, ymin, ymax])
        plt.scatter(D[:,0],D[:,1], edgecolor='grey',facecolor='grey', s = 10)
        plt.title('Team1_Cluster' + str(k) + '_location')
        plt.savefig('Seq_Team1/Cluster' + str(k) + '_location_Team1.png')
        plt.close()

        #--プレイヤーグラフの可視化--
        N_player = len(player1_dic)
        M = np.zeros([N_player, N_player])
        for i in range(len(index)):
            n = index[i]
            S = Seq_Team1_of[n]

            Pass_Series = S[:,3]
            for i in range(len(Pass_Series) - 1):
                now_p_ind = Pass_Series[i]
                next_p_ind = Pass_Series[i+1]

                now_p = player1_dic[now_p_ind]
                next_p = player1_dic[next_p_ind]
                M[now_p, next_p] += 1


        plt.pcolor(M, cmap=plt.cm.Blues)   
        plt.title('Team1_PlayerGraph_Cluster' + str(k))
        plt.savefig('Seq_Team1/Player_Graph_Cluster' + str(k) + '_Team1.png')
        plt.close()

        #-各クラスタのボール軌跡データ作成
        for i in range(len(index)):
            n = index[i]
            S = Seq_Team1_of[n]
            timing = int(S[0,1])
            team = int(S[0,2])
            action = int(S[np.shape(S)[0] - 1, 4])
            X = S[:,5]
            Y = S[:,6]
            fig = plt.figure()
            if action == 15:
                plt.quiver(X[:-1], Y[:-1], X[1:]-X[:-1], Y[1:]-Y[:-1], width=0.003, \
                           scale_units='xy', angles='xy', scale=1, color='red')
            else:
                plt.quiver(X[:-1], Y[:-1], X[1:]-X[:-1], Y[1:]-Y[:-1], width=0.003, \
                           scale_units='xy', angles='xy', scale=1, color='darkcyan')
            plt.axis([-155, 155, -100, 100])
            plt.savefig('Seq_Team1/Seq_Team1_of'+'_no'+str(n)+'_t'+str(timing)+'C' + str(k) + '.png')
            plt.close()


        #--位置情報の可視化--
        index = np.where(labels_Team2 == k)[0]
        for i in range(len(index)):
            n = index[i]
            S = Seq_Team2_of[n]
            x1 = S[:,5]
            x2 = S[:,6]
            d = np.vstack([x1,x2]).T
            if i == 0:
                D = d
            else:
                D = np.vstack([D,d])

        X, Y = np.mgrid[xmin:xmax:1000j, ymin:ymax:1000j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        kernel = gaussian_kde(D.T)
        Z = np.reshape(kernel(positions).T, X.shape)
        plt.imshow(np.rot90(Z), extent=[xmin, xmax, ymin, ymax])
        plt.scatter(D[:,0],D[:,1], edgecolor='grey',facecolor='grey', s = 10)
        plt.title('Team2_Cluster' + str(k) + '_location')
        plt.savefig('Seq_Team2/Cluster' + str(k) + '_location_Team2.png')
        plt.close()

        #--プレイヤーグラフの可視化--
        N_player = len(player2_dic)
        M = np.zeros([N_player, N_player])
        for i in range(len(index)):
            n = index[i]
            S = Seq_Team2_of[n]

            Pass_Series = S[:,3]
            for i in range(len(Pass_Series) - 1):
                now_p_ind = Pass_Series[i]
                next_p_ind = Pass_Series[i+1]

                now_p = player2_dic[now_p_ind]
                next_p = player2_dic[next_p_ind]
                M[now_p, next_p] += 1


        plt.pcolor(M, cmap=plt.cm.Blues)   
        plt.title('Team2_PlayerGraph_Cluster' + str(k))
        plt.savefig('Seq_Team2/Player_Graph_Cluster' + str(k) + '_Team2.png')
        plt.close()

        #-各クラスタのボール軌跡データ作成
        for i in range(len(index)):
            n = index[i]
            S = Seq_Team1_of[n]
            timing = int(S[0,1])
            team = int(S[0,2])
            action = int(S[np.shape(S)[0] - 1, 4])
            X = S[:,5]
            Y = S[:,6]
            fig = plt.figure()
            if action == 15:
                plt.quiver(X[:-1], Y[:-1], X[1:]-X[:-1], Y[1:]-Y[:-1], width=0.003, \
                           scale_units='xy', angles='xy', scale=1, color='red')
            else:
                plt.quiver(X[:-1], Y[:-1], X[1:]-X[:-1], Y[1:]-Y[:-1], width=0.003, \
                           scale_units='xy', angles='xy', scale=1, color='darkcyan')
            plt.axis([-155, 155, -100, 100])
            plt.savefig('Seq_Team2/Seq_Team2_of'+'_no'+str(n)+'_t'+str(timing)+'C' + str(k) + '.png')
            plt.close()

    #pdb.set_trace()






#--main--
input()
#データ読み込み

Seq_Team_of()
#オフェンス時のボール軌跡データ作成

BoF_Team1, BoF_Team2 = make_BoF()
#パス系列と量子化された位置情報を含むBag-of-Feature作成

Possession()
#各チームのボール保持時間とシュートのタイミングを描画

labels_Team1, labels_Team2 = Clustering(BoF_Team1, BoF_Team2)
#BoFを入力にして攻撃パターンをクラスタリング

Visualize_tactical_pattern()
#kmeansで出力されたラベルに基づいて攻撃パターンを色塗り

Cluster_analysis()
#kmeansで出力されたクラスタの平均など分析

pdb.set_trace()
