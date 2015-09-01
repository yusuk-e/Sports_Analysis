# -*- coding:utf-8 -*-

import pdb
from time import time
import datetime as dt
import numpy as np
from scipy.special import gammaln
import resource
import codecs
import random
from collections import defaultdict
from collections import namedtuple
#variable---------------------------------------------------------------------------------
Team_dic = [0,1]
Player_Team0_dic = []
Player_Team1_dic = []
sys_id_dic = []
frame_id_dic = []
Foo_p = namedtuple("Foo_p", "x, y, v")
Foo_b = namedtuple("Foo_b", "x,y,z,v,Team,Status,info")

#-----------------------------------------------------------------------------------------
counter = 0
t0 = time()
#------------------------------------------------------------------------------------
filename = "udp.out"
D = defaultdict(lambda: defaultdict(lambda: defaultdict(int))) #選手の位置情報 D[frame_id][Player_id]
B = defaultdict(int) #ボールの位置情報 B[frame_id]

fin = open(filename)
counter = 0
for row in fin:
    temp0 = row.rstrip("\r\n").split(":")
    frame_id = int(temp0[0])
    frame_id_dic.append(frame_id)

    temp1 = temp0[1].rstrip("\r\n").split(";")

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

        f = Foo_p(float(temp2[3]), float(temp2[4]), float(temp2[5]))
        D[frame_id][Team_id][Player_id] = f

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
    else:
        print "err"

    B[frame_id] = f

fin.close()
#-----------------------------------------------------------------------------------------
print "time:%f" % (time()-t0)
pdb.set_trace()




pdb.set_trace()
