import airsim
import cv2
import numpy as np
import os
import pprint
import setup_path 
import tempfile
import time
import matplotlib.pyplot as plt

numUAV = 4; name = []; s = []
tout = 10; spd = 2; alt = -10
min_dist = 2

pos0 = np.zeros((numUAV,3))
for i in range(numUAV):
    pos0[i,0] = 12.0 - 2.0*i
    pos0[i,1] = 12.0
    pos0[i,2] = -10.0


client = airsim.MultirotorClient()
client.confirmConnection()
for i in range(numUAV ):
    name.append("UAV" + str(i))
    client.enableApiControl(True, name[i])
    client.armDisarm(True, name[i])
    #print("UAV", i, "have been initialized.")
print("All UAVs have been initialized.")

###############################################################################
#--------------------     Functions       --------------------------------------   
################################################################################

def readposition(n,s):
    #s.clear()
    x = client.simGetGroundTruthKinematics(vehicle_name = name[n]).position.x_val
    y = client.simGetGroundTruthKinematics(vehicle_name = name[n]).position.y_val
    z = client.simGetGroundTruthKinematics(vehicle_name = name[n]).position.z_val
    #yaw = client.simGetGroundTruthKinematics(vehicle_name = name[n]).position
    d = [round(x,3),round(y,3),round(z,3)]
    s.append(d)
    return s


def ShapeVectorSquare(y,key): # defining initial Shape vector 
    key = str(key)
    shapevec = {"S10":[2,10],"S01":[0,-y],"S20": [-6,0],"S02": [-y,-y],"S21":[y,0],"S12":[-y,0],"S30":[-4,10],"S03":[-y,0],"S31":[y,-y],"S13":[-y,y],"S32":[0,-y],"S23":[0,y]}
    #shapevec = [[0,y,0,0], [y,0,0,0],  [y,y,0,0], [y,0,0,0], [0,y,0,0]]
                    #S21        S31        S42        S43       S32
    return shapevec[key]
     
def SearchSqShape_Vectors(y):
    Sij = "S"+str(y)+str(0)
    shape_vect = ShapeVectorSquare(20.0,Sij)
    print("Shape Vector Selected is: ",shape_vect)
    return shape_vect


def Initialpositions(s):
    Iposition = [] 
    for i in range(len(s)):
        Iposition.append([])
        for j in range(2):
            #Iposition[i][j] = s[i][j] + pos0[i,j]
            Iposition[i].append(round(s[i][j]+pos0[i][j] , 2))
    return Iposition

    
def CostFunction(cur_pos, shape_vect):
    pos_1 = [cur_pos[0] + 2, cur_pos[1]]
    pos_2 = [cur_pos[0] - 2, cur_pos[1]]
    pos_3 = [cur_pos[0] + 3, cur_pos[1]]
    pos_4 = [cur_pos[0] - 3, cur_pos[1]]
    pos_5 = [cur_pos[0], cur_pos[1] + 2]
    pos_6 = [cur_pos[0], cur_pos[1] - 2]
    pos_7 = [cur_pos[0], cur_pos[1] + 3]
    pos_8 = [cur_pos[0], cur_pos[1] - 3]
    pos_9 = [cur_pos[0] + 2, cur_pos[1] + 2]
    pos_10 = [cur_pos[0] + 3, cur_pos[1] + 3]
    pos_11 = [cur_pos[0] - 2, cur_pos[1] - 2]
    pos_12 = [cur_pos[0] - 3, cur_pos[1] - 3]
    pos_13 = [cur_pos[0] - 1, cur_pos[1] - 1]
    pos_14 = [cur_pos[0] + 1, cur_pos[1] + 1]

    New_Position_list = [pos_1,pos_2,pos_3,pos_4,pos_5,pos_6,pos_7,pos_8,pos_9,pos_10,pos_11,pos_12,pos_13,pos_14]
    #print("New Positions can be selected from",New_Position_list)
    min_cost_pos = check_min(New_Position_list,shape_vect)
    return min_cost_pos
   
def check_min(pos_list,shape_vect):
    errorcost_x = [];  errorcost_y = []

    for i in range(len(pos_list)):
        x = round(shape_vect[0] - pos_list[i][0], 3)  
        y = round(shape_vect[1] - pos_list[i][1], 3)

        errorcost_x.append(round(x**2,3))
        errorcost_y.append(round(y**2,3))
        
    minimum_x = min(errorcost_x);   minimum_y = min(errorcost_y)

    for i in range(len(pos_list)):
        if errorcost_x[i] == minimum_x:
            #print("Min cost for x is at Position ",i,"and \n",pos_list[i][0])
            pos_x =  pos_list[i][0]
        if errorcost_y[i] == minimum_y:
            #print("Min cost for y is at Position ",i,"and \n",pos_list[i][1])
            pos_y =  pos_list[i][1]
    return [pos_x,pos_y]


def plot_trjectory(Current_Pos):
    plt.plot(CurrentPos[1][0],CurrentPos[1][1],color='red', marker='^')
    plt.plot(CurrentPos[2][0],CurrentPos[2][1],color='green', marker='_')
    plt.plot(CurrentPos[3][0],CurrentPos[3][1],color='blue', marker='^')
    plt.legend(['UAV 1','UAV 2', 'UAV 3'])
    plt.pause(0.01)
   
def plot_labels():
    plt.figure()
    plt.title('UAV Position', fontsize=14)
    plt.xlabel('X Direction', fontsize=14)
    plt.ylabel('Y Direction', fontsize=14)
    plt.plot(2,10,color='red', marker='s',markersize=15)
    plt.plot(-6,0,color='green', marker='s',markersize=15)
    plt.plot(-4,10,color='blue', marker='s',markersize=15)

def plot_Final_pos(CurrentPos):
    plt.plot(CurrentPos[1][0],CurrentPos[1][1],color='black', marker='X')
    plt.plot(CurrentPos[2][0],CurrentPos[2][1],color='black', marker='X')
    plt.plot(CurrentPos[3][0],CurrentPos[3][1],color='black', marker='X')
    plt.pause(0.01)

###########################################################################################

airsim.wait_key('Press any key to hover ') # Hover
for i in range(len(name)):
    client.moveByVelocityZAsync(0,0,alt,5,airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(False, 90),vehicle_name = name[i])

print("All UAVs are hovering at height ",abs(alt))
time.sleep(2)

 
#for x in range(numUAV):    CurrentPos =  readposition(x)  # reads the initial  position
 

airsim.wait_key('Press any key to Enter into the loop')

plot_labels()
k = 0; 
while (k<40):
    new_pos = []
    for i in range(numUAV): CurrentPos = readposition(i,new_pos)  # reads the current position     
    print("Current Position is: ", CurrentPos)        
    time.sleep(0.1)
    
    for i in range(1,numUAV):
            shape_vector = SearchSqShape_Vectors(i)
            newposition =  CostFunction(CurrentPos[i],shape_vector)
            time.sleep(0.1)

            print("New position Selected for Drone",i," is : \n", newposition)
            #airsim.wait_key('Press any key to move to new position')
            client.moveToPositionAsync(newposition[0], newposition[1], CurrentPos[0][2], 1.5, 5, vehicle_name = name[i])
  
    plot_trjectory(CurrentPos)    
    k = k+1   
    
plot_Final_pos(CurrentPos)


#########################################################################################
#----------------------   Exit         --------------------------------------------------
airsim.wait_key('Press any key to reset to original state')
new_pos = [] 
for x in range(4):    CurrentPos =  readposition(x,new_pos)  # reads the current position
print("Current Position is: ", CurrentPos)

for i in range(numUAV):
    client.goHomeAsync(vehicle_name = name[i])
    client.armDisarm(False, name[i])
    client.enableApiControl(False, name[i])
    
client.reset
