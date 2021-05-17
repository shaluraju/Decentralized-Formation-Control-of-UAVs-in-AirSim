import airsim
import cv2
import numpy as np
import os
import pprint
import setup_path 
import tempfile
import time
import matplotlib.pyplot as plt
import math

numUAV = 4; name = []; s = []
tout = 10; spd = 2; alt = -10
min_dist = 2

pos0 = np.zeros((numUAV,3))  # Assigning Global Position
for i in range(numUAV):
    pos0[i,0] = 16.0 - 2.0*i
    pos0[i,1] = 12.0
    pos0[i,2] = -10.0


client = airsim.MultirotorClient()
client.confirmConnection()
for i in range(numUAV ):        # Arming all the Vehicles
    name.append("UAV" + str(i))
    client.enableApiControl(True, name[i])
    client.armDisarm(True, name[i])
    #print("UAV", i, "have been initialized.")
print("All UAVs have been initialized.")

###############################################################################
#--------------------     Functions       --------------------------------------   
################################################################################

def plot_trjectory(Current_Pos):
    plt.plot(CurrentPos[1][0],CurrentPos[1][1],abs(CurrentPos[1][2]),color='red', marker=".")
    plt.plot(CurrentPos[2][0],CurrentPos[2][1],abs(CurrentPos[2][2]),color='green', marker='.')
    plt.plot(CurrentPos[3][0],CurrentPos[3][1],abs(CurrentPos[3][2]),color='blue', marker='.')
    plt.legend(['UAV 1','UAV 2', 'UAV 3'])
    plt.pause(0.01)
   
def plot_labels(pos):
    plt.figure()
    plt.axes(projection='3d')
    plt.title('UAV Position', fontsize=14)
    plt.xlabel('X Direction', fontsize=12)
    plt.ylabel('Y Direction', fontsize=12)
    #plt.zlabel('Z Direction', fontsize=12)
    plt.plot(16,12,abs(pos),color='red', marker='s',markersize=10)
    plt.plot(16,22,abs(pos),color='green', marker='s',markersize=10)
    plt.plot(26,12,abs(pos),color='blue', marker='s',markersize=10)
    plt.plot(26,22,abs(pos),color='black', marker='s',markersize=10)

def plot_Final_pos(CurrentPos):
    plt.plot(CurrentPos[0][0],CurrentPos[0][1],abs(CurrentPos[0][2]),color='black', marker='X')
    plt.plot(CurrentPos[1][0],CurrentPos[1][1],abs(CurrentPos[1][2]),color='black', marker='X')
    plt.plot(CurrentPos[2][0],CurrentPos[2][1],abs(CurrentPos[2][2]),color='black', marker='X')
    plt.plot(CurrentPos[3][0],CurrentPos[3][1],abs(CurrentPos[3][2]),color='black', marker='X')
    plt.pause(0.01)

def readposition(n,s):     # Reads the Current Position
    #s.clear()
    x = client.simGetGroundTruthKinematics(vehicle_name = name[n]).position.x_val
    y = client.simGetGroundTruthKinematics(vehicle_name = name[n]).position.y_val
    z = client.simGetGroundTruthKinematics(vehicle_name = name[n]).position.z_val
    #yaw = client.simGetGroundTruthKinematics(vehicle_name = name[n]).position
    d = [round(x,3),round(y,3),round(z,3)]
    s.append(d)
    return s

def ShapeVectorSquare(y,key,pos): # defining initial Shape vector for Square Shape Formation 
    x = pos[0][0] - pos[1][0]
    key = str(key)
    shapevec = {"S10":[2, 10],"S01":[0,-y],"S20": [14, 0],"S02": [-y,-y],"S21":[y,0],"S12":[-y,0],"S30":[16, 10],"S03":[-y,0],"S31":[y,-y],"S13":[-y,y],"S32":[0,-y],"S23":[0,y]}
    #shapevec = {"S10":[x, y],"S01":[0,-y],"S20": [x*2-y, 0],"S02": [-y,-y],"S21":[y,0],"S12":[-y,0],"S30":[x*3-y, y],"S03":[-y,0],"S31":[y,-y],"S13":[-y,y],"S32":[0,-y],"S23":[0,y]}
    return shapevec[key]
     
def SearchSqShape_Vectors(y,pos0,z):        # Picks the shape vector each quadrotor
    Sij = "S"+str(y)+str(0)
    shape_vect = ShapeVectorSquare(10,Sij,pos0)
    shape_vect.append(z)
    print("Shape Vector Selected is: ",shape_vect)
    return shape_vect

    
def CostFunction(cur_pos, shape_vect):       # List of Possible next Positions
    pos_1 = [cur_pos[0] + 2, cur_pos[1], cur_pos[2]]
    pos_2 = [cur_pos[0] - 2, cur_pos[1], cur_pos[2]]
    pos_3 = [cur_pos[0] + 3, cur_pos[1], cur_pos[2]]
    pos_4 = [cur_pos[0] - 3, cur_pos[1], cur_pos[2]]
    pos_5 = [cur_pos[0], cur_pos[1] + 2, cur_pos[2]]
    pos_6 = [cur_pos[0], cur_pos[1] - 2, cur_pos[2]]
    pos_7 = [cur_pos[0], cur_pos[1] + 3, cur_pos[2]]
    pos_8 = [cur_pos[0], cur_pos[1] - 3, cur_pos[2]]
    pos_9 = [cur_pos[0] + 2, cur_pos[1] + 2, cur_pos[2]]
    pos_10 = [cur_pos[0] + 3, cur_pos[1] + 3, cur_pos[2]]
    pos_11 = [cur_pos[0] - 2, cur_pos[1] - 2, cur_pos[2]]
    pos_12 = [cur_pos[0] - 3, cur_pos[1] - 3, cur_pos[2]]
    pos_13 = [cur_pos[0] - 1, cur_pos[1] - 1, cur_pos[2]]
    pos_14 = [cur_pos[0] + 1, cur_pos[1] + 1, cur_pos[2]]
    pos_15 = [cur_pos[0] - 0.5, cur_pos[1] - 0.5, cur_pos[2]]
    pos_16 = [cur_pos[0] + 0.5, cur_pos[1] + 0.5, cur_pos[2]]
    
    New_Position_list = [pos_1,pos_2,pos_3,pos_4,pos_5,pos_6,pos_7,pos_8,pos_9,pos_10,pos_11,pos_12,pos_13,pos_14,pos_15,pos_16]
    
    #print("New Positions can be selected from",New_Position_list)
    min_cost_pos = check_min(New_Position_list,shape_vect)
    return min_cost_pos

def check_min(pos_list,shape_vect):         # Applies the COst Function and finds minimum
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
    return [pos_x,pos_y,pos_list[0][2]]

def Trajectory(cur_pos, shape_vect):   # Builds the Trajectory
    Trajectory_Drone = []
    while (cur_pos != shape_vect):
        cur_pos = CostFunction(cur_pos, shape_vect)
        Trajectory_Drone.append(cur_pos)
    Trajectory_Drone.append(shape_vect)
    #print(Trajectory_Drone)
    return Trajectory_Drone

def Check_Time_Steps(Traj_Dr_1,Traj_Dr_2,Traj_Dr_3):     # Checks the number of steps in eac trajectory
                                                         # and makes them equal 
    L_Traj_1 = len(Traj_Dr_1)
    L_Traj_2 = len(Traj_Dr_2)
    L_Traj_3 = len(Traj_Dr_3)
    N_time_steps = max(L_Traj_1,L_Traj_2,L_Traj_3)
    print("Max Time steps: \n",N_time_steps)
    if L_Traj_1 < N_time_steps:
        for i in range(L_Traj_1, N_time_steps):
            Traj_Dr_1.append( Traj_Dr_1[L_Traj_1 - 1] ) 

    if L_Traj_2 < N_time_steps:
        for i in range(L_Traj_2, N_time_steps):
            Traj_Dr_2.append( Traj_Dr_2[L_Traj_2 - 1] ) 

    if L_Traj_3 < N_time_steps:
        for i in range(L_Traj_3, N_time_steps):
            Traj_Dr_3.append( Traj_Dr_3[L_Traj_3 - 1] ) 

    return Traj_Dr_1, Traj_Dr_2, Traj_Dr_3
   

def Collision_free(Traj_Dr_1, Traj_Dr_2, Traj_Dr_3):    # Makes the path Collision Free
    Traj_Dr_0 = [pos0[0,0],pos0[0,1]]
    CF = 2.5
    for i in range(len(Traj_Dr_1)):

        d10 = round(math.sqrt((Traj_Dr_1[i][0] - Traj_Dr_0[0])**2 + (Traj_Dr_1[i][1] - Traj_Dr_0[1])**2),1)
        d12 = round(math.sqrt((Traj_Dr_1[i][0] - Traj_Dr_2[i][0])**2 + (Traj_Dr_1[i][1] - Traj_Dr_2[i][1])**2),1)
        d13 = round(math.sqrt((Traj_Dr_1[i][0] - Traj_Dr_3[i][0])**2 + (Traj_Dr_1[i][1] - Traj_Dr_3[i][1])**2),1)
        
        d20 = round(math.sqrt((Traj_Dr_2[i][0] - Traj_Dr_0[0])**2 + (Traj_Dr_2[i][1] - Traj_Dr_0[1])**2),1)        
        d23 = round(math.sqrt((Traj_Dr_2[i][0] - Traj_Dr_3[i][0])**2 + (Traj_Dr_2[i][1] - Traj_Dr_3[i][1])**2),1)

        d30 = round(math.sqrt((Traj_Dr_3[i][0] - Traj_Dr_0[0])**2 + (Traj_Dr_3[i][1] - Traj_Dr_0[1])**2),1)        

        print("d10, d12, d13, d20, d23, d30 are:")
                
        d = [d10, d12, d13, d20, d23, d30]
        print(d)
        for j in range(len(d)):
            if d[j] == 0:
                d[j] = 0.01


        if d[0] < 2.5 or d[1] < 2.5 or d[2] < 2.5:
            Traj_Dr_1[i][0] = Traj_Dr_1[i][0] + (1/min(d[0],d[1]))*CF
            Traj_Dr_1[i][1] = Traj_Dr_1[i][1] + (1/min(d[0],d[1]))*CF

        if d[3] < 2.5 or d[4] < 2.5 :
            Traj_Dr_2[i][0] = Traj_Dr_2[i][0] + (1/min(d[3],d[4]))*CF
            Traj_Dr_2[i][1] = Traj_Dr_2[i][1] + (1/min(d[3],d[4]))*CF
  
        if d[5] < 2.5:
            Traj_Dr_3[i][0] = Traj_Dr_3[i][0] + (1/d[5])*CF
            Traj_Dr_3[i][1] = Traj_Dr_3[i][1] + (1/d[5])*CF

    return Traj_Dr_1, Traj_Dr_2, Traj_Dr_3

def local_to_Global(trajectory,n):      # Changes Local Poition to Global Position
    Global_traj = []
    for i in range(len(trajectory)):
        x = pos0[n][0] + trajectory[i][0]
        y = pos0[n][1] + trajectory[i][1]
        Global_traj.append([x,y])
    return Global_traj

def Global_to_local(trajectory,n):      # Changes gLobal Poition to local Position
    local_traj = []
    for i in range(len(trajectory)):
        x = trajectory[i][0] - pos0[n][0]
        y = trajectory[i][1] - pos0[n][1]
        local_traj.append([x,y])
    return local_traj

def local_to_Global_position(cur_pos):
    global_traj = []
    for i in range(4):
        x = cur_pos[i][0] + pos0[i][0]
        y = cur_pos[i][1] + pos0[i][1]
        z = cur_pos[i][2]
        global_traj.append([x,y,z])
    return global_traj


###########################################################################################
# -----------------------------    Actual Program  -----------------------------------
###########################################################################################

airsim.wait_key('Press any key to hover ') # Hover
for i in range(len(name)):
    client.moveByVelocityZAsync(0,0,alt,5,airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(False, 90),vehicle_name = name[i])

print("All UAVs are hovering at height ",abs(alt))
time.sleep(2)
airsim.wait_key('Press any key to Enter into the loop')


new_pos = []
for i in range(numUAV): CurrentPos = readposition(i,new_pos)  # reads the current position     
print("Current Position is: ", CurrentPos)        
time.sleep(0.1)
plot_labels(CurrentPos[0][2])
global_pos = local_to_Global_position(CurrentPos)
plot_trjectory(global_pos)

# Trajectory For the Drones
shape_vector_1 = SearchSqShape_Vectors(1,pos0,CurrentPos[1][2])
Traj_Drone_1 =  Trajectory(CurrentPos[1],shape_vector_1)
Glo_Traj_1 =  local_to_Global(Traj_Drone_1,1)

shape_vector_2 = SearchSqShape_Vectors(2,pos0,CurrentPos[2][2])
Traj_Drone_2 =  Trajectory(CurrentPos[2],shape_vector_2)
Glo_Traj_2 =  local_to_Global(Traj_Drone_2,2)

shape_vector_3 = SearchSqShape_Vectors(3,pos0,CurrentPos[3][2])
Traj_Drone_3 =  Trajectory(CurrentPos[3],shape_vector_3)
Glo_Traj_3 =  local_to_Global(Traj_Drone_3,3)

# To make the Trajectory have equal time steps
Traj_Dr_1, Traj_Dr_2, Traj_Dr_3 = Check_Time_Steps(Glo_Traj_1, Glo_Traj_2, Glo_Traj_3)  
print("Trajectory after Checking")

# Checking For Collisions
path_1, path_2, path_3 = Collision_free(Traj_Dr_1, Traj_Dr_2, Traj_Dr_3)
print("Collision free path")

# Converting From Global to Local
Traj_Dr_1 =  Global_to_local(path_1,1)
Traj_Dr_2 =  Global_to_local(path_2,2)
Traj_Dr_3 =  Global_to_local(path_3,3)

# Loop to Follow Trajectory
for i in range(len(Traj_Drone_1)):
    client.moveToPositionAsync(Traj_Dr_1[i][0], Traj_Dr_1[i][1], CurrentPos[0][2], 3.5, 10, vehicle_name = name[1])
    client.moveToPositionAsync(Traj_Dr_2[i][0], Traj_Dr_2[i][1], CurrentPos[0][2], 3.5, 10, vehicle_name = name[2])
    client.moveToPositionAsync(Traj_Dr_3[i][0], Traj_Dr_3[i][1], CurrentPos[0][2], 3.5, 10, vehicle_name = name[3])
    time.sleep(0.5)
    CurrentPos.clear()
    for j in range(4): 
        CurrentPos = readposition(j,new_pos)
        
    print("Current Position is", CurrentPos)
    global_pos = local_to_Global_position(CurrentPos)
    plot_trjectory(global_pos)
    global_pos.clear()

k = 0; 
while (k<12):
        new_pos = []
        for i in range(numUAV): CurrentPos = readposition(i,new_pos)  # reads the current position     
        print("Current Position is: ", CurrentPos)        
        time.sleep(0.1)
        global_pos = local_to_Global_position(CurrentPos)
        print("global Position is", global_pos)    
        for i in range(1,numUAV):

            shape_vector = SearchSqShape_Vectors(i,pos0,CurrentPos[0][2])
            newposition =  CostFunction(CurrentPos[i],shape_vector)
            time.sleep(0.1)
            print("New position Selected for Drone",i," is : \n", newposition)
            client.moveToPositionAsync(newposition[0], newposition[1], CurrentPos[0][2], 1.5, 5, vehicle_name = name[i])

        plot_trjectory(global_pos)    
        k = k+1
        global_pos.clear()
global_pos = local_to_Global_position(CurrentPos)
print("global Position is", global_pos)
plot_Final_pos(global_pos)


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
plt.show()

































##########################################################################################
"""
while (True):
    read_new_pos = [];  
    for x in range(numUAV):    CurrentPos =  readposition(x,read_new_pos)  # reads the current position
    print("Current Position is: ", CurrentPos)
    for x in range(numUAV):
        if x > 0:
            y = x           
            Err = ShapeError(CurrentPos[y], CurrentPos, y)
            print("Error Matrices: ",Err)
            CF = CostFunction(Err[0],1)
            newposition = NewPosition(CF, Err[0])
    break 
print("New Position Choosen is : ",newposition)
airsim.wait_key('Press any key to move vehicle 4 to new position')
px = newposition[0]
py = newposition[1]
print("Px is: ",px)
print("Py is: ",py)


def IncreDecre_step(error, current_position):
    new_position = [0,0]
    if error[0] > 0:
        new_position[0] = round((current_position[0] + 0.1),3)
    elif error[0] < 0:
         new_position[0] = round((current_position[0] - 0.1),3)
    if error[1] > 0:
        new_position[1] = round((current_position[1] + 0.1),3)
    elif error[1] < 0:
         new_position[1] = round((current_position[1] - 0.1),3)

    return new_position

def ShapeofQuadrotors(): # let's you to choose a formation
    Formation_shape = ['square','triangle','circle']
    print("choose a formation shape among the listed shapes:", Formation_shape)
    choosen_shape = input()
    if (choosen_shape.lower() in Formation_shape): 
        print(choosen_shape, " is selected ") 
        return choosen_shape
    else: print("please choose a feasible shape")
    
def readyaw(n):
    client.moveByAngleZAsync(0,0,alt,30,5,vehicle_name = name[n]).join()
    state = client.getMultirotorState(vehicle_name = name[n]).rc_data.yaw
    print("state: %s" % pprint.pformat(state))

def VehicleState(n): # prints vehicle state
    state = client.getMultirotorState(vehicle_name = name[n])
    print("state is: %s" % pprint.pformat(state))

def ShapeVector(x,y): # returns shape vector b/w any two positions 
    Sh = []
    for i in range(len(name)):
        Sh.append(x[i]-y[i])
    #print(Sh)
    return Sh

def SearchSqShape_Vectors(position, set, y):
    for i in range(len(set)):
        if i!= y:
            Sij = "S"+str(y)+str(i)
            SV = ShapeVectorSquare(3.000,Sij)
            #print("Shape Vector selected is : \n",SV)
            error.append([position[0] - set[i][0] - SV[0]  ,position[1] - set[i][1] - SV[1]])
            #print("Error among drones is: \n",error)
    return shape_vect

def ShapeVectorSquare(y,key): # defining initial Shape vector 
    key = str(key)
    shapevec = {"S10":[0,y],"S01":[0,-y],"S20": [y,y],"S02": [-y,-y],"S21":[y,0],"S12":[-y,0],"S30":[y,0],"S03":[-y,0],"S31":[y,-y],"S13":[-y,y],"S32":[0,-y],"S23":[0,y]}
    #shapevec = [[0,y,0,0], [y,0,0,0],  [y,y,0,0], [y,0,0,0], [0,y,0,0]]
                    #S21        S31        S42        S43       S32
    return shapevec[key]

def check_min(pos_list,shape_vect):
    errorcost_x = [];  errorcost_y = []

    for i in range(len(pos_list)):
        x = shape_vect[0] - pos_list[i][0] 
        y = shape_vect[1] - pos_list[i][1] 
        print("error in position \n", x,y)
        errorcost_x.append(round(x**2,3))
        errorcost_y.append(round(y**2,3))
        print("Error Cost for ",i, "is\n",errorcost_x[i],errorcost_y[i])
        
    minimum_x = min(errorcost_x); print("minimum cost in x is",minimum_x)
    minimum_y = min(errorcost_y); print("minimum cost in x is",minimum_y)
    for i in range(len(pos_list)):
        if errorcost_x[i] == minimum_x:
            print("Min cost for x is at Position ",i,"and \n",pos_list[i][0])
            pos_x =  pos_list[i][0]
        if errorcost_y[i] == minimum_y:
            print("Min cost for y is at Position ",i,"and \n",pos_list[i][1])
            pos_y =  pos_list[i][1]
    return [pos_x,pos_y]
%%

    pos_1 = [cur_pos[0] + 0.5, cur_pos[1]]
    pos_2 = [cur_pos[0] - 0.5, cur_pos[1]]
    pos_3 = [cur_pos[0] + 1.0, cur_pos[1]]
    pos_4 = [cur_pos[0] - 1.0, cur_pos[1]]
    pos_5 = [cur_pos[0], cur_pos[1] + 0.5]
    pos_6 = [cur_pos[0], cur_pos[1] - 0.5]
    pos_7 = [cur_pos[0], cur_pos[1] + 1.0]
    pos_8 = [cur_pos[0], cur_pos[1] - 1.0]
    pos_9 = [cur_pos[0] + 0.5, cur_pos[1] + 0.5]
    pos_10 = [cur_pos[0] + 1.0, cur_pos[1] + 1.0]
    pos_11 = [cur_pos[0] - 0.5, cur_pos[1] - 0.5]
    pos_12 = [cur_pos[0] - 1.0, cur_pos[1] - 1.0]
    pos_13 = [cur_pos[0] - 0.01, cur_pos[1] - 0.01]
    pos_14 = [cur_pos[0] + 0.01, cur_pos[1] + 0.01]


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


        for i in range(numUAV):
        if i > 0: # not controlling the leader i.e UAV 1
            shape_vector = SearchSqShape_Vectors(i,pos0)
            newposition =  CostFunction(CurrentPos[i],shape_vector)
            time.sleep(0.1)

            print("New position Selected for Drone",i," is : \n", newposition)
            #airsim.wait_key('Press any key to move to new position')
            client.moveToPositionAsync(newposition[0], newposition[1], CurrentPos[0][2], 1.5, 5, vehicle_name = name[i])
  
    plot_trjectory(CurrentPos)  
"""