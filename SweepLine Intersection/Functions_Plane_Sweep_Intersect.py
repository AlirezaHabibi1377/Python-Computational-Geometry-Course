# All modules used for the Intersection program
import random
import matplotlib.pyplot as plt
import numpy as np
import bisect as bi
import tkinter as tk
from tkinter import filedialog

# ============================================

# Initialize global variables
slope , arzo , line = [] , [] , []
all_intersect_points , all_intersect_lines = [] , []

def Create_random_line(number = 5 , lower = 1 , upper = 40):
    '''
    Create a list of random lines.
    Each line is defined by two endpoints with random coordinates.
    '''
    
    for _ in range(number):
        
        line.append([[random.randint(lower,upper),
        random.randint(lower,upper)],
        [random.randint(lower,upper),
        random.randint(lower,upper)]])
     
    return line

def plot_all_point_intersect():
    '''
    Plot all intersection points.
    '''
    
    for iter in range(len(all_intersect_points)):
        plt.plot(all_intersect_points[iter][0] , all_intersect_points[iter][1] , 'o')
        plt.pause(0.3)

    plt.show()

def Read_Textfile_and_Create_Line():    
    '''
    Read line data from a text file. Each line in the file defines a line with four coordinates.
    '''
    root = tk.Tk()
    file_path = filedialog.askopenfilename(title="Convex Hull",filetypes = (("Text","*.txt"),("all files","*.*")))
        
    p1 = []
    p2 = []
    
    with open(file_path) as f:
       for linee in f:
           x1 , y1 , x2 , y2 = linee.split()
           p1.append(float(x1))
           p1.append(float(y1))
           p2.append(float(x2))
           p2.append(float(y2))
           l = [p1 , p2]
           line.append(l)
           p1 = []
           p2 = []
Read_Textfile_and_Create_Line() # Call function to read file and create lines

T = []

def Calculate_Slope_Plot_Lines():
    '''
    Calculate slopes and intercepts for all lines and plot the lines.
    '''
    for il in range(len(line)):
            
        X_point = [line[il][0][0] , line[il][1][0]]
        Y_point = [line[il][0][1] , line[il][1][1]]

        if Y_point[0] != Y_point[1]:
            slope.append(np.polyfit(X_point , Y_point , 1)[0])
            arzo.append(np.polyfit( X_point , Y_point , 1)[1])

        plt.plot(X_point , Y_point ,linewidth=2 , c ='m')
        plt.pause(0.5)
Calculate_Slope_Plot_Lines() # Call function to calculate slopes and plot lines

def Intersect_point(l1 , l2):
    '''
    Calculate the intersection point of two lines if they intersect within their segments.
    '''

    intersection_p = []
    
    point1_l1_x , point2_l1_x = l1[0][0] , l1[1][0]
    point1_l2_x , point2_l2_x = l2[0][0] , l2[1][0]

    point1_l1_y , point2_l1_y = l1[0][1] , l1[1][1]
    point1_l2_y , point2_l2_y = l2[0][1] , l2[1][1]

    if max(point1_l1_x , point2_l1_x) >= min(point1_l2_x , point2_l2_x) and max(point1_l2_x , point2_l2_x) >= min(point1_l1_x , point2_l1_x) :

        m_1 , b_1 = np.polyfit([point1_l1_x , point2_l1_x] , [point1_l1_y , point2_l1_y] , 1)[0] , np.polyfit([point1_l1_x , point2_l1_x] , [point1_l1_y , point2_l1_y] , 1)[1]
        m_2 , b_2 = np.polyfit([point1_l2_x , point2_l2_x] , [point1_l2_y , point2_l2_y] , 1)[0] , np.polyfit([point1_l2_x , point2_l2_x] , [point1_l2_y , point2_l2_y] , 1)[1]

        if m_1 != m_2:

            x_between = round((b_2-b_1)/(m_1-m_2), 3)

            num1 = max(min(point1_l1_x , point2_l1_x) , min(point1_l2_x , point2_l2_x))
            num2 = min(max(point1_l1_x , point2_l1_x) , max(point1_l2_x , point2_l2_x))

            if num1 <= x_between <= num2:
                y_between = round(m_1*x_between+b_1, 3)
                intersection_p = [x_between, y_between, -1, 2] 
    return intersection_p

def sort(lineme):
    '''
    Create and sort the event queue based on x and y coordinates.
    Define the type of endpoint (start or end of line).
    '''
    Q = []
    for i in range(len(lineme)):
        lineme[i][0].append(i) , lineme[i][1].append(i)
        Q.append(lineme[i][0]) , Q.append(lineme[i][1])

    Q.sort() , Q.sort(key=lambda x: x[1], reverse=True)
    # print('Q: ',Q)

    check = [0]*len(lineme)
    for i in range(len(Q)):

        if check[Q[i][2]] == 0:
            Q[i].append(0)
            check[Q[i][2]] = 1

        else:
            Q[i].append(1)
    
    return Q

def HandleEventPoint(point , eventQ , statusT):
    '''
    Calculate the U(p) , L(p) , C(p) :
    U(p) ---> all segment of that point(p) is upper endpoint
    L(p) ---> all segment of that point(p) is lower endpoint
    C(p) ---> all segment of that point(p) is intersect point
    and the 3 condition :  
    1) L(p) U U(p) U C(p)
    2) U(p) U C(p)
    3) L(p) U C(p)
    and then uppdated StatusT
    
    Determine the upper, lower, and concurrent points for the current event.
    Update the status list accordingly.
    '''
    Lp = []
    Up = []
    Cp = []

    for i in range(len(eventQ)):

        actual_Q  = eventQ[i][0:2] 
        actual_num_Q = eventQ[i][2]
        actual_P = point[0:2]

        if np.array_equal(actual_Q, actual_P) and eventQ[i][3] == 0:
            Up.append(actual_num_Q) 

        for it in range(len(statusT)):


            if statusT[it] == actual_num_Q and eventQ[i][3] == 1 and np.array_equal(actual_Q, actual_P):
                Lp.append(actual_num_Q)

            ac = slope[statusT[it]]
            bc = arzo[statusT[it]]
            yc = ac * point[0] + bc
            if (abs(round(yc,3) - point[1]) < 0.001):
                try:
                    Cp.index(statusT[it])
                except Exception:
                    try:
                        Lp.index(statusT[it])
                    except Exception:
                        Cp.append(statusT[it])
    Cp.reverse()
    return Up , Lp , Cp

def FindNewEvent(point , iter , eventQ):
    '''
    check of the Point p is below of Sweepline and right of point p
    and the append to eventQ

    Add new intersection points to the event queue if they lie below the sweep line.
    ''' 
    try:
        Q_new = [x[0:4] for x in eventQ]
        equ = Q_new.index(iter[0:4])
    # below of Sweepline
    except:
        if iter[1] <= point[1] and iter != []:  
            if iter[1] == point[1]:
                # the right of Point p
                if iter[0] > point[0]:
                    # calculate y coordinates                 
                    yq = [x[1] for x in eventQ]
                    # sort the y coordinates(lower to upper)  
                    yq.reverse()
                    # calculate index of item 
                    bb = bi.bisect(yq, iter[1])
                    # because of yq is inverse : len(yq) - bb
                    eventQ.insert(len(yq)-bb, iter)
            else:
                # extract y coordinates
                yq = [x[1] for x in eventQ]
                # sort the y coordinates(lower to upper)
                yq.reverse()
                # calculate index of item
                bb = bi.bisect(yq, iter[1])  
                # because of yq is inverse : len(yq) - bb
                eventQ.insert(len(yq) - bb, iter)
    return eventQ  

def Sort_new_statusT(T , new , point , slope , arzo):
    '''
    sorted new status to StatusT :
    T ---> StatusT
    new ---> line count that intend append to Status
    Point ---> Point
    slope ---> dip of line
    arzo ---> Width of origin line

    Insert a new line segment into the status list while maintaining order.
    '''
    xt = []
    if T is None:
        T.extend([new])
    else:
        for iter in T:
            xt.append(round((point[1] - arzo[iter]) / slope[iter], 2))
        if point[2] == -1:
            indT = bi.bisect_right(xt, round(point[0],2))
        else:  
            # find the intersect of line with line of me
            # and find the x coordinate of line, can find the order of lines
            try:  
                # line of intersect with line of me in upper endpoint
                same_line = T[xt.index(point[0])]  
                # decline y coordinate
                new_y = point[1]- (1e-10)  
                # find x coordinate in StatusT
                xSameLine = (new_y - arzo[same_line]) / slope[same_line]  
                # find x coordinate that intend append to StatusT
                my_x = (new_y - arzo[new]) / slope[new] 
                # new line is right
                if my_x > xSameLine:  # khate jadid samte rast ast
                    
                    indT = bi.bisect_right (xt,point[0])
                # new line is left    
                else: 
                    
                    indT = bi.bisect_left (xt,point[0])
            # not corresponding
            except: 
                indT = bi.bisect_left (xt,point[0])
        T.insert(indT, new)
    return T

def UpperPoint_calculate_intersect(point, T,eventQ ,line , all_intersect_points, lR):
    '''
    Calculate the intersects for upper endpoints:
    point ---> points
    T ---> StatusT
    eventQ ---> Q
    line ---> lines
    Intersect_Point ---> Intersect_Point
    IR ---> check the line intersect with right line or left right 
    
    Calculate intersections for upper endpoints of lines.
    '''
    try:
        ind = T.index(point[2])
        # (ind+lR) = 0 : last ind = -1 that mistake
        if ((ind+lR) >= 0): 
            sL = line[T[ind + lR]]
            s = line[T[ind]]
            inter = Intersect_point(sL, s)
            if inter != []:
                Q = FindNewEvent(point, inter, eventQ)
                try:
                    repeat_intersect = all_intersect_points.index(inter[0:2])
                except Exception:
                    all_intersect_points.append(inter[0:2])
                    all_intersect_lines.append([sL[0][2], s[0][2]])
    except Exception:
        n = []

    return all_intersect_points, all_intersect_lines

def IntersectPoint_calculate_intersect(point,T , eventQ, line, all_intersect_points, inds, lR, lR2):
    '''
    Calculate the intersects for intersect endpoints:
    point ---> points
    T ---> StatusT
    eventQ ---> eventQ
    line ---> lines 
    Intersects_Point ---> Intersects_Point
    inds ---> index of mobility or move , 
    lR ---> -1 and 0 if for left of point me , 
    lR2 ---> 2 and 1 if for right of point me
    '''
    try:
        if (inds + lR) >= 0:
            sL = line[T[inds + lR]]
            s = line[T[inds + lR2]]
            inter = Intersect_point(sL, s)
            if inter != []:
                Q = FindNewEvent(point, inter, eventQ)
                try:
                    repeat_intersect = all_intersect_points.index(inter[0:2])
                except Exception:
                    all_intersect_points.append(inter[0:2])
                    all_intersect_lines.append([sL[0][2], s[0][2]])
    except Exception:
        n = []

    return all_intersect_points, all_intersect_lines