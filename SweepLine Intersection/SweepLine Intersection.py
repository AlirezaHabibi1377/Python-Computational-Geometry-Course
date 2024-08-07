# Import necessary modules
import matplotlib.pyplot as plt
import numpy as np  
import bisect as bi 
import tkinter as tk
from tkinter import filedialog
from Functions_Plane_Sweep_Intersect import *
# ============================================

# Initialize an empty list for the status of intersecting lines
T = []

# Sort the initial event queue based on the input line data
eventQ = sort(line)

# Process events from the event queue until it is empty
while eventQ != []:

    # Get the current event point from the event queue
    p = eventQ[0]
    
    # Handle the event point to get upper, lower, and concurrent points
    [Up , Lp , Cp] = HandleEventPoint(p , eventQ , T)

    # Create a combined list of upper, lower, and concurrent points
    uniuon_UpLpCp = list(Up)
    uniuon_UpLpCp.extend(Lp)
    uniuon_UpLpCp.extend(Cp)
    
    # Create a combined list of upper and concurrent points
    uniuon_UpCp = list(Cp)
    uniuon_UpCp.extend(Up)
    
    # Create a combined list of lower and concurrent points
    uniuon_LpCp = list(Lp)
    uniuon_LpCp.extend(Cp)

    # deleted duplicate in Lp , Cp , Up
    # Remove duplicates from the combined lists
    uniuon_UpLpCp = [iter for num, iter in enumerate(uniuon_UpLpCp) if iter not in uniuon_UpLpCp[:num]]
    uniuon_UpCp = [iter for num, iter in enumerate(uniuon_UpCp) if iter not in uniuon_UpCp[:num]]
    uniuon_LpCp = [iter for num, iter in enumerate(uniuon_LpCp) if iter not in uniuon_LpCp[:num]]

    # find the intersect point for U(p) U C(p) U L(p)
    # Check if the current event point is an intersection point
    if len(uniuon_UpLpCp) > 1:
        try:
            repeat_intersect = all_intersect_points.index(p[0:2])
        except Exception:
            all_intersect_points.append(p[0:2])
            all_intersect_lines.append(uniuon_UpLpCp) 

    # Store the current status of the list for comparison    
    StatusT_before = list(T)
    
    # Remove lower and concurrent points from the status list
    try:
        for i_lcp in range(len(uniuon_LpCp)):
            StatusT_d = T.index(uniuon_LpCp[i_lcp])
            del (T[StatusT_d])
    except Exception:
        n = []
    
    # Add new upper and concurrent points to the status list
    for i_rep in uniuon_UpCp:
        try:
            indexis = T.index(i_rep)
        
        # append the Up U Cp in StatusT
        except Exception:
             T = Sort_new_statusT(T, i_rep, p ,slope, arzo)
    
     # Handle lower endpoints                    
    if not uniuon_UpCp:
        
        try:
            ind = StatusT_before.index(p[2])
            if ind > 0:
                
                # Find the lines immediately before and after the current line
                sL = line[StatusT_before[ind - 1]]
                sR = line[StatusT_before[ind + 1]]
                
                # Calculate the intersection point between the neighboring lines
                interp = Intersect_point(sL, sR)
                if interp is not []:
                    
                    # Update the event queue with the new intersection point
                    eventQ = FindNewEvent(p, interp, eventQ)
                    try:
                        repeat_intersect = all_intersect_points.index(interp[0:2])
                    except Exception:
                        all_intersect_points.append(interp[0:2])
                        all_intersect_lines.append([sL[0][2], sR[0][2]])
        except Exception:
            n = []

    else:
        
        # Handle upper endpoints
        if p[2] != -1:
            l_uInter = -1
            r_uInter = 1
            uniuon_LpCp.reverse()

            # Calculate intersections for the upper points
            [all_intersect_points, all_intersect_lines] = UpperPoint_calculate_intersect(p , T, eventQ , line , all_intersect_points , l_uInter)
            [all_intersect_points, all_intersect_lines] = UpperPoint_calculate_intersect(p , T , eventQ , line , all_intersect_points , r_uInter)

        # Handle intersection points
        else:

            try:
                for inter in range(len(StatusT_before)):
                    if StatusT_before[inter] != T[inter]:
                        index_inter = inter  
                        break   
                    
                # Calculate intersections on the left and right of the first line
                [all_intersect_points, all_intersect_lines] = IntersectPoint_calculate_intersect(p , T , eventQ , line , all_intersect_points , inter , -1 , 0)
                [all_intersect_points, all_intersect_lines] = IntersectPoint_calculate_intersect(p ,  T, eventQ , line , all_intersect_points , inter , 2 , 1)

            except Exception:
                n = []
                
    # Remove the processed event point from the event queue
    del (eventQ[0])

# Sort and print the results
all_intersect_points.sort(key=lambda x: x[1] , reverse= True)
print("All intersect points (sorted by y-coordinate):", all_intersect_points)
print('*' * 60)
print("All intersect lines (format: [6,0] means line 6 intersects with line 0):", all_intersect_lines)

# Plot all intersection points
plot_all_point_intersect()