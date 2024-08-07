# ==== Created By : Alireza habibi - 810399023 & Ali rezaei - 810399034 =====

import matplotlib.pyplot as plt
import bisect as b
import shapely.geometry
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import copy

# ==================== Functions for Monotone Triangulation ====================

def create_polygon_with_InputData():
    '''
    Prompts the user to select a text file containing polygon vertices.
    The file should have vertex coordinates, each on a new line.
    
    Returns:
        polygon (list of lists): A list of vertices, where each vertex is a list containing x and y coordinates.
    '''
    polygon = [] 
    XY = []
    root = tk.Tk()
    file_path = filedialog.askopenfilename(title="Triangulate",filetypes = (("Text","*.txt"),("all files","*.*")))
    with open(file_path) as f:
        for line in f:
            x , y = line.split()
            XY.append(float(x))
            XY.append(float(y))
            polygon.append(XY)
            XY = []
    return polygon

def plot_polygone(input_Polygon):  
    '''
    Plots the input polygon using matplotlib.
    
    Parameters:
        input_Polygon (list of lists): A list of vertices where each vertex is a list containing x and y coordinates.
    '''
    px , py = [] , []
    for XY_poly in input_Polygon:
        px.append (XY_poly[0]) 
        py.append (XY_poly[1])
        plt.plot(XY_poly[0],XY_poly[1],'bo')

    plt.plot(px,py,'m-',linewidth = '2.2')

def Plot_diagonals(Diag):
    '''
    Plots the diagonals of the polygon that are detected by the algorithm.
    
    Parameters:
        Diag (list of lists): A list of diagonals where each diagonal is a list containing two vertices.
    '''
    for i_d in Diag:
        xx = [i_d[0][0],i_d[1][0]]
        yy = [i_d[0][1],i_d[1][1]]
        plt.plot(xx,yy,'g-')
        plt.pause(0.2)
       
def Geometry_polygon(polygon):
    '''
    Creates a Shapely Polygon geometry from a list of vertices.
    
    Parameters:
        polygon (list of lists): A list of vertices where each vertex is a list containing x and y coordinates.
    
    Returns:
        Polygon_Geometry (shapely.geometry.Polygon): Shapely Polygon geometry.
    '''
    Polygon_Geometry = shapely.geometry.Polygon(polygon) 
    return Polygon_Geometry

def Geometry_line(line):
    '''
    Creates a Shapely LineString geometry from a list of vertices.
    
    Parameters:
        line (list of lists): A list of vertices where each vertex is a list containing x and y coordinates.
    
    Returns:
        line (shapely.geometry.LineString): Shapely LineString geometry.
    '''
    line = shapely.geometry.LineString(line)
    return line

def Create_Sort_Q(polygon):
    '''
    Creates and sorts a list of polygon vertices by their x and y coordinates, removing duplicates.
    
    Parameters:
        polygon (list of lists): A list of vertices where each vertex is a list containing x and y coordinates.
    
    Returns:
        Sort_Q (list of lists): A sorted list of unique vertices.
    '''
    polygon.sort()
    polygon = [i for n, i in enumerate(polygon) if i not in polygon[:n]]
    polygon.sort(key=lambda x: x[1], reverse=True) 
    Sort_Q = polygon
    return Sort_Q
                
def Triangulation(Polygon):
    '''
    Performs triangulation on a monotone polygon and plots the results.
    
    Parameters:
        Polygon (list of lists): A list of vertices where each vertex is a list containing x and y coordinates.
    '''
    # Create a sample polygon from the input polygon
    polygon_sample = copy.deepcopy(Polygon)

    # Generate a sorted list of vertices
    Sort_Q = Create_Sort_Q(polygon_sample)
    
    # Initialize the stack with the first two vertices
    Stack=[Sort_Q[0],Sort_Q[1]]

    # List to store diagonals
    diags = []
    
    # Create Shapely Polygon geometry
    Polygon_Geometry = Geometry_polygon(polygon) 

    # Process vertices to create diagonals and triangulate the polygon
    for j in range(3,len(Sort_Q)):

        Vj = Sort_Q[j-1]

        # Find the index of the last vertex in the stack
        Vj_last_index = polygon.index(Stack[len(Stack)-1])

        # Find the index of the current vertex
        Vj_index = polygon.index(Vj)
        
        # Determine if the two vertices are on the same chain
        DeltaIndex = (Vj_last_index-Vj_index)

        # this section is for analysis that two vertex is on same chain or not 
        # (chain : reflexive chain - mutual chain)
        # and with comparision its detect Triangulation of polygon.
        # "if" section ====> two vertex is on same chain ---> we can drawing some diagonal (not all)
        # If vertices are on the same chain, process diagonals
        
        if abs(DeltaIndex) == 1 or abs(DeltaIndex) == len(polygon)-2:  
            
            # save latest vertex of Stack 
            lastest_delete = Stack[len(Stack)-1] 

            # delete Latest vertex of Stack
            del(Stack[len(Stack)-1])
            
            # move of end to start on Stack
            for item in Stack[::-1]: 
                
                # diagonal
                d = [Vj,item]  
                
                # Call function for line Geometry 
                line_Geometry = Geometry_line(d)
                line_intersection = Polygon_Geometry.intersection(line_Geometry)

                # Check for diagonal is valid or unvalid
                # if intersect of line and polygon = line ----> diagonal is valid(diagonal is in polygon)
                # othervise diagonal is unvalid
                # Validate diagonal based on intersection with the polygon
                if line_intersection == line_Geometry:
                    
                    lastest_delete = item
                    del(Stack[-1])

                    # append d to diagonal matrix
                    diags.append(d)  

                else:
                    break

            # Stack must Uppdated with append Vj and latest vertex    
            Stack.append(lastest_delete)  
            Stack.append(Vj)  

        # "else" section  : mutual chain(oposite of reflexive chain)
        else:  

            # append all diagonal to diagonal matrix because all of it valid
            # Process diagonals for vertices on different chains
            for items in Stack[1:]:
                d = [Vj,items]
                diags.append(d)
                
            # delete all of vertex of Stack
            del(Stack[:-1])
            
            # Uppdated Stack with append Vj to it
            Stack.append(Vj)
            
    # Handle the final vertices of the polygon
    for i_n in Stack[1:-1]:
        diags.append([Vj[-1],i_n])

    # Plot the polygon and its diagonals
    plot_polygone(polygon)

    # plot diagonal of monotone polygon and create Triangulation
    Plot_diagonals(diags)

    plt.show()

# Create the input polygon from user data
polygon = create_polygon_with_InputData()  
# polygon=[[2361.9771,1048.9401],[2416.6060,1147.4432],[2580.4924,1232.8916],[2818.0090,1232.8916],[3003.2719,1161.6846],[2915.3908,1021.6441],[2780.0063,962.3049],[2578.1172,962.3049],[2399.9798,1007.4027],[2361.9771,1048.9401]]

# Perform triangulation and plot results
Triangulation(polygon)   
