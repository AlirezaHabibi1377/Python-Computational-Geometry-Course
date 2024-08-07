# Import necessary modules
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog

def Determinant(self):
    """
    Compute the determinant of a 3x3 matrix formed from the last three points in the list.
    This is used to determine the convex hull.
    """
    a = [[self[-3][0] , self[-3][1] , 1],
    [self[-2][0] , self[-2][1] , 1],
    [self[-1][0] , self[-1][1] , 1]]
    return np.linalg.det(a)

def Draw_lines_between_points(self):
    """
    Plot lines connecting the given points, which are vertices of the convex hull.
    Points are marked in red, and lines are drawn in yellow.
    """
    cx = []
    cy = []
    for i in range(len(self)):
        cx.append(self[i][0]), cy.append(self[i][1])
    plt.plot(cx, cy,"r^") # Plot points in red
    plt.plot(cx, cy , "y") # Plot lines in yellow

def Draw_points(self):
    """
    Plot the input set of points.
    Points are marked in green.
    """
    cx = []
    cy = []
    for i in range(len(self)):
        cx.append(self[i][0]), cy.append(self[i][1])
    plt.plot(cx, cy,"g^") # Plot points in green

# Initialize lists to hold coordinates
coord = []
txtcoord = []

# Open a file dialog to browse and select a text file containing input points
root = tk.Tk()
file_path = filedialog.askopenfilename(title="Convex Hull",filetypes = (("Text","*.txt"),("all files","*.*")))

# Read the coordinates from the selected text file and store them in a list
with open(file_path) as f:
    for line in f:
        x , y = line.split()
        txtcoord.append(float(x))
        txtcoord.append(float(y))
        coord.append(txtcoord)
        txtcoord = []
print(coord)       

# Sort the coordinates lexicographically
coord_sort = sorted(coord)

# Initialize upper and lower lists for convex hull construction
l_upper = coord_sort[0:2]
l_lower = [coord_sort[-1],coord_sort[-2]]

# Construct the upper part of the convex hull       
for i_upp in range(2,len(coord_sort)):
    l_upper.append(coord_sort[i_upp])

    while (len(l_upper) > 2 and Determinant(l_upper) >= 0):
            
        l_upper.pop(-2)

# Construct the lower part of the convex hull       
for i_lower in range(len(coord_sort)-3 , -1 , -1):
    l_lower.append(coord_sort[i_lower])
     
    while (len(l_lower) > 2 and Determinant(l_lower) >= 0 ):

        l_lower.pop(-2)

# Plot the input points
Draw_points(coord_sort)

# Plot the upper part of the convex hull
Draw_lines_between_points(l_upper)

# Plot the lower part of the convex hull
Draw_lines_between_points(l_lower)
plt.title('* Convex Hull *')

# Combine the upper and lower hulls, remove duplicates, and sort the vertices
j = l_upper[:-1]+l_lower
g = sorted(j)

# Output the vertices that form the convex hull
print("Output ConvexHull: ", g[1:])
plt.show()