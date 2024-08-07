import matplotlib.pyplot as plt
import math
import numpy as np
import bisect as bi
import tkinter as tk
from tkinter import filedialog

#================= Functions for Voronoi Diagram ====================

def InputPointsCreate():
    '''
    Open a file dialog to select a text file containing coordinates of input points for the Voronoi diagram.
    Returns:
        List of points where each point is a list [x, y].
    '''
    point = []
    XY = []
    root = tk.Tk()
    file_path = filedialog.askopenfilename(
        title="Triangulate", filetypes=(("Text", "*.txt"), ("all files", "*.*")))
    with open(file_path) as f:
        for line in f:
            x, y = line.split()
            XY.append(float(x))
            XY.append(float(y))
            point.append(XY)
            XY = []
    return point

point = InputPointsCreate()

def PlotInputPoint(point):
    '''
    Plot the input points for the Voronoi diagram.
    
    Args:
        points (list): List of points where each point is a list [x, y].
    '''
    for i in point:
        x = i[0]
        y = i[1]
        plt.plot(x, y, 'b*')

PlotInputPoint(point)

class TreeStatus():
    '''
    Represents a node in the beachline status tree of the Voronoi diagram.
    Attributes:
        root: Root node of the tree.
        right: Right subtree.
        left: Left subtree.
        TreeParent: Parent node of the current node.
        TreeParent_dir: Direction of the parent node.
    '''
    def __init__(self,ro=None,r=None,l=None):

        # The root of tree
        self.root = ro

        # The segment is right of tree under
        self.right = r 

        # The segment is left of tree under
        self.left = l 

        self.TreeParent = None
        # The direction of treeParent
        self.TreeParent_dir = None

def IntersectionOf_ARC(Arc_1, Arc_2, POI):
    '''
    Calculate the intersection points of two arcs given a point of interest.
    
    Args:
        Arc_1 (list): Parameters [x, y] of the first arc.
        Arc_2 (list): Parameters [x, y] of the second arc.
        POI (list): Point [x, y] where arcs are intersecting.
        
    Returns:
        tuple: Coordinates of intersection points as lists [x1, x2], [y1, y2].
    '''

    # Coefficients for the equations of the arcs
    # this is factor of X in equation of First Arc
    Zx_1 = - (Arc_1[0]/(Arc_1[1]-POI[1]))

    # this is factor of X in equation of second Arc
    Zx_2 = - (Arc_2[0]/(Arc_2[1]-POI[1]))

    # this is factor of X^2 in equation of First Arc
    Zx2_1 = 1/(2*(Arc_1[1]-POI[1]))

    # this is factor of X^2 in equation of second Arc
    Zx2_2 = 1/(2*(Arc_2[1]-POI[1]))

    # The constant in equation of First Arc
    Const_1 = (Arc_1[0]**2 + Arc_1[1]**2 - POI[1]**2)/(2*(Arc_1[1]-POI[1]))
    
    # The constant in equation of second Arc
    Const_2 = (Arc_2[0]**2 + Arc_2[1]**2 - POI[1]**2)/(2*(Arc_2[1]-POI[1]))

    # the equation of Arc
    x1 = (-(Zx_1-Zx_2) + math.sqrt((Zx_1-Zx_2)**2-4*(Zx2_1-Zx2_2)*(Const_1-Const_2))) / (2*(Zx2_1-Zx2_2))
    x2 = (-(Zx_1-Zx_2) - math.sqrt((Zx_1-Zx_2)**2-4*(Zx2_1-Zx2_2)*(Const_1-Const_2))) / (2*(Zx2_1-Zx2_2))
    y1 = 1/(2*(Arc_1[1]-POI[1]))*(x1**2-2*Arc_1[0]
                               * x1+Arc_1[0]**2+Arc_1[1]**2-POI[1]**2)
    y2 = 1/(2*(Arc_1[1]-POI[1]))*(x2**2-2*Arc_1[0]
                               * x2+Arc_1[0]**2+Arc_1[1]**2-POI[1]**2)
    
    # this point is coordinate of Intersect of 2 Arc
    x = [[x1, y1], [x2, y2]] 

    # the soreted
    x.sort()
    xs = [x[0][0], x[1][0]]
    ys = [x[0][1], x[1][1]]

    return xs, ys

def search_in_searchTree(Status, POI):
    '''
    Search the beachline status tree to find the appropriate arc and its parent.

    Args:
        Status (TreeStatus): The current beachline status.
        POI (list): The point [x, y] to find the intersecting arc.

    Returns:
        tuple: Node containing the arc, parent node, direction of the parent, and the subtree traversed.
    '''
    i_node , i_subtree = 0 , 0
    
    subtree = []
    node = Status.root

    # the parent and direction of parent
    parent = None
    parent_dir = None

    # this segment work until we not arrive to leaf
    while type(node) is not list: 
        try:

            # the leaf of tree
            nodes = node.root
            
            print("nodes: ",nodes)
            
            # the First Arc
            arc1 = nodes[0]
            print("arc1: ",arc1)
            # the second Arc
            arc2 = nodes[1]
            print("arc2: ",arc2)
            # calculate Intersection of 2 Arc
            [x, y] = IntersectionOf_ARC(arc1, arc2, POI)
            print("x",x)
            print("y",y)
            # find solution of question that <pj,pi> or <pi,pj>
            if arc1[1] > arc2[1]:
                result = x[0]
            else:
                result = x[1]

            # find solution of question that move to Right of tree or Left of tree
            if POI[0] > result:

                # the parent node
                parent = node

                # direction of child of parent node
                parent_dir = 'r'
                node = node.right

                # find solution of question that move to which subtree for specification deltaH of tree
                if i_subtree == 0:  # taeine inke dar kodam zir derakht harekat mikonim --> baraye taeine ekhtelafe ertefa
                    subtree.append('r')
                    i_subtree = 1
                elif i_subtree == 1:
                    subtree.append('r')
                    i_subtree = 2
           
            # move to Left   
            else:

                # parent node
                parent = node

                # direction of child in parent node
                parent_dir = 'l'
                node = node.left

                # find solution of question that move to which subtree for specification deltaH of tree
                if i_subtree == 0:  # taeine inke dar kodam zir derakht harekat mikonim --> baraye taeine ekhtelafe ertefa
                    subtree.append('l')
                    i_subtree = 1
                elif i_subtree == 1:
                    subtree.append('l')
                    i_subtree = 2
            i_node = 1

        except:
            break
    if i_node == 0:
        node = node.root
    return node, parent, parent_dir, subtree

def RotateOfTree(tree, direction, Parent=None, Parent_dir=None, node12=None):
    '''
    Perform a rotation to balance the beachline status tree.

    Args:
        tree (TreeStatus): The current tree to rotate.
        direction (str): The direction of rotation ('l' or 'r').
        Parent (TreeStatus): Parent of the node being rotated.
        Parent_dir (str): Direction of the parent node.
        node12 (list): Optional, contains nodes to be adjusted during rotation.

    Returns:
        tuple: Updated parent nodes after rotation.
    '''

    if node12 is not None:
        # First node
        n1 = node12[0]  
        n1_parent = node12[1]
        n1_parent_dir = node12[2]

        # second node
        n2 = node12[3]
        n2_parent = node12[4]
        n2_parent_dir = node12[5]

    if Parent is None:
        if direction == 'l':
            t1 = tree
            left = t1.root
            right = t1.root.right.right
            lr = t1.root.right.left
            tree.root = t1.root.right
            tree.root.TreeParent = left.TreeParent
            tree.root.TreeParent_dir = left.TreeParent_dir
            tree.root.left = left
            tree.root.left.TreeParent = tree.root
            tree.root.left.TreeParent_dir = 'l'
            tree.root.right = right
            tree.root.left.right = lr
            if type(tree.root.left.right) is not list:
                tree.root.left.right.TreeParent = tree.root.left
                tree.root.left.right.TreeParent_dir = 'r'
            elif node12 is not None and tree.root.left.right == n1:
                n1_parent = tree.root.left
                n1_parent_dir = 'r'
            elif node12 is not None and tree.root.left.right == n2:
                n2_parent = tree.root.left
                n2_parent_dir = 'r'

        else:
            t1 = tree
            right = t1.root
            left = t1.root.left.left
            rl = t1.root.left.right
            tree.root = t1.root.left
            tree.root.TreeParent = right.TreeParent
            tree.root.TreeParent_dir = right.TreeParent_dir
            tree.root.right = right
            tree.root.right.TreeParent = tree.root
            tree.root.right.TreeParent_dir = 'r'
            tree.root.left = left
            tree.root.right.left = rl
            if type(tree.root.right.left) is not list:
                tree.root.right.left.TreeParent = tree.root.right
                tree.root.right.left.TreeParent_dir = 'l'
            elif node12 is not None and tree.root.right.left == n1:
                n1_parent = tree.root.right
                n1_parent_dir = 'l'
            elif node12 is not None and tree.root.right.left == n2:
                n2_parent = tree.root.right
                n2_parent_dir = 'l'

    # this segment for rotate of subtree
    else:
        if direction == 'l':
            t1 = tree
            if Parent_dir == 'l':
                left = Parent.left
                right = tree.right.right
                lr = Parent.left.right.left
                Parent.left = tree.right
                Parent.left.TreeParent = Parent
                Parent.left.TreeParent_dir = Parent_dir
                Parent.left.left = left
                Parent.left.left.TreeParent = Parent.left
                Parent.left.left.TreeParent_dir = 'l'
                Parent.left.right = right
                Parent.left.left.right = lr
                if type(Parent.left.left.right) is not list:
                    Parent.left.left.right.TreeParent = Parent.left.left
                    Parent.left.left.right.TreeParent_dir = 'r'
                elif Parent.left.left.right == n1:
                    n1_parent = Parent.left.left
                    n1_parent_dir = 'r'
                elif Parent.left.left.right == n2:
                    n2_parent = Parent.left.left
                    n2_parent_dir = 'r'

            else:
                left = Parent.right
                right = tree.right.right
                lr = Parent.right.right.left
                Parent.right = t1.right
                Parent.right.TreeParent = Parent
                Parent.right.TreeParent_dir = Parent_dir
                Parent.right.left = left
                Parent.right.left.TreeParent = Parent.right
                Parent.right.left.TreeParent_dir = 'l'
                Parent.right.right = right
                Parent.right.left.right = lr
                if type(Parent.right.left.right) is not list:
                    Parent.right.left.right.TreeParent = Parent.right.left
                    Parent.right.left.right.TreeParent_dir = 'r'
                elif Parent.right.left.right == n1:
                    n1_parent = Parent.right.left
                    n1_parent_dir = 'r'
                elif Parent.right.left.right == n2:
                    n2_parent = Parent.right.left
                    n2_parent_dir = 'r'

        else:
            if Parent_dir == 'l':
                right = Parent.left
                left = tree.left.left
                rl = Parent.left.left.right
                Parent.left = tree.left
                Parent.left.TreeParent = Parent
                Parent.left.TreeParent_dir = Parent_dir
                Parent.left.right = right
                Parent.left.right.TreeParent = Parent.left
                Parent.left.right.TreeParent_dir = 'r'
                Parent.left.left = left
                Parent.left.right.left = rl
                if type(Parent.left.right.left) is not list:
                    Parent.left.right.left.TreeParent = Parent.left.right
                    Parent.left.right.left.TreeParent_dir = 'l'
                elif Parent.left.right.left == n1:
                    n1_parent = Parent.left.right
                    n1_parent_dir = 'l'
                elif Parent.left.right.left == n2:
                    n2_parent = Parent.left.right
                    n2_parent_dir = 'l'

            else:
                right = Parent.right
                left = tree.left.left
                rl = Parent.right.left.right
                Parent.right = tree.left
                Parent.right.TreeParent = Parent
                Parent.right.TreeParent_dir = Parent_dir
                Parent.right.right = right
                Parent.right.right.TreeParent = Parent.right
                Parent.right.right.TreeParent_dir = 'r'
                Parent.right.left = left
                Parent.right.right.left = rl
                if type(Parent.right.right.left) is not list:
                    Parent.right.right.left.TreeParent = Parent.right.right
                    Parent.right.right.left.TreeParent_dir = 'l'
                elif Parent.right.right.left == n1:
                    n1_parent = Parent.right.right
                    n1_parent_dir = 'l'
                elif Parent.right.right.left == n2:
                    n2_parent = Parent.right.right
                    n2_parent_dir = 'l'

    if node12 is not None:
        return(n1_parent, n1_parent_dir, n2_parent, n2_parent_dir)

def insertToList(Arcs, Neighbor, Arc, POI):
    '''
    Updates the list of arcs and creates new neighbor triples with the given point of interest (POI).
    
    Args:
        Arcs (list): List of current arcs in the Voronoi diagram.
        Neighbor (list): List of neighboring arcs.
        Arc (list): The arc that is above the POI.
        POI (list): The point of interest to insert into the list.
        
    Returns:
        list: Updated list of neighbors, including new triples formed by inserting the POI.
    '''
    i_index = 0
    while True:

        # the find the indexs that equal with Arc of above point
        # Find the index of the arc in the Arcs list
        list_index = [i for i, n in enumerate(Arcs) if n == Arc[0]][i_index]
        next_index = list_index + 1

        # index of list is equal length of Arc negative 1
        # Handle edge cases for indices at boundaries
        if list_index == len(Arcs) - 1:
            next_index = list_index
        last_index = list_index - 1
        if list_index == 0:
            last_index = 0
        
        # this segment for equal neighbor of index with tree  
        # Check if the arc needs to be updated based on its neighbors
        if len(Arcs) == 1 or Arcs[next_index] == Neighbor[1] or Arcs[last_index] == Neighbor[0]:
            
            # Update the arcs list with the new POI and the new arc
            Arcs.insert(list_index+1, POI)
            Arcs.insert(list_index+2, Arc[0])

            # Create new triples for left and right neighbors
            seganeh1 = []  # Left neighbor triple
            seganeh2 = []  # Right neighbor triple

            # If not at the start, create left neighbor triple
            if list_index > 0:  
                seganeh1 = [Arcs[list_index-1], Arc[0], POI]

            # If not at the end, create right neighbor triple
            if list_index < len(Arcs)-3 and len(Arcs) > 3:
                seganeh2 = [POI, Arc[0], Arcs[list_index+3]]

            # Return the new neighbor triples
            seganeh = [seganeh1, seganeh2]
            return seganeh
        
        else:

            # Increment index to find the next matching arc
            i_index += 1  

def DeleteFromList(Arcs, Neighbor, CirclueEvent, Edges):
    '''
    Updates the list of arcs and creates new neighbor triples after deleting an arc due to a circle event.
    
    Args:
        Arcs (list): List of current arcs in the Voronoi diagram.
        Neighbor (list): List of neighboring arcs.
        CircleEvent (tuple): Contains circle event information (center, radius, point).
        Edges (list): List of edges for the Voronoi diagram.
        
    Returns:
        list: Updated list of neighbors, including new triples formed by removing the arc.
    '''

    i_index = 0

    # Extract information from the circle event
    p = CirclueEvent[2][0] # Point of interest (current site event)
    c = CirclueEvent[0] # Center of the circle
    d = CirclueEvent[2][-1] # Sweep line position

    while True:
        # Find the index of the arc in the Arcs list
        list_index = [i for i, n in enumerate(Arcs) if n == p[0]][i_index]
        next_index = list_index + 1
        if list_index == len(Arcs) - 1:
            next_index = list_index
        last_index = list_index - 1
        if list_index == 0:
            last_index = 0

        # Check if the arc should be removed based on its neighbors
        if Arcs[next_index] == Neighbor[1] or Arcs[last_index] == Neighbor[0]:
            
            # Remove the arc from the list
            del(Arcs[list_index])

            # Create new triples for left and right neighbors
            seganeh1 = []
            seganeh2 = []
            
            # Update edges for the circle event
            edegs_for_circles(p[0], Arcs[list_index-1],
                              Arcs[list_index], c, Edges, d)

            # If not at the start, create left neighbor triple
            if list_index > 1:
                seganeh1 = [Arcs[list_index-2],
                            Arcs[list_index-1], Arcs[list_index]]

            # If not at the end, create right neighbor triple
            if list_index < len(Arcs)-2:
                seganeh2 = [Arcs[list_index-1],
                            Arcs[list_index], Arcs[list_index+1]]

            # Return the new neighbor triples
            seganeh = [seganeh1, seganeh2]
            return seganeh

        else:
            # Increment index to find the next matching arc
            i_index += 1

def ObtainCircle(Triple, SiteEvent, IndexTripleN, hc= None): 
    '''
    Finds the circle event for a given triple of arcs in the Voronoi diagram.
    
    Args:
        Triple (list): List of three arcs involved in the circle event.
        SiteEvent (list): Site event containing the point of interest.
        IndexTripleN (int): Index to specify whether to use the first or second intersection (0 or 1).
        hc (list, optional): Additional arc for handling circle events if applicable.
        
    Returns:
        tuple: Coordinates of the circle event [x, y] or empty if no valid event.
    '''

    # Unpack arcs from the triple
    arc1, arc2, arc3 = Triple[0] , Triple[1] , Triple[2]

    # Adjust site event point slightly
    pp = list(SiteEvent)
    if hc is None:
        pp[1] = pp[1] - 0.1
    
    # Find intersection points between arcs
    [x12, y12] = IntersectionOf_ARC(arc1, arc2, pp)
    x12 = x12[IndexTripleN]
    y12 = y12[IndexTripleN]

    # the intersect point in Arc beetwen Arc2 and Arc3
    [x23, y23] = IntersectionOf_ARC(arc2, arc3, pp)
    x23 = x23[IndexTripleN]
    y23 = y23[IndexTripleN]

    xc = []
    if hc is not None and hc != [] and IndexTripleN == 0:
        [xc, yc] = IntersectionOf_ARC(arc3, hc, pp)
        if xc[0] > x23:
            xc = xc[0]
        else:
            xc = xc[1]

    elif hc is not None and hc != [] and IndexTripleN == 1:
        [xc, yc] = IntersectionOf_ARC(arc1, hc, pp)
        if xc[1] < x12:
            xc = xc[1]
        else:
            xc = xc[0]

    # Calculate the slopes of the lines passing through the intersections
    # the slope of line equation that pass of Point1 & Point2
    a12 = np.polyfit([arc1[0], arc2[0]], [arc1[1], arc2[1]], 1)[0]
    a23 = np.polyfit([arc2[0], arc3[0]], [arc2[1], arc3[1]], 1)[0]
    a12 , a23 = -1/a12 , -1/a23

    # Calculate y-intercepts of the lines
    b12 = y12 - a12 * x12 
    b23 = y23 - a23 * x23 
    
    # Find the intersection of the perpendicular bisectors
    if (a12 - a23) != 0:
        # the intersect point of pass line of breakpoints
        x = (b23 - b12)/(a12 - a23)
        y = a12 * x + b12

    # Determine if the circle event is valid based on the index
    if IndexTripleN == 0 and hc is None and x < arc3[0]:
        xy = [round(x, 2), round(y, 2)]
    elif IndexTripleN == 1 and hc is None and x > arc1[0]:
        xy = [round(x, 2), round(y, 2)]
    elif IndexTripleN == 0 and hc is not None and (xc == [] or x < xc):
        xy = [round(x, 2), round(y, 2)]
    elif IndexTripleN == 1 and hc is not None and (xc == [] or x > xc):
        xy = [round(x, 2), round(y, 2)]
    else:
        xy = []
    return xy, y

def find_subtree(Status, ParentDeleteNode): 
    '''
    Finds the subtree related to the parent node that is being deleted from the beachline status tree.
    
    Args:
        Status (TreeStatus): The current beachline status tree.
        ParentDeleteNode (TreeStatus): The parent node of the node being deleted.
        
    Returns:
        str: The direction of the subtree ('l' for left or 'r' for right).
    '''
    t_root = Status.root
    while True:
        if t_root.left == ParentDeleteNode:
            sh = 'l'
            break
        elif t_root.right == ParentDeleteNode:
            sh = 'r'
            break
        else:
            ParentDeleteNode = ParentDeleteNode.TreeParent
    return sh

def EdgesSiteEvent(POI, Arc, Edges):
    '''Update the DCEL (Doubly Connected Edge List) for edges with a new site event.
    
    Parameters:
    POI: The new site event.
    Arc: The arc that intersects with the new site event.
    Edges: The list of edges used to create the Voronoi diagram.
    
    Output:
    Updates the Edges list with a new edge, including the line equation and endpoints.
    '''
    
    # Adjust POI slightly to ensure non-zero denominator
    pp = list(POI) 
    pp[1] = pp[1] - 0.1
    [xi, yi] = IntersectionOf_ARC(POI, Arc, pp)

    e = [[POI, Arc], [xi, yi], [], 1]
    Edges.append(e)

def edegs_for_circles(POI, Arc_1, Arc_2, Vertex, Edges, PSweepLine):
    '''Update the DCEL for edges involving two arcs and a new vertex.
    
    Parameters:
    POI: The arc to be deleted.
    Arc_1: The previous arc.
    Arc_2: The next arc.
    Vertex: The vertex of the Voronoi diagram.
    Edges: The list of edges used to create the Voronoi diagram.
    PSweepLine: The position of the sweep line.
    
    Output:
    Updates the Edges list with a new edge, including the line equation and endpoints.
    '''

    a12 = [POI, Arc_1]
    a21 = [Arc_1, POI]
    a13 = [POI, Arc_2]
    a31 = [Arc_2, POI]

    for i in Edges:
        if i[0] == a12 or i[0] == a21:
            i[2].append(Vertex)
        if i[0] == a13 or i[0] == a31:
            i[2].append(Vertex)

    [xi, yi] = IntersectionOf_ARC(Arc_1, Arc_2, PSweepLine)
    e = [[Arc_1, Arc_2], [xi, yi], [Vertex], 2]
    Edges.append(e)

*** def BoundBox_Edge(Edge, EquationLine_BB, Edge_Of_BB):
    '''Find the intersection points between an edge and the bounding box.
    
    Parameters:
    Edge: An edge of the Voronoi diagram.
    EquationLine_BB: The line equations of the bounding box.
    Edge_Of_BB: The edges of the bounding box.
    
    Output:
    Returns the intersection points of the edge with the bounding box.
    '''

    dist1 = math.hypot(Edge[1][0][0]-Edge[2][0][0], Edge[1]
                       [1][0]-Edge[2][0][1]) 
    dist2 = math.hypot(Edge[1][0][1]-Edge[2][0][0], Edge[1]
                       [1][1]-Edge[2][0][1]) 

    if dist2 > dist1:
        sp = [Edge[1][0][1], Edge[1][1][1]]
    else:
        sp = [Edge[1][0][0], Edge[1][1][0]]

    le = np.polyfit(Edge[1][0], Edge[1][1], 1) 
    a = le[0]
    b = le[1]
    a1 = EquationLine_BB[0][0]
    b1 = EquationLine_BB[0][1]
    a2 = EquationLine_BB[1][0]
    b2 = EquationLine_BB[1][1]
    a3 = EquationLine_BB[2][0]
    b3 = EquationLine_BB[2][1]
    a4 = EquationLine_BB[3][0]
    b4 = EquationLine_BB[3][1]

    xy = []
    x = Edge_Of_BB[0][0][0]
    y = a*x + b

    if round(x, 2) == Edge_Of_BB[0][0][0] and y >= Edge_Of_BB[0][0][1] and y <= Edge_Of_BB[0][1][1]:
        xy.append([x, y])

    x = (b2-b)/(a-a2)
    y = a*x + b

    if round(y, 2) == Edge_Of_BB[1][0][1] and x >= Edge_Of_BB[1][0][0] and x <= Edge_Of_BB[1][1][0]:
        xy.append([x, y])

    x = Edge_Of_BB[2][0][0]
    y = a*x + b

    if round(x, 2) == Edge_Of_BB[2][0][0] and y <= Edge_Of_BB[2][0][1] and y >= Edge_Of_BB[2][1][1]:
        xy.append([x, y])

    x = (b4-b)/(a-a4)
    y = a*x + b

    if round(y, 2) == Edge_Of_BB[3][0][1] and x <= Edge_Of_BB[3][0][0] and x >= Edge_Of_BB[3][1][0]:
        xy.append([x, y])

    dists = []
    for i_xy in xy:

        # the distansces
        dist = math.hypot(i_xy[0]-sp[0], i_xy[1]-sp[1])
        dists.append(dist)

    if Edge[3] == 1:
        xy = xy[dists.index(min(dists))]
    else:

        # the equation of line
        le2 = np.polyfit(Edge[0][0], Edge[0][1], 1)
        aa = le2[0]
        bb = le2[1]

        x = (bb-b)/(a-aa)
        y = a*x + b

        # the distansces
        dist1 = math.hypot(x-xy[0][0], y-xy[0][1])
        dist2 = math.hypot(x-xy[1][0], y-xy[1][1])
        if dist1 > dist2:
            xy = xy[1]
        else:
            xy = xy[0]

    return xy

def HandleSiteEvent(p,t,th,q,list_of_arcs,ed): 
    '''Handle a site event in the Voronoi diagram sweep line algorithm.
    
    Parameters:
    p: The site event.
    t: The BeachLine tree.
    th: The height difference of the tree.
    q: The event queue.
    list_of_arcs: The list of arcs in the diagram.
    ed: The list of edges used to create the Voronoi diagram.
    
    Output:
    Updates the tree, event queue, and DCEL accordingly.
    '''

    # Initialize the tree if it's empty
    if t.root == None:
        # the definition leaf of tree
        t.root = TreeStatus([p,None])
        list_of_arcs.append(p)
    
    # Handle the case when the tree is not empty
    else:

        # the Arc of above point
        [arc,parent,parent_dir,st] = search_in_searchTree(t,p)
        
        EdgesSiteEvent(p,arc[0],ed)

        # Remove related circle events
        if arc[1] != None:
            while True:
                try:
                    circle = [arc[1],'circle']
                    qq = [i[0:2] for i in q]
                    del(q[qq.index(circle)])
                except:
                    break
        
        # the updateed Right subtree
        if parent_dir == 'r':
            parent.right = TreeStatus([arc[0],p])
            parent.right.left = arc
            
            # Update circle events
            node1 = parent.right.left
            parent_node1 = parent.right
            
            # the direction of move to node1 of parent
            parent_node1_dir = 'l'

            parent_node1.TreeParent = parent
            parent_node1.TreeParent_dir = 'r'

            parent.right.right = TreeStatus([p, arc[0]])
            parent.right.right.left = [p,None]

            parent.right.right.right = arc
            
            # this is for updated in circle event
            node2 = parent.right.right.right
            parent_node2 = parent.right.right
            
            # this is for move to Node2 of Parent in tree
            parent_node2_dir = 'r'

            # definition of Parent and direction of Parent in tree
            parent_node2.TreeParent = parent.right

            # definition of Parent and direction of Parent in tree
            parent_node2.TreeParent_dir = 'r'
            nodes = [node1,parent_node1,parent_node1_dir,node2,parent_node2,parent_node2_dir]


            if st[0] == 'r':

                # definition of tree Height difference  + tree Balanced
                if th == [0,1]: #taein ertefaeh derakht va motavaze kradan dar sorate lozaom
                    if st[1] == 'l':
                        [parent_node1,parent_node1_dir,parent_node2,parent_node2_dir] = RotateOfTree(t.root.right,'r',t.root,'r',nodes)
                    [parent_node1,parent_node1_dir,parent_node2,parent_node2_dir] = RotateOfTree(t,'l',None,None,nodes)
                elif th == [1,0]:
                    th[0] = 0
                    th[1] = 1
                elif th == [0,0]:
                    if st[1] == 'l':
                        [parent_node1,parent_node1_dir,parent_node2,parent_node2_dir] = RotateOfTree(t.root.right,'r',t.root,'r',nodes)
                    [parent_node1,parent_node1_dir,parent_node2,parent_node2_dir] = RotateOfTree(t,'l',None,None,nodes)
            elif st[0] == 'l':
                
                # definition of Height difference  + tree Balanced
                if th == [0,1]: 
                    th[0] = 1
                    th[1] = 0
                elif th == [1,0]:
                    if st[1] == 'r':
                        [parent_node1,parent_node1_dir,parent_node2,parent_node2_dir] = RotateOfTree(t.root.left,'l',t.root,'l',nodes)
                    [parent_node1,parent_node1_dir,parent_node2,parent_node2_dir] = RotateOfTree(t,'r',None,None,nodes)
                elif th == [0,0]:
                    if st[1] == 'r':
                        [parent_node1,parent_node1_dir,parent_node2,parent_node2_dir] = RotateOfTree(t.root.left,'l',t.root,'l',nodes)
                        th[1] = 1
                    [parent_node1,parent_node1_dir,parent_node2,parent_node2_dir] = RotateOfTree(t,'r',None,None,nodes)

            # find the latest Arc
            neighber = parent.left
            while type(neighber) is not list:
                neighber = neighber.right
            
            neighber = [neighber[0],0]
        
        # updated in Left subtree
        elif parent_dir == 'l': 
            parent.left = TreeStatus([arc[0], p])
            parent.left.left = arc
            
            # for updated in Circle Event
            node1 = parent.left.left
            
            # parent
            parent_node1 = parent.left 
            
            # the direction of move of Parent to Node1
            parent_node1_dir = 'l'
            parent_node1.TreeParent = parent
            parent_node1.TreeParent_dir = 'l'

            parent.left.right = TreeStatus([p, arc[0]])
            parent.left.right.left = [p,None]
            parent.left.right.right = arc

            # for updated in Circle Event
            node2 = parent.left.right.right
            parent_node2 = parent.left.right 
            
            # the direction of move of Parent to Node2
            parent_node2_dir = 'r'

            # definition of Parent and direction of Parent in tree
            parent_node2.TreeParent = parent.left
            
            # the definition of Parent and directional of Parent in tree
            parent_node2.TreeParent_dir = 'r'
            
            nodes = [node1,parent_node1,parent_node1_dir,node2,parent_node2,parent_node2_dir]
            if st[0] == 'r':
                
                # the definition of Height difference and Balanced
                if th == [0,1]: 
                    if st[1] == 'l':
                        [parent_node1,parent_node1_dir,parent_node2,parent_node2_dir] = RotateOfTree(t.root.right,'r',t.root,'r',nodes)
                    [parent_node1,parent_node1_dir,parent_node2,parent_node2_dir] = RotateOfTree(t,'l',None,None,nodes)
                elif th == [1,0]:
                    th[0] = 0
                    th[1] = 1
                elif th == [0,0]:
                    if st[1] == 'l':
                        [parent_node1,parent_node1_dir,parent_node2,parent_node2_dir] = RotateOfTree(t.root.right,'r',t.root,'r',nodes)
                    [parent_node1,parent_node1_dir,parent_node2,parent_node2_dir] = RotateOfTree(t,'l',None,None,nodes)
            elif st[0] == 'l':
                
                # the definition for Height difference + Balanced tree
                if th == [0,1]: 
                    th[0] = 1
                    th[1] = 0
                    
                elif th == [1,0]:
                    if st[1] == 'r':
                        [parent_node1,parent_node1_dir,parent_node2,parent_node2_dir] = RotateOfTree(t.root.left,'l',t.root,'l',nodes)
                    [parent_node1,parent_node1_dir,parent_node2,parent_node2_dir] = RotateOfTree(t,'r',None,None,nodes)
                elif th == [0,0]:
                    if st[1] == 'r':
                        [parent_node1,parent_node1_dir,parent_node2,parent_node2_dir] = RotateOfTree(t.root.left,'l',t.root,'l',nodes)
                        th[1] = 1
                    [parent_node1,parent_node1_dir,parent_node2,parent_node2_dir] = RotateOfTree(t,'r',None,None,nodes)

            # the finded nect Arc
            neighber = parent.right 
            while type(neighber) is not list:
                neighber = neighber.left
            neighber = [0,neighber[0]]

        # the updated tree for the First phase
        else:
            t.root = TreeStatus([arc[0], p])
            t.root.left = arc
            
            # the updated in Circle Event
            node1 = t.root.left 
            
            parent_node1 = t.root 

            # the move directional of Parent to Node1
            parent_node1_dir = 'l'
            parent_node1.TreeParent = t
            parent_node1.TreeParent_dir = 'root'
            
            t.root.right = TreeStatus([p, arc[0]])
            t.root.right.left = [p, None]
            t.root.right.right = arc
            
            # for updated in Circle Event
            node2 = t.root.right.right 
            parent_node2 = t.root.right 

            # for move directional of Parent to Node2
            parent_node2_dir = 'r'

            # this is for definition Parent + move directional in tree
            parent_node2.TreeParent = t.root

            # this is for definition Parent + move directional in tree
            parent_node2.TreeParent_dir = 'r'
            
            # the definition of Height difference in tree + Balanced tree
            th[1] = 1 

            # the find next Arc and Latest Arc
            neighber = [0,0]

        # the find triple Neighbors
        seganeh = insertToList(list_of_arcs,neighber,arc,p) 
        circleEvent=[]
        if seganeh[0] != []:
            [circleEvent,y_circle] = ObtainCircle(seganeh[0],p,0) 
            
            # this is for when we have Circle Event
            if circleEvent != []: 

                # the pointer for Tree and Q
                node1[1] = circleEvent 
                circle = [circleEvent,'circle',[node1,parent_node1,parent_node1_dir,parent_node1.TreeParent,parent_node1.TreeParent_dir]] 
                
                # the coordinates of point
                x_arc = seganeh[0][0][0]
                y_arc = seganeh[0][0][1]

                # the coordinate of circle center
                x_circle = circleEvent[0]

                # the radius of circle
                dist = math.hypot(x_arc-x_circle,y_arc-y_circle)
                circle[2].append([x_circle,y_circle-dist])

                # the y coordinates of Q
                y = []
                for i_q in q:
                    if i_q[1] == 'site':
                        y.append(i_q[0][1])
                    else:
                        y.append(i_q[2][-1][1])

                y.reverse()
                ind = bi.bisect_left(y,y_circle-dist)
                ind = len(y) - ind

                # the add to Q
                q.insert(ind,circle)
        circleEvent2 =[]
        if seganeh[1] != []:

            # calculated the circle event in Exist for Right triple Neighbors 
            [circleEvent2,y_circle] = ObtainCircle(seganeh[1],p,1) 
            
            # this segment for when we have circle event
            if circleEvent2 != [] and circleEvent2 != circleEvent:
                
                # the pointer for Tree + Q
                node2[1] = circleEvent2 #pointer bayne t va Q
                circle = [circleEvent2,'circle',[node2,parent_node2,parent_node2_dir,parent_node2.TreeParent,parent_node2.TreeParent_dir]] 
                
                # the coordinate of point
                x_arc = seganeh[1][0][0] 
                y_arc = seganeh[1][0][1]
                
                # the corrdinates of circle center
                x_circle = circleEvent2[0] 
                
                # the radius of circle
                dist = math.hypot(x_arc-x_circle,y_arc-y_circle)
                circle[2].append([x_circle,y_circle-dist])

                # the Y coordinates of Q
                y = []
                for i_q in q:
                    if i_q[1] == 'site':
                        y.append(i_q[0][1])
                    else:
                        y.append(i_q[2][-1][1])
                y.reverse()

                # the below coordinates of circle
                ind = bi.bisect_left(y,y_circle-dist)
                ind = len(y) - ind

                # the add to Q
                q.insert(ind,circle) 

def handleCircleEvent(p, t, th, q, list_of_arcs, vertexs, ed):
    '''
    Handles circle events in the Fortune's algorithm, which involves updating the beachline,
    event queue, and the dual-complex edge list (DCEL). 

    Parameters:
        p: The current circle event, a tuple of (event, type, attributes)
        t: The beachline tree
        th: The height difference between the left and right subtrees of the tree
        q: The event queue
        list_of_arcs: List of arcs in the Voronoi diagram
        vertexs: List of vertices of the Voronoi diagram
        ed: List of edges in the Voronoi diagram

    Updates:
        - The beachline tree
        - The event queue
        - The DCEL for edges and vertices
    '''
    
    # Add the current vertex to the list of vertices
    vertexs.append(p[0])
    
    # Extract attributes of the parent arc
    parent = p[2][1]  
    parent_dir = p[2][2]
    Gparent = p[2][3]
    Gparent_dir = p[2][4] 
    pl = p[2][5]

    # Initialize placeholders for affected nodes
    node1 = []
    node2 = []
    i_delete = 0

    # Remove related circle events from the queue
    while True: 
        try:
            circle = [p[2][0][1], 'circle']
            qq = [i[0:2] for i in q]
            qq_index = [i for i, n in enumerate(qq) if n == circle][i_delete]
            if qq_index != 0:
                del(q[qq_index])
            i_delete += 1
        except:
            break

    # Handle the case where the grandparent's direction is 'right'
    if Gparent_dir == 'r':
        
        # this is for rotation
        parent_sh = Gparent.right
        sh = find_subtree(t, parent_sh) 

        if parent_dir == 'r':

            # the delete of Arc + related leaf + tree changed
            Gparent.right = parent.left 

            # this segment is for definition parent
            if type(Gparent.right) is not list:
                Gparent.right.TreeParent = Gparent
                Gparent.right.TreeParent_dir = 'r'
            
            # Find the neighbor arc to the right
            parent_neighber = Gparent 
            parent_neighber_dir = 'r'

            # for find the Neighbor arc
            neighber = parent.left 
            while type(neighber) is not list:
                parent_neighber = neighber
                parent_neighber_dir = 'r'
                neighber = neighber.right

            # Remove the neighbor circle event from the queue
            while True:
                try:
                    circle = [neighber[1], 'circle']
                    qq = [i[0:2] for i in q]
                    del(q[qq.index(circle)])
                except:
                    break
            
            # Update node1 and neighbor
            neighber[1] = None
            node1 = neighber
            neighber = [neighber[0], []]
            parent_neighbers = [parent_neighber, parent_neighber_dir, [], []]

        elif parent_dir == 'l':
            Gparent.right = parent.right

            # the definition Parent
            if type(Gparent.right) is not list:
                Gparent.right.TreeParent = Gparent
                Gparent.right.TreeParent_dir = 'r'

            # Find the neighbor arc to the left
            parent_neighber = Gparent
            parent_neighber_dir = 'r'
            neighber = parent.right
            
            # Remove the neighbor circle event from the queue
            while type(neighber) is not list:
                parent_neighber = neighber
                parent_neighber_dir = 'l'
                neighber = neighber.left

            # the delete of neighbor Circle event
            while True:
                try:
                    circle = [neighber[1], 'circle']
                    qq = [i[0:2] for i in q]
                    del(q[qq.index(circle)])
                except:
                    break

            # Update node2 and neighbor
            neighber[1] = None
            node2 = neighber
            neighber = [[], neighber[0]]
            parent_neighbers = [[], [], parent_neighber, parent_neighber_dir]

            # Find the left neighbor of the grandparent
            neighber_left = Gparent.left
            while type(neighber_left) is not list:
                parent_neighber = neighber_left
                parent_neighber_dir = 'r'
                neighber_left = neighber_left.right

            # Remove the neighbor circle event from the queue
            while True:
                try:
                    circle = [neighber_left[1], 'circle']
                    qq = [i[0:2] for i in q]
                    del(q[qq.index(circle)])
                except:
                    break

            # Update node1 and neighbor
            neighber_left[1] = None
            node1 = neighber_left
            neighber[0] = neighber_left[0]
            parent_neighbers[0] = parent_neighber
            parent_neighbers[1] = parent_neighber_dir

        # Update the breakpoint in the grandparent's subtree
        if type(Gparent.right) is list:

            # Breakpoint update
            Gparent.root = [Gparent.root[0],
                            Gparent.right[0]]
        else:
            Gparent.root = [Gparent.root[0],
                            Gparent.right.root[0]]

        if type(parent_neighbers[0]) is list:
            parent_neighbers[0] = Gparent
        elif type(parent_neighbers[2]) is list:
            parent_neighbers[2] = Gparent

    # Handle the case where the grandparent's direction is 'left'
    elif Gparent_dir == 'l': 
        
        # for rotation
        parent_sh = Gparent.left
        sh = find_subtree(t, parent_sh)

        if parent_dir == 'r':

            # the delete of arc + related leaf + chainged tree
            Gparent.left = parent.left
            if type(Gparent.left) is not list:
                Gparent.left.TreeParent = Gparent
                Gparent.left.TreeParent_dir = 'l'
            
            # Find the neighbor arc to the right
            parent_neighber = Gparent
            parent_neighber_dir = 'l'
            neighber = parent.left
            while type(neighber) is not list:
                parent_neighber = neighber
                parent_neighber_dir = 'r'
                neighber = neighber.right

            # Remove the neighbor circle event from the queue
            while True:
                try:
                    circle = [neighber[1], 'circle']
                    qq = [i[0:2] for i in q]
                    del(q[qq.index(circle)])
                except:
                    break

            # Update node1 and neighbor
            neighber[1] = None
            node1 = neighber
            neighber = [neighber[0], []]
            parent_neighbers = [parent_neighber, parent_neighber_dir, [], []]
            
            # Find the right neighbor of the grandparent
            neighber_right = Gparent.right
            while type(neighber_right) is not list:
                parent_neighber = neighber_right
                parent_neighber_dir = 'l'
                neighber_right = neighber_right.left

            # Remove the neighbor circle event from the queue
            while True: 
                try:
                    circle = [neighber_right[1], 'circle']
                    qq = [i[0:2] for i in q]
                    del(q[qq.index(circle)])
                except:
                    break

            # Update node2 and neighbor
            neighber_right[1] = None 

            node2 = neighber_right
            neighber[1] = neighber_right[0]
            parent_neighbers[2] = parent_neighber
            parent_neighbers[3] = parent_neighber_dir

        elif parent_dir == 'l':
            Gparent.left = parent.right
            if type(Gparent.left) is not list: 
                Gparent.left.TreeParent = Gparent
                Gparent.left.TreeParent_dir = 'l'

            # Find the neighbor arc to the left
            parent_neighber = Gparent 
            parent_neighber_dir = 'l'

            # finded the neighbor arc
            neighber = parent.right
            while type(neighber) is not list:
                parent_neighber = neighber
                parent_neighber_dir = 'l'
                neighber = neighber.left

            # Remove the neighbor circle event from the queue
            while True:
                try:
                    circle = [neighber[1], 'circle']
                    qq = [i[0:2] for i in q]
                    del(q[qq.index(circle)])
                except:
                    break

            # Update node2 and neighbor
            neighber[1] = None

            node2 = neighber
            neighber = [[], neighber[0]]
            parent_neighbers = [[], [], parent_neighber, parent_neighber_dir]

        # Update the breakpoint in the grandparent's subtree
        if type(Gparent.left) is list:

            # Breakpoint update
            Gparent.root = [Gparent.left[0],
                            Gparent.root[1]]
        else:
            Gparent.root = [Gparent.left.root[0],
                            Gparent.root[1]]

        if type(parent_neighbers[0]) is list:
            parent_neighbers[0] = Gparent
        elif type(parent_neighbers[2]) is list:
            parent_neighbers[2] = Gparent

    # for definition of latest neighbor
    if node1 == [] or node2 == []:
        ggparent = Gparent.TreeParent
        i_new = 1
        if Gparent.TreeParent_dir == 'r':

            new_neghbor = ggparent.left
            new_parent = ggparent
            new_parent_dir = 'l'
            while type(new_neghbor) is not list:
                if i_new > 1:
                    new_parent = new_neghbor
                    new_parent_dir = 'r'
                new_neghbor = new_neghbor.right
                i_new += 1
        elif Gparent.TreeParent_dir == 'l':

            new_neghbor = ggparent.right
            new_parent = ggparent
            new_parent_dir = 'r'
            while type(new_neghbor) is not list:
                if i_new > 1:
                    new_parent = new_neghbor
                    new_parent_dir = 'l'
                new_neghbor = new_neghbor.left
                i_new += 1

        # the delete neighbor circle event
        while True: 
            try:
                circle = [new_neghbor[1], 'circle']
                qq = [i[0:2] for i in q]
                del(q[qq.index(circle)])
            except:
                break

        # the delete neighbor circle event
        new_neghbor[1] = None

        if node1 == []:
            node1 = new_neghbor
            parent_neighbers[0] = new_parent
            parent_neighbers[1] = new_parent_dir
        if node2 == []:
            node2 = new_neghbor
            parent_neighbers[2] = new_parent
            parent_neighbers[3] = new_parent_dir
    
    #this is for Rotate
    # move to Right sebtree 
    if sh == 'r':

        # Height difference
        if th == [0, 1]:
            th[1] = 0
        
        # Height difference
        elif th == [1, 0]:
            # this is for ReBalancing
            RotateOfTree(t, 'r')
            rot_dir = 'r'

            # Height difference
            th[0] = 0
        
        # Height difference
        elif th == [0, 0]:
            th[0] = 1

    # move to Left sebtree 
    elif sh == 'l': 
        
        # Height difference
        if th == [0, 1]:

            # this is for ReBalancing 
            RotateOfTree(t, 'l')
            rot_dir = 'l'
            th[1] = 0 

        # Height difference
        elif th == [1, 0]:
            th[0] = 0

        # Height difference
        elif th == [0, 0]:
            th[1] = 1

    # delet kardan sahmi az list va tashkile seganeha
    # the delete of Arc in treiple neighbors list
    seganeh = DeleteFromList(list_of_arcs, neighber, p, ed)

    # the middle arc
    arc_middle = list_of_arcs[int(len(list_of_arcs)/2)-1]
    if arc_middle != t.root.root[0]:
        try:
            if rot_dir == 'r':
                RotateOfTree(t, 'l')
                RotateOfTree(t.root.left, 'l', t.root, 'l', [
                            node1, parent_neighbers[0], parent_neighbers[1], node2, parent_neighbers[2], parent_neighbers[3]])
                RotateOfTree(t, 'r')
            elif rot_dir == 'l':
                RotateOfTree(t, 'r')
                RotateOfTree(t.root.right, 'r', t.root, 'r', [
                            node1, parent_neighbers[0], parent_neighbers[1], node2, parent_neighbers[2], parent_neighbers[3]])
                RotateOfTree(t, 'l')
        except:
            if th == [0, 1]:
                RotateOfTree(t.root.right, 'r', t.root, 'r', [
                            node1, parent_neighbers[0], parent_neighbers[1], node2, parent_neighbers[2], parent_neighbers[3]])
                RotateOfTree(t, 'l')
                th[1] = 0
            elif th == [1, 0]:
                RotateOfTree(t.root.left, 'l', t.root, 'l', [
                            node1, parent_neighbers[0], parent_neighbers[1], node2, parent_neighbers[2], parent_neighbers[3]])
                RotateOfTree(t, 'r')
                th[0] = 0

    circleEvent = []

    # the left triple 
    if seganeh[0] != []: 
        if seganeh[1] == []:
            # calculate circle event for exist for Left triple neighbor
            [circleEvent, y_circle] = ObtainCircle(seganeh[0], pl, 0, [])
        else:
            # calculate circle event for exist for Left triple neighbor
            [circleEvent, y_circle] = ObtainCircle(
                seganeh[0], pl, 0, seganeh[1][2])

        # when we have circle event
        if circleEvent != [] and circleEvent != p[0]:

            # the pointer for Tree + Q
            node1[1] = circleEvent
            cpar = parent_neighbers[0]

            # the directional parent
            cpar_dir = parent_neighbers[1]

            circle = [circleEvent, 'circle', [node1, cpar,
                                              cpar_dir, cpar.TreeParent, cpar.TreeParent_dir]]
            x_arc = seganeh[0][0][0] 
            y_arc = seganeh[0][0][1]

            # the  circle center coordiante
            x_circle = circleEvent[0]

            # the radius circle
            dist = math.hypot(x_arc-x_circle, y_arc-y_circle)
            if y_circle-dist < p[2][-1][1]:
                circle[2].append([x_circle, y_circle-dist])

                # the Y coordiantes of Q
                y = []
                for i_q in q:
                    if i_q[1] == 'site':
                        y.append(i_q[0][1])
                    else:
                        y.append(i_q[2][-1][1])

                y.reverse()
                ind = bi.bisect_left(y, y_circle-dist)
                ind = len(y) - ind

                # the added ro Q
                q.insert(ind, circle)
    circleEvent2 = []

    # the left triple
    if seganeh[1] != []:
        if seganeh[0] == []:
            # calculate circle event for exist for Right triple neighbor
            [circleEvent2, y_circle] = ObtainCircle(seganeh[1], pl, 1, [])
        else:
            # calculate circle event for exist for Right triple neighbor
            [circleEvent2, y_circle] = ObtainCircle(
                seganeh[1], pl, 1, seganeh[0][0])

        # if we have circle event
        if circleEvent2 != [] and circleEvent2 != p[0] and circleEvent2 != circleEvent:

            # this is pointer T + Q
            node2[1] = circleEvent2

            # the related parent
            cpar = parent_neighbers[2]

            # the related parent
            cpar_dir = parent_neighbers[3]

            circle = [circleEvent2, 'circle', [node2, cpar,
                                               cpar_dir, cpar.TreeParent, cpar.TreeParent_dir]]
            
            # the coordinate of point
            x_arc = seganeh[1][0][0]
            y_arc = seganeh[1][0][1]

            # the center coordinate of circle 
            x_circle = circleEvent2[0]

            # the radius od circle
            dist = math.hypot(x_arc-x_circle, y_arc-y_circle)
            if y_circle-dist < p[2][-1][1]:
                circle[2].append([x_circle, y_circle-dist])

                # the Y coordinates of Q
                y = []
                for i_q in q:
                    if i_q[1] == 'site':
                        y.append(i_q[0][1])
                    else:
                        y.append(i_q[2][-1][1])
                y.reverse()
                ind = bi.bisect_left(y, y_circle-dist)
                ind = len(y) - ind

                # added to Q
                q.insert(ind, circle)

    try:
        iii = list_of_arcs.index(p[2][0][0])
    
    except:
        i_delete = 0
        while True:
            try:
                q_i = [i[2][0][0] for i in q]
                qq_index = [i for i, n in enumerate(
                    q_i) if n == p[2][0][0]][i_delete]
                if qq_index != 0:
                    del(q[qq_index])
                i_delete += 1
            except:
                break

# ***************************** Finished Functions for Voronoi Diagram ******************************
# ****************************************************************************************************

# Sort points by X coordinates
point.sort()

# Sort points by Y coordinates in descending order
point.sort(key=lambda x: x[1], reverse=True) 

# Initialize the event queue with site events
q = []
for i_point in point:

    # creation of Q and certain that the Point is Site event or Circle event
    q.append([i_point, 'site'])
    print(" Event Q:" + str(q))

# Initialize the beachline tree
t = TreeStatus()  # Tree structure to maintain the beachline

# Initialize the heights of the left and right subtrees
t_height = [0, 0]  # Heights of the left and right subtrees

# Initialize lists for arcs, vertices, and edges
list_of_arcs = []
i_cir = 0
vertexs = []
edges = []

# Process events until the event queue is empty
while len(q) > 0:

    # Process site events
    if q[0][1] == 'site':  # agar noghte morede baresi az noeh site event bood
        HandleSiteEvent(q[0][0], t, t_height, q, list_of_arcs, edges)

    # Process circle events
    else:  # agar noghte morede baresi az noeh circle event bood
        handleCircleEvent(q[0], t, t_height, q, list_of_arcs, vertexs, edges)
        i_cir += 1

    # Remove the processed event from the queue
    del(q[0])

# Output the list of vertices
print('vertexs = ', vertexs)

# Add vertices to the original points list for plotting
for i_v in vertexs:
    point.append(i_v)

# Plot the vertices
for i in vertexs:
    x = i[0]
    y = i[1]
    plt.plot(x, y, 'ro')

# Define the bounding box
min_x = min([i[0] for i in point])
min_y = min([i[1] for i in point])
max_x = max([i[0] for i in point])
max_y = max([i[1] for i in point])

# Define the bounding box coordinates with a margin
bb = [[min_x - 3, min_y - 3], [min_x - 3, max_y + 3],
      [max_x + 3, max_y + 3], [max_x + 3, min_y-3]]

# Compute the equations of the bounding box edges
eq1 = np.polyfit([bb[0][0], bb[1][0]], [bb[0][1], bb[1][1]],
eq2 = np.polyfit([bb[1][0], bb[2][0]], [bb[1][1], bb[2][1]],
                 1)
eq3 = np.polyfit([bb[2][0], bb[3][0]], [bb[2][1], bb[3][1]],
                 1)
eq4 = np.polyfit([bb[3][0], bb[0][0]], [bb[3][1], bb[0][1]],
                 1)

# Store the equations of the bounding box edges      
bb_edge_eq = [[eq1[0], eq1[1]], [eq2[0], eq2[1]], [
    eq3[0], eq3[1]], [eq4[0], eq4[1]]]

bb_edge = [[bb[0], bb[1]], [bb[1], bb[2]], [bb[2], bb[3]], [bb[3], bb[0]]]

# Initialize list for plotting
plot = []

# Add edges to the plot list, considering bounding box edges
for i_ed in edges:
    if len(i_ed[2]) > 1:
        plot.append(i_ed[2])
    else:
        ne = BoundBox_Edge(i_ed, bb_edge_eq, bb_edge)
        i_ed[2].append(ne)
        plot.append(i_ed[2])

# Append bounding box edges to the plot
plot += (bb_edge)

# Plot all edges
for i in plot:
    x = [ii[0] for ii in i]
    y = [ii[1] for ii in i]
    plt.plot(x, y, 'g-')
    plt.pause(0.5)

# Show the final Voronoi diagram
plt.show()