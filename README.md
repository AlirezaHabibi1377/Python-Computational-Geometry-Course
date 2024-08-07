Hello, friends,

I have provided all of my Python projects from the Computational Geometry course of my Master's degree below. I hope you find them helpful for your research and projects.

# 1) Convex Hull Visualization

This project provides a visualization tool for computing and displaying the convex hull of a set of 2D points. The convex hull is the smallest convex polygon that can enclose all the given points.

## Requirements

- Python 3.x
- matplotlib
- numpy
- tkinter

## Usage

1. Install the necessary Python libraries:

    ```bash
    pip install matplotlib numpy
    ```

2. Run the script:

    ```bash
    python convex_hull.py
    ```

3. Select the input text file containing the coordinates of the points when prompted. The file should contain points in the following format:

    ```
    x1 y1
    x2 y2
    ...
    xn yn
    ```

4. The script will display the points and the convex hull in a plot.

## Output

When you run the script and provide the input file, the output will be a plot displaying the points and the convex hull. Below is an example of the output from the input file:

![Convex Hull Example](Convex%20Hull%20Visualization/ConvexHull_Figure.png)

## Conclusion

This script provides a simple yet effective way to visualize the convex hull of a set of 2D points. It demonstrates basic concepts of computational geometry and can be extended or modified for more advanced applications.


# 2) SweepLine Intersection

## Introduction

This project implements a plane sweep algorithm to find and visualize intersections among a set of lines in a 2D plane. The algorithm processes a series of line segments to determine their intersection points and plots these points for visualization. It uses the sweep line technique, which involves sweeping a vertical line across the plane and maintaining a status list of active line segments.

## Features

- **Random Line Generation**: Generate random lines within specified bounds.
- **File Reading**: Read lines from a text file and add them to the processing list.
- **Intersection Calculation**: Determine intersection points between lines.
- **Event Handling**: Manage events (line endpoints and intersections) during the sweep.
- **Visualization**: Plot all intersection points on a 2D graph using Matplotlib.

## Dependencies

- `matplotlib` for plotting the intersection points.
- `numpy` for numerical operations and line fitting.
- `bisect` for efficient sorting and searching.
- `tkinter` for file dialog and GUI elements.

## Usage

1. **Import Modules**: Import the necessary Python modules and functions from `Functions_Plane_Sweep_Intersect.py`.

   ```python
   import matplotlib.pyplot as plt
   import numpy as np  
   import bisect as bi 
   import tkinter as tk
   from tkinter import filedialog
   from Functions_Plane_Sweep_Intersect import *
   ```

## Output

After running the algorithm, the following outputs are produced:

1. **Intersection Points**: A list of all unique intersection points, sorted by their y-coordinate. Each point is represented by its x and y coordinates.
2. **Intersecting Lines**: A list of line pairs that intersect with each other. The format is `[line1, line2]`, indicating that line `line1` intersects with line `line2`.
3. **Visualization**: A plot showing all intersection points on a 2D plane. Each point is marked to visually confirm the intersections.

![Intersection Plot](SweepLine%20IntersectionConvex/Intersect_plot.png)

