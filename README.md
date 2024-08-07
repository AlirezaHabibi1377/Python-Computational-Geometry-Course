Hello, friends,

I have provided all of my Python projects from the Computational Geometry course of my Master's degree below. I hope you find them helpful for your research and projects.

# <span style="color:blue;">1) Convex Hull Visualization</span>

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
    python ConvexHull_Incremental.py
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

# <span style="color:blue;">2) SweepLine Intersection</span>

The **Plane Sweep Intersection Algorithm** is a computational geometry technique designed to efficiently find all intersections among a set of line segments in a 2D plane. This algorithm simulates a vertical line, called the sweep line, which moves from left to right across the plane. As the sweep line progresses, it processes various events, such as the start or end of line segments and intersections between segments.

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

![Intersection Plot](SweepLine%20Intersection/Intersect_plot.png)

## Conclusion

The plane sweep algorithm effectively identifies and visualizes intersections among line segments in a 2D plane. By handling events such as line endpoints and intersection points, the algorithm efficiently determines all intersecting lines and provides a clear graphical representation. The sorted output of intersection points and the list of intersecting lines offer valuable insights into the geometric relationships between the input line segments.

This approach can be extended to more complex geometrical problems and adapted for various applications in computational geometry.

# <span style="color:blue;">3) Monotone Triangulation</span>

## Introduction

This project implements a monotone triangulation algorithm for polygons. The algorithm performs triangulation on a polygon provided by the user and visualizes both the polygon and the resulting diagonals. Monotone triangulation is used in computational geometry to divide a polygon into triangles, facilitating various applications such as rendering and collision detection.

## Usage

1. **Input Data**: Prepare a text file with polygon vertex coordinates. Each line should contain the x and y coordinates of a vertex, separated by a space.
2. **Run the Script**: Execute the script in an environment where Python and the required libraries (`matplotlib`, `shapely`, `tkinter`, `copy`) are installed.
3. **Select File**: A file dialog will appear prompting you to select the text file containing the polygon data.
4. **View Results**: The script will plot the polygon and its diagonals. A window with the plot will appear showing the triangulation.

## Output

After running the algorithm, the following outputs are produced:

1. **Polygon Plot**: The input polygon is plotted, showing the vertices and the edges connecting them.
2. **Diagonals Plot**: The diagonals added during triangulation are plotted, demonstrating how the polygon is divided into triangles.
3. **Visualization**: The final plot displays both the polygon and its diagonals, helping to visually confirm the triangulation results.
![Intersection Plot](Monotone%20Triangulation/Triangulation_Monoton_Polygon.png)

```bash
python TriangulateMonoton.py
```

## Conclusion

This script provides a practical implementation of monotone triangulation, useful for dividing polygons into simpler triangular regions. The algorithm visualizes the results, which helps in understanding how the polygon is triangulated. By using this script, users can efficiently visualize and verify triangulations for various polygon shapes.

# <span style="color:blue;">4) Voronoi Diagram</span>

This project implements a Voronoi Diagram generator using Python. The Voronoi Diagram is a fundamental geometric structure used in various fields such as computational geometry, geographic information systems, and more. This implementation uses a sweep line algorithm to compute the diagram and visualize the results using Matplotlib.

## Introduction

A Voronoi Diagram partitions a plane into regions based on the distance to a specified set of points. Each region corresponds to one of the points, and any location within a region is closer to its corresponding point than to any other. This visualization is particularly useful for problems involving proximity and spatial relationships.

This code performs the following steps to generate and visualize a Voronoi Diagram:
1. **Input Points**: Points are provided, which will serve as the sites for the Voronoi diagram.
2. **Event Queue Initialization**: Points are sorted and initialized as site events in the event queue.
3. **Beachline Tree**: A tree structure maintains the current status of the beachline, which is updated as events are processed.
4. **Event Processing**: The code processes both site events and circle events from the queue.
5. **Plotting**: The resulting vertices and edges of the Voronoi diagram are plotted.

## Requirements

To run this code, you'll need:
- Python 3.x
- Matplotlib
- NumPy
- Tkinter (for file dialogs)

You can install the required packages using pip:

```bash
pip install matplotlib numpy
```

## Usage

1. **Prepare Your Points**: Replace the sample points (InputPoints1.txt & InputPoints2.txt) in the script with your own set of points.
2. **Run the Script**: Execute the Python script from the command line:

```bash
python VoronoiDiagram.py
```
3. **View Results**: The Voronoi diagram will be displayed in a new window showing the diagram with vertices and edges based on the input points.

## Output

The output of the script includes:

- Vertices: Red points representing the vertices of the Voronoi cells.
- Edges: Green lines representing the edges of the Voronoi cells.
- Bounding Box: The plot includes a bounding box around the Voronoi diagram to ensure all regions are visible.

**Output of 'InputPoints1.txt':**

![Intersection Plot](Voronoi%20Diagram/Figure_1.png)

This image shows the Voronoi diagram generated from the points specified in 'InputPoints1.txt'. The diagram includes the vertices and edges of the Voronoi cells, with a bounding box to ensure visibility of all regions.

**Output of 'InputPoints2.txt':**

![Intersection Plot](Voronoi%20Diagram/Figure_2.png)

This image depicts the Voronoi diagram for the points in 'InputPoints2.txt'. Similar to the previous output, it displays the computed Voronoi cells with vertices and edges, highlighting the spatial partitioning based on the input points.

## Conclusion

The Voronoi Diagram Generator script provides a straightforward method for visualizing spatial relationships between a set of points using a Voronoi diagram. By following the instructions in this README, you can easily input your own set of points, run the script, and visualize the resulting Voronoi diagram. This tool is valuable for understanding proximity-based partitioning and can be adapted for various applications in computational geometry and data analysis. If you have any questions or need further modifications, feel free to reach out or contribute to the project.

## Contact

For questions or further information, please contact:

Email: mo.alireza77habibi@gmail.com
