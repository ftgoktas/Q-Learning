# We're going through 10 by 10 based maze.
# Assumption: We have one start and one end.

from __future__ import absolute_import
import math
from typing import List, Any, Tuple

from maze import Maze
from maze_viz import Visualizer
import random
import numpy as np


def RandomSearch(maze):
    row_curr, col_curr = maze.entry_coor  # Where to start searching

    maze.grid[row_curr][col_curr].visited = True
    # Set initial cell to visited

    visited_cells = list()
    # Stack of visited cells for backtracking

    path = list()
    # To track path of solution and backtracking cells

    while (row_curr, col_curr) != maze.exit_coor:
        # While the exit cell has not been encountered

        neighbour_indices = maze.find_neighbours(row_curr, col_curr)
        # Find neighbour indices

        neighbour_indices = maze.validate_neighbours_solve(neighbour_indices, row_curr, col_curr)
        if neighbour_indices is not None:
            # If there are unvisited neighbour cells

            visited_cells.append((row_curr, col_curr))
            # Add current cell to stack

            path.append(((row_curr, col_curr), False))
            # Add coordinates to part of search path
            # false indicates not a back tracker visit

            row_next, col_next = random.choice(neighbour_indices)
            # Choose random neighbour

            maze.grid[row_next][col_next].visited = True
            # Move to that neighbour

            row_curr = row_next
            col_curr = col_next

        elif len(visited_cells) > 0:
            # If there are no unvisited neighbour cells

            path.append(((row_curr, col_curr), True))
            # Add coordinates to part of search path
            # true indicates it is a back tracker visit

            row_curr, col_curr = visited_cells.pop()
            # Pop previous visited cell (backtracking)

    path.append(((row_curr, col_curr), False))  # Append final location to path

    return path

#general a unique number for each cell
def getCell(r,c,NUM_COLS):
    return r*NUM_COLS+c

#creates R
def createR(maze):
    # create matrix x*y
    R = np.matrix(np.ones(shape=(maze.num_rows * maze.num_cols, maze.num_rows * maze.num_cols)))
    R *= -1

    # assign zeros to paths
    for r in range(maze.num_rows):
        for c in range(maze.num_cols):
            neighbour_indices = maze.find_neighbours(r, c)
            # Find neighbour indices
            neighbour_indices = maze.validate_neighbours_solve(neighbour_indices, r, c)
            for n in neighbour_indices:
                R[getCell(r,c,maze.num_cols), getCell(n[0],n[1],maze.num_cols)] = 0

    # assign 1 to goal point
    r = maze.exit_coor[0]
    c = maze.exit_coor[1]
    neighbour_indices = maze.find_neighbours(r, c)
    # Find neighbour indices
    neighbour_indices = maze.validate_neighbours_solve(neighbour_indices, r, c)
    for n in neighbour_indices:
        R[getCell(n[0], n[1], maze.num_cols), getCell(r, c, maze.num_cols)] = 1

    print("R=",R)
    return R

#update Q function
def update(current_state, next_state, gamma, Q, R, num_cols):
    next = getCell(next_state[0],next_state[1],num_cols)

    max_index = np.where(Q[next,] == np.max(Q[next,]))[1]

    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size=1))
    else:
        max_index = int(max_index)

    max_value = Q[next, max_index]

    curr = getCell(current_state[0],current_state[1],num_cols)

    Q[curr, next] = \
        R[curr, next] + gamma * max_value
    #print('max_value', R[curr, next] + gamma * max_value)

#Get max Q function
def GetMaxQOption(row_curr, col_curr, neighbour_indices, Q, num_cols):
    max = -math.inf
    current_state=row_curr * num_cols + col_curr
    for n in neighbour_indices:
        next = n[0] * num_cols + n[1]
        if (Q[current_state,next] >max):
            max = Q[current_state,next]
            r,c = n[0],n[1]
    return r, c


# Generate Maze
numRows = 10
numCols = 10
theMaze = Maze(numRows, numCols, id=0)

print("Random Path")
####Random Path
# generate a solution by exploring a random path
theMaze.solution_path = RandomSearch(theMaze)

# Show what points were explored to find the solution
vis = Visualizer(theMaze, cell_size=1, media_filename="")
vis.animate_maze_solution()

# Show Solution Path
vis = Visualizer(theMaze, cell_size=1, media_filename="")
vis.show_maze_solution()

############################
print("Q Learning")
####Q Learning
# generate a solution using Q Learning
theMaze.ClearSolution();
theMaze.solution_path = QLearning(theMaze)

# Show what points were explored to find the solution
#vis = Visualizer(theMaze, cell_size=1, media_filename="")
#vis.animate_maze_solution()

# Show Solution Path
vis = Visualizer(theMaze, cell_size=1, media_filename="")
vis.show_maze_solution()
