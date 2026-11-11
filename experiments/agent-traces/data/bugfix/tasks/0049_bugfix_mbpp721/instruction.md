You are a software engineer working in a sandbox. Use your tools to inspect and edit the code.

The file /home/user/input/solution.py defines `maxAverageOfPath`. It should do this:

Given a square matrix of size N*N given as a list of lists, where each cell is associated with a specific cost. A path is defined as a specific sequence of cells that starts from the top-left cell move only right or down and ends on bottom right cell. We want to find a path with the maximum average over all existing paths. Average is computed as total cost divided by the number of cells visited in the path.

Example: `maxAverageOfPath([[1, 2, 3], [6, 5, 4], [7, 3, 9]]) == 5.2` must hold.

The function has a bug and some calls return the wrong result. Find the bug and fix it by editing /home/user/input/solution.py in place. Keep the function's name and signature. When the fix is done, stop.
