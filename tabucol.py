from collections import deque
from random import randrange
# import cnfgen
# from pysat.solvers import Solver
import numpy as np
from multiprocessing.pool import Pool
import os
import networkx as nx

import glob
import random
import numpy as np
from scipy.sparse import load_npz
import pandas as pd
class GraphDataset(object):

    def __init__(self,folder_path,ordered=False):
        super().__init__()

        self.file_paths=glob.glob(f'{folder_path}/*.npz')
        self.file_paths.sort()
        self.ordered=ordered

        if self.ordered:
            self.i = 0

    def __len__(self):
        return len(self.file_paths)
    
    def get(self):
        if self.ordered:
            file_path = self.file_paths[self.i]
            self.i = (self.i + 1)%len(self.file_paths)
        else:
            file_path = random.sample(self.file_paths, k=1)[0]
        return load_npz(file_path).toarray()

def tabucol(graph, number_of_colors=3,attempts=50, tabu_size=7, reps=100, max_iterations=10000, debug=False):

    for _ in range(attempts):
        colors = list(range(number_of_colors))
        # number of iterations of the tabucol algorithm
        iterations = 0
        # initialize tabu as empty queue
        tabu = deque()
        
        # solution is a map of nodes to colors
        # Generate a random solution:
        solution = dict()
        # for i in range(len(graph)):
        for i in graph.nodes():
            solution[i] = colors[randrange(0, len(colors))]

        # Aspiration level A(z), represented by a mapping: f(s) -> best f(s') seen so far
        aspiration_level = dict()

        while iterations < max_iterations:
            # Count node pairs (i,j) which are adjacent and have the same color.
            move_candidates = set()  # use a set to avoid duplicates
            conflict_count = 0

            for u,v in graph.edges():
                if solution[u]==solution[v]:
                    move_candidates.add(u)
                    move_candidates.add(v)
                    conflict_count += 1

            move_candidates = list(move_candidates)  # convert to list for array indexing

            if conflict_count == 0:
                # Found a valid coloring.
                break

            # Generate neighbor solutions.
            new_solution = None
            for r in range(reps):
                # Choose a node to move.
                node = move_candidates[randrange(0, len(move_candidates))]
                
                # Choose color other than current.
                new_color = colors[randrange(0, len(colors) - 1)]
                if solution[node] == new_color:
                    # essentially swapping last color with current color for this calculation
                    new_color = colors[-1]

                # Create a neighbor solution
                new_solution = solution.copy()
                new_solution[node] = new_color
                # Count adjacent pairs with the same color in the new solution.
                new_conflicts = 0
                for u,v in graph.edges():
                    if new_solution[u]==new_solution[v]:
                        new_conflicts += 1

                if new_conflicts < conflict_count:  # found an improved solution
                    # if f(s') <= A(f(s)) [where A(z) defaults to z - 1]
                    if new_conflicts <= aspiration_level.setdefault(conflict_count, conflict_count - 1):
                        # set A(f(s) = f(s') - 1
                        aspiration_level[conflict_count] = new_conflicts - 1

                        if (node, new_color) in tabu: # permit tabu move if it is better any prior
                            tabu.remove((node, new_color))
                            if debug:
                                print("tabu permitted;", conflict_count, "->", new_conflicts)
                            break
                    else:
                        if (node, new_color) in tabu:
                            # tabu move isn't good enough
                            continue
                    if debug:
                        print (conflict_count, "->", new_conflicts)
                    break

            # At this point, either found a better solution,
            # or ran out of reps, using the last solution generated
            
            # The current node color will become tabu.
            # add to the end of the tabu queue
            tabu.append((node, solution[node]))
            if len(tabu) > tabu_size:  # queue full
                tabu.popleft()  # remove the oldest move

            # Move to next iteration of tabucol with new solution
            solution = new_solution
            iterations += 1
            if debug and iterations % 500 == 0:
                print("iteration:", iterations)

        if conflict_count==0:
            # print(solution)
            return True, solution,iterations
    return False ,None, None



from argparse import ArgumentParser
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--distribution", type=str,default='3col_50vertices', help="Distribution of dataset")
    parser.add_argument("--num_color", type=int,default=3, help="The number of color")
    args = parser.parse_args()
    dataset=GraphDataset(f'../data_color/testing/{args.distribution}',ordered=True)
    arguments=[]

    for _ in range(len(dataset)):
        G=dataset.get()
        G=nx.from_numpy_array(G)
        arguments.append((G,))
    with Pool(100) as pool:
        results=pool.starmap(tabucol, arguments)

    df={'Solved':[],'Solution':[],'Step':[]}
    for solved,solution,step in results:
        df['Solved'].append(solved)
        df['Solution'].append(solution)
        df['Step'].append(step)
    
    df=pd.DataFrame(df)
    print('Solved instances:',df['Solved'].mean())
    save_path=f'data/{args.distribution}_tabucol'
    os.makedirs(save_path,exist_ok=True)
    df.to_pickle(os.path.join(save_path,'results'))

    
