import math
import numpy as np
import random
import networkx as nx
import pandas as pd

import glob
import numpy as np
from scipy.sparse import load_npz
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


class Greedy(object):
  def __init__(self, Ma,num_color):
    self.Ma = Ma
    self.n = Ma.shape[0]
    self.solution = np.zeros( self.n ) - 1  #no color assigned to any vertex
    self.num_color=num_color
    
  def execute(self):
    self.solution[0] = 0 #first color to first vertex
    
    all_colors = set( range(self.n) )
    # all_colors=set(range(3))
    unavailable = set()
    for i in range(1, self.n):
      for j in range( 0, self.n):
        if self.Ma[i,j] == 1:
          if self.solution[j] != -1:
            unavailable.add( self.solution[j] )
          #end if
        #end if
      #end for
      available = all_colors - unavailable
      self.solution[i] = np.amin( np.array( list( available )))
      unavailable.clear()
    #end for
      
    conflict_count=0
    for i in range(self.n):
      for j in range( i + 1, self.n):
        if self.Ma[i,j] == 1 and self.solution[i] == self.solution[j]:
          conflict_count+=1
        #   return -1
          

    # print(self.solution)
    # return conflict_count
    return np.amax(self.solution)+1<=self.num_color
    # return np.amax( self.solution) + 1
  
def mod_greedy(graph,number_of_colors:int,random_solution:bool,irrevesible:bool):

  if isinstance(graph,np.ndarray):
     graph=nx.from_numpy_array(graph)
   
   
  solution = {}
  for i in range(graph.number_of_nodes()):
    if random_solution:
      solution[i] = random.randint(1, number_of_colors)
    else:
      solution[i] = 1
    
  conflict_count = sum(solution[u] == solution[v] for u, v in graph.edges())
  # print("Initial conflict count:", conflict_count)
  # print("Number of edges in the graph",graph.number_of_edges())

  max_iterations = 1000
  if irrevesible:
    actions_hist=set([])

  for _ in range(max_iterations):
    best_delta = float('-inf')
    best_color = None
    best_node = None
    
    for node in range(graph.number_of_nodes()):
      if irrevesible and node in actions_hist:
        continue
      delta_1 = sum(solution[neighbor] == solution[node] for neighbor in graph.neighbors(node))
      for color in range(1, number_of_colors + 1):
        if color != solution[node]:
            delta_2 = sum(color == solution[neighbor] for neighbor in graph.neighbors(node))

            total_delta = delta_1 - delta_2
            if total_delta > best_delta:
                best_delta = total_delta
                best_color = color
                best_node = node

      # print(best_delta)
    if best_delta > 0:
        solution[best_node] = best_color
        if irrevesible:
          actions_hist.add(best_node)
    else:
        break
      
  # print(solution)
  conflict_count = sum(solution[u] == solution[v] for u, v in graph.edges())

  # print("Final conflict count:", conflict_count)
  return conflict_count
           
if __name__ == "__main__":

  dataset=GraphDataset('../data_color/testing/3col_100vertices',ordered=True)

  df={'S2V-Simplified':[],'Simplified LS-DQN':[]}
  greedy_solved=0
  ls_solved=0

  for _ in range(len(dataset)):
    G=dataset.get()
    # greedy = Greedy(Ma=G,num_color=3)
    print('Simplified S2V-DQN')
    _temp1=mod_greedy(G,number_of_colors=3,random_solution=False,irrevesible=True)
    if _temp1==0:
       greedy_solved+=1
    df['S2V-Simplified'].append(_temp1)
    print('Simplified LS-DQN')
    _temp2=float('inf')
    for _ in range(50):
      _temp2=min(mod_greedy(G,number_of_colors=3,random_solution=True,irrevesible=False),_temp2)
    # break
    if _temp2==0:
       ls_solved+=1
    df['Simplified LS-DQN'].append(_temp2)
    # break
    # print( greedy.execute() ) 
            
  df=pd.DataFrame(df)

  print(df)
  print('Greedy solved:',greedy_solved)
  print('LS solved:',ls_solved)
  print('Local Search mean',df['Simplified LS-DQN'].mean())     
      