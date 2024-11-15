

<div align="center">
  
# UCS2504 - Foundations of Artificial Intelligence
  
### NAME: PRANAV MOORTHI
### REG: 312222 5001 093
### CLASS: SEC-B

</div>



## Assignment 1 - DFS
**Date :** 01/08/2024

**Problem Description :** 

1. Representing Search Problems :

A search problem consists of
- a start node
- a neighbors function that, given a node, returns an enumeration of the edges from the
node
- a specification of a goal in terms of a Boolean function that takes a node and returns true
if the node is a goal
- a (optional) heuristic function that, given a node, returns a non-negative real number. The
heuristic function defaults to zero.

As far as the searcher is concerned a node can be anything. In the simple examples, the node is a string. Define an abstract class Search problem with methods start node(), is goal(),
neighbors() and heuristic().

The neighbors is a list of edges. A (directed) edge consists of two nodes, a from node and a
to node. The edge is the pair (from node,to node), but can also contain a non-negative cost
(which defaults to 1) and can be labeled with an action. Implement a class Edge. Define a suitable repr () method to print the edge.

2. Explicit Representation of Search Graph :

The first representation of a search problem is from an explicit graph (as opposed to one that is
generated as needed). An explicit graph consists of
- a set of nodes
- a list of edges
- a start node
- a set of goal nodes
- (optionally) a dictionary that maps a node to a heuristic value for that node

To define a search problem, we need to define the start node, the goal predicate, the neighbors function and the heuristic function. Define a concrete class
Search problem from explicit graph(Search problem).

Give a title string also to the search problem. Define a suitable repr () method to print the
graph.

3. Paths :

A searcher will return a path from the start node to a goal node. Represent the path in terms of a recursive data structure that can share subparts. A path is either:
- a node (representing a path of length 0) or
- an initial path and an edge, where the from node of the edge is the node at the end of
initial.

Implement a class Path(). Define a suitable repr () method to print the path.

4. Example Search Problems :

Using Search problem from explicit graph, represent the following graphs.

For example, the first graph can be created with the code
from searchProblem import Edge, Search_problem_from_explicit_graph,
Search_problem
problem1 = Search_problem_from_explicit_graph(’Problem 1’,
{’A’,’B’,’C’,’D’,’G’},
[Edge(’A’,’B’,3), Edge(’A’,’C’,1), Edge(’B’,’D’,1), Edge(’B’,’G’,3),
Edge(’C’,’B’,1), Edge(’C’,’D’,3), Edge(’D’,’G’,1)],
start = ’A’,
goals = {’G’})

5. Searcher :

A Searcher for a problem is given can be asked repeatedly for the next path. To solve a problem, you can construct a Searcher object for the problem and then repeatedly ask for the next path using search. If there are no more paths, None is returned. Implement Searcher class with DFS (Depth-First Search).

To use depth-first search to find multiple paths for problem1, copy and paste the following
into Python’s read-evaluate-print loop; keep finding next solutions until there are no more:

Depth-first search for problem1; do the following:
searcher1 = Searcher(searchExample.problem1)
searcher1.search() # find first solution
searcher1.search() # find next solution (repeat until no solutions)

**Algorithm :**
```bash
Input: problem
Output: solution, or failure

frontier ← [initial state of problem]
explored = {}
while frontier is not empty do
  node ← remove a node from frontier
  if node is a goal state then
    return solution
  end
  add node to explored
  add the successor nodes to frontier only if not in frontier or explored
end
return failure
```

**Code :** 
```Python
class Search_Problem:
    def start_node(self):
        pass
    def is_goal(self):
        pass
    def neighbours(self):
        pass
    def heuristic(self):
        pass

class Edge:
    def __init__(self,from_node,to_node,cost):
        self.from_node = from_node
        self.to_node = to_node
        self.cost = cost

    def __repr__(self):
        return str(self.from_node) + "->" + str(self.to_node) + " cost: " + str(self.cost)
    

class Search_Problem_Explicit_Graph(Search_Problem):
    def __init__(self,title,nodes,edges,start,goals = [], hmap = {}):
        self.title = title
        self.nodes = nodes
        self.edges = edges
        self.start = start
        self.goals = goals

        self.neighbours = {}
        for i in nodes:
            self.neighbours[i] = []

        for i in edges:
            self.neighbours[i.from_node].append(i)

    def start_node(self):
        return self.start
        
    def is_goal(self,node):
        if node in self.goals:
            return True
        return False
    def neighbours(self,node):
        return self.neighbours[node]
    

    def heuristic(self):
        return super().heuristic()
    
    def __repr__(self):
        edges_string = ",".join(str(edge) for edge in self.edges)

        return str(self.title) + edges_string
    

class Path:
    def __init__(self,node,parent_path = None,edge = None):
        self.node = node
        self.parent_path = parent_path
        self.edge = edge

        if edge is None:
            self.cost = 0

        else:
            self.cost = parent_path.cost + edge.cost

    def nodes(self):
        if self.parent_path:
            return self.parent_path.nodes() + (self.node,)

    def __repr__(self):
        if self.parent_path == None:
            return f"{self.node} cost({self.cost})"
        else:
            return f"{self.parent_path} --> {self.node} cost({self.cost})"
        
    
    
class Searcher:
    def __init__(self,problem):
        self.problem = problem
        self.frontier = [Path(problem.start_node())]
        self.explored = set()

    def search(self):
        while self.frontier:
            path = self.frontier.pop()
            node = path.node

            if self.problem.is_goal(node):
                return path
            
            self.explored.add(node)

            for i in self.problem.neighbours[node]:
                new_path = Path(i.to_node,path,i)
                self.frontier.append(new_path)

        return None
    

search = Searcher(Search_Problem_Explicit_Graph('DFS',['A','B','C','D','G'],[Edge('A','B',3),Edge('A','C',1),Edge('B','D',1),Edge('B','G',3),
                                                                             Edge('C','B',1),Edge('C','D',3),Edge('D','G',1)],start = 'A',goals = ['G']))

result = search.search()
while(result is not None):
    print(result)
    result = search.search()
```

**Testing :**
```bash
pranav@Pranav-L ailab % python3 dfs.py
A cost(0) --> C cost(1) --> D cost(4) --> G cost(5)
A cost(0) --> C cost(1) --> B cost(2) --> G cost(5)
A cost(0) --> C cost(1) --> B cost(2) --> D cost(3) --> G cost(4)
A cost(0) --> B cost(3) --> G cost(6)
A cost(0) --> B cost(3) --> D cost(4) --> G cost(5)
```

## Assignment 2 - Uniform Cost Search
**Date :** 08/08/2024

**Problem Description :** 

1. Representing Search Problems :

A search problem consists of
- a start node
- a neighbors function that, given a node, returns an enumeration of the edges from the
node
- a specification of a goal in terms of a Boolean function that takes a node and returns true
if the node is a goal
- a (optional) heuristic function that, given a node, returns a non-negative real number. The
heuristic function defaults to zero.

As far as the searcher is concerned a node can be anything. In the simple examples, the node is a string. Define an abstract class Search problem with methods start node(), is goal(),
neighbors() and heuristic().

The neighbors is a list of edges. A (directed) edge consists of two nodes, a from node and a
to node. The edge is the pair (from node,to node), but can also contain a non-negative cost
(which defaults to 1) and can be labeled with an action. Implement a class Edge. Define a suitable repr () method to print the edge.

2. Explicit Representation of Search Graph :

The first representation of a search problem is from an explicit graph (as opposed to one that is
generated as needed). An explicit graph consists of
- a set of nodes
- a list of edges
- a start node
- a set of goal nodes
- (optionally) a dictionary that maps a node to a heuristic value for that node

To define a search problem, we need to define the start node, the goal predicate, the neighbors function and the heuristic function. Define a concrete class
Search problem from explicit graph(Search problem).

Give a title string also to the search problem. Define a suitable repr () method to print the
graph.
3. Paths :

A searcher will return a path from the start node to a goal node. Represent the path in terms of a recursive data structure that can share subparts. A path is either:
- a node (representing a path of length 0) or
- an initial path and an edge, where the from node of the edge is the node at the end of
initial.

Implement a class Path(). Define a suitable repr () method to print the path.

4. Example Search Problems :

Using Search problem from explicit graph, represent the following graphs.

For example, the first graph can be created with the code
from searchProblem import Edge, Search_problem_from_explicit_graph,
Search_problem
problem1 = Search_problem_from_explicit_graph(’Problem 1’,
{’A’,’B’,’C’,’D’,’G’},
[Edge(’A’,’B’,3), Edge(’A’,’C’,1), Edge(’B’,’D’,1), Edge(’B’,’G’,3),
Edge(’C’,’B’,1), Edge(’C’,’D’,3), Edge(’D’,’G’,1)],
start = ’A’,
goals = {’G’})

5. Frontier as a Priority Queue
In many of the search algorithms, such as Uniform Cost Search, A* and other best-first searchers, the frontier is implemented as a priority queue. Use Python’s built-in priority queue implementations heapq (read the Python documentation, https://docs.python. org/3/library/heapq.html).
Implement FrontierPQ. A frontier is a list of triples. The first element of each triple is the value to be minimized. The second element is a unique index which specifies the order that the elements were added to the queue, and the third element is the path that is on the queue. The use of the unique index ensures that the priority queue implementation does not compare paths; whether one path is less than another is not defined. It also lets us control what sort of search (e.g., depth-first or breadth-first) occurs when the value to be minimized does not give a unique next path. Use a variable frontier index to maintain the total number of elements of the frontier that have been created.


6. Searcher
A Searcher for a problem can be asked repeatedly for the next path. To solve a problem, you can construct a Searcher object for the problem and then repeatedly ask for the next path using search. If there are no more paths, None is returned. Implement Searcher class using using the FrontierPQ class.

**Algorithm :**
```bash
Input: problem
Output: solution, or failure

frontier ← Priority Queue
Add starting node to frontier
explored ← Set
while frontier is not empty do
  path ← remove the frontier node with shortest distance
  v ← path.node
  if v is a goal node then return solution
    if v is not in explored
      for each successor w of v do
  	    new_path ← path + v
  	    new_cost ← path.cost + heuristic(u)
  	    Add new_path to Frontier
return failure
```
**Code :** 
```python
class SearchProblem:
    def start_node(self):
        pass
    def is_goal(self):
        pass
    def neighbours(self):
        pass
    def heuristic(self):
        pass

class Edge:
    def __init__(self,start,end,cost):
        self.start = start
        self.end = end
        self.cost = cost

    def __repr__(self):
        return f"{self.start} -> {self.end}"
    
class Search_Problem_Explicit_Graph(SearchProblem):
    def __init__(self,title,nodes,edges,start,goals = [],hmap = {}):
        self.title = title
        self.nodes = nodes
        self.edges = edges
        self.start = start
        self.goals = goals
        self.hmap = hmap

        self.neighbours = {k:[x for x in edges if x.start == k] for k in nodes}

    def start_node(self):
        return self.start
    
    def is_goal(self,node):
        if node in self.goals:
            return True
        return False
    
    def neighbours(self,node):
        return self.neighbours[node]
    
    def heuristic(self,node):
        return self.hmap[node]
    
class Path:
    def __init__(self,node,parent_path = None,edge = None):
        self.node = node
        self.parent_path = parent_path
        self.edge = edge

        if self.parent_path:
            self.cost = parent_path.cost + edge.cost
        else:
            self.cost = 0
    def distance(self):
        return self.cost
    def __repr__(self):
        if self.parent_path:
            return f"{self.parent_path}-->{self.node}"
        return f"{self.node}"
    
class Searcher:
    def __init__(self,problem):
        self.problem = problem
        self.frontier = [(0,Path(self.problem.start_node()))]
        self.explored = set()

    def heappq(self):
        index = self.findmin()
        return self.frontier.pop(index)
    
    def findmin(self):
        min_index = 0
        min_value = self.frontier[0][0]
        for i in range(len(self.frontier)):
            if self.frontier[i][0] < min_value:
                min_value = self.frontier[i][0]
                min_index = i

        return min_index

    def search(self):
        while self.frontier:
            path = self.heappq()[1]
            node = path.node
            if self.problem.is_goal(path.node):
                return path
            
            self.explored.add(path.node)
            for edge in self.problem.neighbours[node]:
                if edge.end not in self.explored:
                    new_path = Path(edge.end,path,edge)
                    self.frontier.append((new_path.cost,new_path))

        return None
    
problem = Search_Problem_Explicit_Graph("UCS",
       {'A','B','C','D','G'},
    [Edge('A','B',3), Edge('A','C',1), Edge('B','D',1), Edge('B','G',3),
         Edge('C','B',1), Edge('C','D',3), Edge('D','G',1)],
start = 'A',
goals = {'G'}
)

uniformCostSearcher = Searcher(problem)
solution = uniformCostSearcher.search()
print("Path to Goal Found:")
print(solution)
```
**Testing :**
```bash
pranav@Pranav-L ailab % python3 ucs.py
Path to Goal Found:
A-->C-->B-->D-->G
```


## Assignment 3 - A* Algorithm
**Date :** 12/08/2024

**Problem Description 1:** 

In a 3×3 board, 8 of the squares are filled with integers 1 to 9, and one square is left empty. One move is sliding into the empty square the integer in any one of its adjacent squares. The start state is given on the left side of the figure, and the goal state given on the right side. Find a sequence of moves to go from the start state to the goal state.

1) Formulate the problem as a state space search problem.
2) Find a suitable representation for the states and the nodes.
3) Solve the problem using any of the uninformed search strategies.
4) We can use Manhattan distance as a heuristic h(n). The cheapest cost from the current node to the goal node, can be estimated as how many moves will be required to transform the current node into the goal node. This is related to the distance each tile must travel to arrive at its destination, hence we sum the Manhattan distance of each square from its home position.
5) An alternative heuristic should consider the number of tiles that are “out-of-sequence”.
An out of sequence score can be computed as follows:
- a tile in the center counts 1,
- a tile not in the center counts 0 if it is followed by its proper successor as defined
by the goal arrangement,
- otherwise, a tile counts 2.
6) Use anyone of the two heuristics, and implement Greedy Best-First Search.
7) Use anyone of the two heuristics, and implement A* Search

**Algorithm 1:**
1. A* Algortihm
```bash
Input: problem
Output: solution, or failure

frontier ← Priority Queue
add starting node to frontier with priority = heuristic(start) + 0  

while frontier is not empty do
    path ← remove node from frontier with lowest priority
    node ← path.node
    add node to explored set
    
    for each neighbor of node do
        if neighbor not in explored set then
            new_path ← Path(neighbor, path, edge)
            if neighbor is a goal node then
                return new_path as solution
            
            frontier.add((heuristic(neighbor) + g(new_path), new_path))
            
return failure
```
2. Greedy-Best First Search
```bash
Input: problem
Output: solution, or failure

frontier ← Priority Queue
add starting node to frontier with priority = heuristic(start)

while frontier is not empty do
    path ← remove node from frontier with lowest priority
    node ← path.node
    add node to explored set
    
    for each neighbor of node do
        if neighbor not in explored set then
            new_path ← Path(neighbor, path, edge)
            if neighbor is a goal node then
                return new_path as solution
            
            frontier.add((heuristic(neighbor), new_path))
            
return failure
```
**Code 1:** 
1. A* Code
```python
import copy

class Node:
    def __init__(self,state):
        self.state = state
        return True     
    def find_blanks(self):
        for i in range(3):
            for j in range(3):
                if self.state[i][j] == None:
                    return i, j
                 
    def manhattan_distance(self):
        h = 0
        for i in range(3):
            for j in range(3):
                num = self.state[i][j]
                if num is None:
                    continue
                exp_row = (num - 1)//3
                exp_col = (num - 1)%3
                h += (abs(exp_row - i) + abs(exp_col - j))
        return h
    
    def __repr__(self):
        return ("\n").join(str(row) for row in self.state)
    
class Solution:
    def __init__(self,start):
        self.node = Node(start)
        self.frontier = [(self.node,[],self.node.manhattan_distance())]
        self.explored = set()
    def create_node(self,node,a,b,dir):
        if dir == 'UP':
            new_state = copy.deepcopy(node.state)
            new_state[a][b] , new_state[a-1][b] = new_state[a-1][b],new_state[a][b]
            return Node(new_state)
        
        if dir == 'DOWN':
            new_state = copy.deepcopy(node.state)
            new_state[a][b] , new_state[a+1][b] = new_state[a+1][b],new_state[a][b]
            return Node(new_state)
        if dir == 'LEFT':
            new_state = copy.deepcopy(node.state)
            new_state[a][b] , new_state[a][b -1] = new_state[a][b - 1],new_state[a][b]
            return Node(new_state)
        if dir == 'RIGHT':
            new_state = copy.deepcopy(node.state)
            new_state[a][b] , new_state[a][b + 1] = new_state[a][b + 1],new_state[a][b]
            return Node(new_state)
    def getmin(self):
        self.frontier.sort(key = lambda x : x[2])
        return self.frontier.pop(0)
    def solve(self):
        while self.frontier:
            node,path,dis = self.getmin()

            self.explored.add(node)
            new_states = []

            i,j = node.find_blanks()
            if i > 0:
                up_node = self.create_node(node,i,j,'UP')
                new_states.append(up_node)
            if i < 2:
                down_node = self.create_node(node,i,j,'DOWN')
                new_states.append(down_node)
            if j > 0:
                left_node = self.create_node(node,i,j,'LEFT')
                new_states.append(left_node)
            if j < 2:
                right_node = self.create_node(node,i,j,'RIGHT')
                new_states.append(right_node)
            
            for state in new_states:
                if state.manhattan_distance() == 0:
                    return state.state , path + [state.state]
                if state not in self.explored:
                    self.frontier.append((state,path + [state.state],state.manhattan_distance() + len(path)))

        return None

            
            
node = Node([[1,2,3],[None,4,6],[7,5,8]])

solve = Solution([[1,2,3],[None,4,6],[7,5,8]])


state,path = solve.solve()

for part in path:
    print(part)
    print("-->")

```
2. Greedy-Best First Search
```python
class Node:
    def __init__(self, state, parent=None):
        self.state = state.copy()
        self.parent = parent

    def find_blank(self):
        for i in range(3):
            for j in range(3):
                if self.state[i][j] is None:
                    return i, j

    def manhattan_heuristic(self):
        heuristic = 0
        for row in range(3):
            for col in range(3):
                num = self.state[row][col]
                if num is None:
                    continue
                exp_row = (num - 1) // 3
                exp_col = (num - 1) % 3

                heuristic += abs(exp_row - row) + abs(exp_col - col)
        return heuristic
    
    def moves(self):
        if self.parent is None:
            return 0
        return self.parent.moves() + 1

    def gbfsHeuristic(self):
        return self.manhattan_heuristic()

    def toString(self):
        string = ""
        for i in range(3):
            for j in range(3):
                if self.state[i][j] is None:
                    string += '  '
                else:
                    string += str(self.state[i][j]) + ' '
            string += '\n'
        return string

    def _repr_(self):
        return self.toString()

    def get_path(self):
        string = ""
        if self.parent:
            string += self.parent.get_path() + '\n'
        string += self.toString()
        return string

class Solution:
    def __init__(self, state):
        node = Node(state)
        self.frontier = [(node.gbfsHeuristic(), node)]
        self.explored = []

    def minPop(self):
        self.frontier.sort(key=lambda x: x[0])
        return self.frontier.pop(0)
    
    def create_node(self, parent, direction, i, j):
        new_state = [parent.state[0].copy(), parent.state[1].copy(), parent.state[2].copy()]
        if direction == 'UP':
            new_state[i][j], new_state[i - 1][j] = new_state[i - 1][j], None
        elif direction == 'DOWN':
            new_state[i][j], new_state[i + 1][j] = new_state[i + 1][j], None
        elif direction == 'RIGHT':
            new_state[i][j], new_state[i][j + 1] = new_state[i][j + 1], None
        else:
            new_state[i][j], new_state[i][j - 1] = new_state[i][j - 1], None
        return Node(new_state, parent)
    
    def solve(self):
        while self.frontier:
            # Pop the frontier
            node = self.minPop()[1]

            # Check if this node is already present in explored
            if node in self.explored:
                continue 
            
            # Otherwise, set as explored
            self.explored.append(node)
            
            # Add its children into the frontier
            i, j = node.find_blank()
            
            if i > 0:
                up_node = self.create_node(node, 'UP', i, j)
                if up_node.manhattan_heuristic() == 0:
                    return up_node
                self.frontier.append((up_node.gbfsHeuristic(), up_node))
            if i < 2:
                down_node = self.create_node(node, 'DOWN', i, j)
                if down_node.manhattan_heuristic() == 0:
                    return down_node
                self.frontier.append((down_node.gbfsHeuristic(), down_node))
            if j < 2:
                right_node = self.create_node(node, 'RIGHT', i, j)
                if right_node.manhattan_heuristic() == 0:
                    return right_node
                self.frontier.append((right_node.gbfsHeuristic(), right_node))
            if j > 0:
                left_node = self.create_node(node, 'LEFT', i, j)
                if left_node.manhattan_heuristic() == 0:
                    return left_node
                self.frontier.append((left_node.gbfsHeuristic(), left_node))
        
        return None
                        
if __name__ == "__main__":
    state = [[1, 2, 3], [None, 4, 6], [7, 5, 8]]
    solution = Solution(state)

    result = solution.solve()
    print(result.get_path())
```
**Testing 1:**
1. A* Output
```bash
pranav@Pranav-L ailab % python3 8puzzleGB.py
[1, 2, 3]
[4, None, 6]
[7, 5, 8]
-->
[1, 2, 3]
[4, 5, 6]
[7, None, 8]
-->
[1, 2, 3]
[4, 5, 6]
[7, 8, None]
-->
```
2. Greedy-Best First Search
```
1 2 3 
  4 6
7 5 8

1 2 3
4   6
7 5 8

1 2 3
4 5 6
7   8

1 2 3
4 5 6
7 8
```

**Problem Description 2:** 

You are given an 8-litre jar full of water and two empty jars of 5- and 3-litre capacity. You have to get exactly 4 litres of water in one of the jars. You can completely empty a jar into another jar with space or completely fill up a jar from another jar.

1. Formulate the problem: Identify states, actions, initial state, goal state(s). Represent the state by a 3-tuple. For example, the intial state state is (8,0,0). (4,1,3) is a goal state
(there may be other goal states also).
2. Use a suitable data structure to keep track of the parent of every state. Write a function to print the sequence of states and actions from the initial state to the goal state.
3. Write a function next states(s) that returns a list of successor states of a given state s.
4. Implement Breadth-First-Search algorithm to search the state space graph for a goal state that produces the required sequence of pourings. Use a Queue as frontier that stores the discovered states yet be explored. Use a dictionary for explored that is used to store the explored states.
5. Modify your program to trace the contents of the Queue in your algorithm. How many
states are explored by your algorithm?

**Algorithm 2:**
```bash
Input: problem
Output: solution, or failure

frontier ← Queue
add starting node to frontier
parent[start] ← None

while frontier is not empty do
    path ← remove node from frontier
    node ← path.node
    add node to explored set
    
    for each neighbor of node do
        if neighbor not in explored set then
            new_path ← Path(neighbor, path)
            if neighbor is a goal node then
                return new_path as solution
            
            frontier.append(new_path)
            
return failure
```

**Code 2:** 
```python
class Node:
    def __init__(self,start):
        self.state = start
    
    def __repr__(self):
        return f"{self.state}"
    
class Solution:
    def __init__(self,node):
        self.node = node
        self.frontier = [(self.node,[self.node])]
        self.explored = set()
    
    def create_state(self,node):
        new_states = []

        x,y,z = node.state
        if x > 0:
            if y < 5:
                transfer = min(x,5-y)
                new_states.append(Node([x - transfer,y + transfer, z]))
            if z < 3:
                transfer = min(x,3-z)
                new_states.append(Node([x - transfer,y , z + transfer]))

        if y > 0:
            if x < 8:
                transfer = min(y,8-x)
                new_states.append(Node([x + transfer,y - transfer, z]))
            if z < 3:
                transfer = min(y,3-z)
                new_states.append(Node([x ,y - transfer , z + transfer]))

        if z > 0:
            if y < 5:
                transfer = min(z,5-y)
                new_states.append(Node([x ,y + transfer, z - transfer]))
            if x < 8:
                transfer = min(z,8-x)
                new_states.append(Node([x + transfer,y , z - transfer]))
        return new_states
    
    def solve(self):
        while self.frontier:
            node,path = self.frontier.pop(0)

            self.explored.add(node)
            new_states = self.create_state(node)

            for state in new_states:
                if 4 in state.state:
                    return state,path + [state]
                if state not in self.explored:
                    self.frontier.append((state,path + [state]))
                
            
node = Node([8,0,0])
solve = Solution(node)

state, path = solve.solve()

for part in path:
    print(part)
```

**Testing 2:**
```bash
[8, 0, 0]
[3, 5, 0]
[3, 2, 3]
[6, 2, 0]
[6, 0, 2]
[1, 5, 2]
[1, 4, 3]
```

## Assignment 4
**Date :** 29/08/2024

**Problem Description :** 

Place 8 queens “safely” in a 8×8 chessboard – no queen is under attack from any other queen
(in horizontal, vertical and diagonal directions). Formulate it as a constraint satisfaction problem.
- One queen is placed in each column.
- Variables are the rows in which queens are placed in the columns
- Assignment: 8 row indexes.
- Evaluation function: the number of attacking pairs in 8-queens
Implement a local search algorithm to find one safe assignment.

**Algorithm 1 - Local Search:**
```bash
Input: problem
Output: solution, or failure

current ← initial state of problem
while true do
    neighbors ← generate neighbors of current
    best_neighbor ← find the best state in neighbors

    if best_neighbor is better than current then
        current ← best_neighbor
    else
        return current as solution
```

**Code 1:** 
```python
import random

def init(n):
    l = list(range(n))
    random.shuffle(l)
    return tuple(l)

def neighbors(state, n):
    lst = []
    for i in range(n):
        for j in range(i + 1, n):
            new = state[:i] + (state[j],) + state[i + 1:j] + (state[i],) + state[j + 1:]
            lst.append(new)
    return lst

def evaluate(state, n):
    c = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) == abs(state[i] - state[j]):
                c += 1
    return c

def print_board(state):
    n = len(state)
    board = [['.' for _ in range(n)] for _ in range(n)]
    for row, col in enumerate(state):
        board[row][col] = 'Q'
    for row in board:
        print(" ".join(row))
    print("\n")

def local_search(n):
    cur_state = init(n)
    cur_val = evaluate(cur_state, n)
    print("Initial board:")
    print_board(cur_state)
    while cur_val > 0:
        xx = neighbors(cur_state, n)
        for i in xx:
            x = evaluate(i, n)
            if x < cur_val:
                cur_val = x
                cur_state = i
                print(f"Current state with evaluation {cur_val}:")
                print_board(cur_state)
                break
        else:
            print("\nRandom Restart!\n")
            return local_search(n)
            break
    print("Solution found:")
    print_board(cur_state)
    return cur_state

n = int(input("No. of rows = "))
local_search(n)
```

**Testing 1:**
```bash
No. of rows = 4
Initial board:
. . . Q
Q . . .
. Q . .
. . Q .


Current state with evaluation 1:
Q . . .
. . . Q
. Q . .
. . Q .


Current state with evaluation 0:
. Q . .
. . . Q
Q . . .
. . Q .


Solution found:
. Q . .
. . . Q
Q . . .
. . Q .

```

**Algorithm 2 - Stochastic Search:**
```bash
Input: problem
Output: solution, or failure

current ← initial solution of problem
while stopping criteria not met do
    if current is a valid solution then
        return current as solution
    
    neighbor ← randomly select a neighbor of current
    neighbor_value ← evaluate(neighbor)

    if neighbor_value < evaluate(current) then
        current ← neighbor
    else
        if random() < acceptance_probability(current, neighbor_value) then
            current ← neighbor
return failure
```

**Code 1:** 
```python
import random

def no_attacking_pairs(board):
    """ Count the number of pairs of queens that are attacking each other. """
    n = len(board)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (board[i] == board[j] or
                abs(board[i] - board[j]) == abs(i - j)):
                count += 1
    return count

def possible_successors(conf):
    n = len(conf)
    state_value = {}

    for i in range(n):
        for j in range(n):
            if j != conf[i]: 
                x = conf[:i] + [j] + conf[i + 1:]
                ap = no_attacking_pairs(x)
                state_value[ap] = x

    min_conflicts = min(state_value.keys())
    return state_value[min_conflicts], min_conflicts

def print_board(board):
    """ Display the board with queens as 'Q' and empty spaces as '.' """
    n = len(board)
    for row in range(n):
        line = ""
        for col in range(n):
            if board[row] == col:
                line += "Q "
            else:
                line += ". "
        print(line)
    print("\n")

def random_restart(n):
    global iteration
    iteration += 1
    print(f"\nRandom Restart #{iteration}")
    l = [random.randint(0, n - 1) for _ in range(n)]
    print_board(l)
    return l

def eight_queens(initial):
    conflicts = no_attacking_pairs(initial)
    print("Initial configuration:")
    print_board(initial)
    
    while conflicts > 0:
        new, new_conflicts = possible_successors(initial)
        if new_conflicts < conflicts:
            conflicts = new_conflicts
            initial = new
            print("New configuration with fewer conflicts:")
            print_board(initial)
        else:
            initial = random_restart(len(initial))
            conflicts = no_attacking_pairs(initial)
    
    print("Solution found:")
    print_board(initial)
    return initial

iteration = 0
n = int(input('No. of rows = '))
board = random_restart(n)

solution = eight_queens(board)
print("Number of random restarts =", iteration)
print("Final configuration of the board =")
print_board(solution)
```

**Testing 1:**
```bash
No. of rows = 4

Random Restart #1
. . Q .
. . . Q
. . . Q
. Q . .


Initial configuration:
. . Q .
. . . Q
. . . Q
. Q . .


New configuration with fewer conflicts:
. . Q .
Q . . .
. . . Q
. Q . .


Solution found:
. . Q .
Q . . .
. . . Q
. Q . .


Number of random restarts = 1
Final configuration of the board =
. . Q .
Q . . .
. . . Q
. Q . .
```

## Assignment 5
**Date :** 05/09/2024

**Problem Description :** 

1. Class Variable
Define a class Variable consisting of a name and a domain. The domain of a variable is a list or a tuple, as the ordering will matter in the representation of constraints. We would like to create a Variable object, for example, as
X = Variable(’X’, {1,2,3})

2. Class Constraint
Define a class Constraint consisting of
- A tuple (or list) of variables called the scope.
- A condition, a Boolean function that takes the same number of arguments as there are
variables in the scope. The condition must have a name property that gives a printable
name of the function; built-in functions and functions that are defined using def have such
a property; for other functions you may need to define this property.
- An optional name
We would like to create a Variable object, for example, as Constraint([X,Y],lt) where lt is
a function that tests whether the first argument is less than the second one.
Add the following methods to the class.
def can_evaluate(self, assignment):
"""
assignment is a variable:value dictionary
returns True if the constraint can be evaluated given assignment
"""
def holds(self,assignment):
"""returns the value of Constraint evaluated in assignment.
precondition: all variables are assigned in assignment, ie self.can_evaluate(assignment) """

3. Class CSP
A constraint satisfaction problem (CSP) requires:
- variables: a list or set of variables
- constraints: a set or list of constraints.
Other properties are inferred from these:
- var to const is a mapping fromvariables to set of constraints, such that var to const[var]
is the set of constraints with var in the scope.
Add a method consistent(assignment) to class CSP that returns true if the assignment is consistent with each of the constraints in csp (i.e., all of the constraints that can be evaluated
evaluate to true).

We may create a CSP problem, for example, as

X = Variable(’X’, {1,2,3})
Y = Variable(’Y’, {1,2,3})
Z = Variable(’Z’, {1,2,3})
csp0 = CSP("csp0", {X,Y,Z},
[Constraint([X,Y],lt),
Constraint([Y,Z],lt)])

The CSP csp0 has variables X, Y and Z, each with domain {1, 2, 3}. The con straints are X < Y and Y < Z.

4. 8-Queens
Place 8 queens “safely” in a 8×8 chessboard – no queen is under attack from any other queen (in horizontal, vertical and diagonal directions). Formulate it as a constraint satisfaction problem.
- One queen is placed in each column.
- Variables are the rows in which queens are placed in the columns
- Assignment: 8 row indexes.
Represent it as a CSP.

5. Simple DFS Solver
Solve CSP using depth-first search through the space of partial assignments. This takes in a CSP problem and an optional variable ordering (a list of the variables in the CSP). It returns a generator of the solutions.

**Algorithm :**
```bash
Input: assignment, CSP (Constraint Satisfaction Problem)
Output: solution, or failure

function backtrack(assignment, csp):
    if length of assignment equals number of csp variables then
        return assignment

    unassigned ← variables in csp not in assignment
    var ← first variable in unassigned

    for each value in var.domain do
        new_assignment ← copy of assignment
        new_assignment[var] ← value

        if csp is consistent with new_assignment then
            result ← backtrack(new_assignment, csp)
            if result is not None then
                return result

    return None
```

**Code :** 
```python
import random
import itertools


class Variable(object):
    def __init__(self, name, domain, position=None):
        self.name = name
        self.domain = domain
        self.position = position if position else (random.random(), random.random())
        self.size = len(domain)

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Variable({self.name})"
    
    def __eq__(self, other):
        if isinstance(other, Variable):
            return self.name == other.name
        return False

    def __hash__(self):
        return hash(self.name)

    

class Constraint(object):
    def __init__(self, scope, condition, string=None, position=None):
        self.scope = scope
        self.condition = condition
        if string is None:
            self.string = f"{self.condition.__name__}({self.scope})"
        else:
            self.string = string
        self.position = position
    
    def __repr__(self):
        return self.string

    def can_evaluate(self, assignment):
        return all(v in assignment for v in self.scope)

    def holds(self, assignment):
        return self.condition(*tuple(assignment[v] for v in self.scope))

class CSP(object):
    def __init__(self, title, variables, constraints):
        self.title = title
        self.variables = variables
        self.constraints = constraints
        self.var_to_const = {var:set() for var in self.variables}

        for con in constraints:
            for var in con.scope:
                self.var_to_const[var].add(con)

    def __str__(self):
        return str(self.title)
    
    def __repr__(self):
        return f"CSP({self.title}, {self.variables}, {([str(c) for c in self.constraints])}"
    
    def consistent(self, assignment):
        return all(con.holds(assignment)
                   for con in self.constraints
                   if con.can_evaluate(assignment))
   

def n_queens_csp(n):
    variables = [Variable(f"Q{i}", list(range(n))) for i in range(n)]
    
    constraints = []
    
    for (var1, var2) in itertools.combinations(variables, 2):
        constraints.append(Constraint([var1, var2], lambda x, y: x != y, f"{var1} != {var2}"))
    
    for (var1, var2) in itertools.combinations(variables, 2):
        row1 = int(var1.name[1:])
        row2 = int(var2.name[1:])
        constraints.append(Constraint([var1, var2], lambda x, y, row_diff=abs(row1 - row2): abs(x - y) != row_diff,
                                      f"abs({var1} - {var2}) != {abs(row1 - row2)}"))
    
    csp = CSP(f"{n}-Queens Problem", variables, constraints)
    
    solution = backtrack({}, csp)
    
    return solution


def backtrack(assignment,csp):
    if len(assignment) == len(csp.variables):
        return assignment
    
    unassigned = [x for x in csp.variables if x not in assignment]
    var = unassigned[0]

    for value in var.domain:
        new_assignment = assignment.copy()
        new_assignment[var] = value
        
        if csp.consistent(new_assignment):
            result = backtrack(new_assignment, csp)
            if result is not None:
                return result
            
    return None


def print_n_queens_solution(n, solution):
    # Create an empty board
    board = [['.' for _ in range(n)] for _ in range(n)]
    
    # Place queens on the board
    for var, value in solution.items():
        row = int(var.name[1])  
        col = value  
        board[row][col] = 'Q'
    
    # Print the board
    for row in reversed(board):
        print(' '.join(row))
    print()  


solution = n_queens_csp(8)
print_n_queens_solution(8, solution)
```

 **Testing :**
```bash
pranav@Pranav-L ailab % python3 n-queens.py
. . . Q . . . .
. Q . . . . . .
. . . . . . Q .
. . Q . . . . .
. . . . . Q . .
. . . . . . . Q
. . . . Q . . .
Q . . . . . . .
```

## Assignment 6
**Date :** 05/09/2024

**Problem Description :** 

Consider two-player zero-sum games, where a player only wins when another player loses. This can be modeled with a single utility which one agent (the maximizing agent) is trying maximize and the other agent (the minimizing agent) is trying to minimize. Define a class Node to represent a node in a game tree.

class Node(Displayable):
"""A node in a search tree. It has a name a string isMax is True if it is a maximizing node, otherwise it is minimizing node children is the list of children value is what it evaluates to if it is a leaf.
"""
Create the game tree given below:
1. Implement minimax algorithm for a zero-sum two player game as a function minimax(node,
depth). Let minimax(node, depth) return both the score and the path. Test it on the
game tree you have created.
2. Modify the minimax function to include αβ-pruning.

**Algorithm Minimax:**
```bash
Input: node
Output: score, path

function minimax(node):
    if node is a leaf then
        return evaluate(node), None

    if node is a maximizing node then
        max_score ← -∞
        max_path ← None
        for each child in node.children() do
            score, path ← minimax(child)
            if score > max_score then
                max_score ← score
                max_path ← (child.name, path)

        return max_score, max_path

    else  
        min_score ← ∞
        min_path ← None
        for each child in node.children() do
            score, path ← minimax(child)
            if score < min_score then
                min_score ← score
                min_path ← (child.name, path)

        return min_score, min_path

```

**Code Minimax:** 
```python
class Node:
    def __init__(self,name,isMax,value = None,allchildren = None):
        self.name = name
        self.isMax = isMax
        self.value = value
        self.allchildren = allchildren
    

    def isLeaf(self):
        if self.allchildren is None:
            return True
        
    def children(self):
        return self.allchildren
    
    def evaluate(self):
        return self.value
    

def minimax(node):
    if node.isLeaf():
        return node.evaluate(),None
    
    elif node.isMax:
        max_score = float('-inf')
        max_path = None
        for c in node.children():
            score,path = minimax(c)
            if score > max_score:
                max_score = score
                max_path = c.name,path

        return max_score,max_path
    else:
        min_score = float('inf')
        min_path = None
        for c in node.children():
            score,path = minimax(c)
            if score < min_score:
                min_score = score
                min_path = c.name,path

        return min_score,min_path
    

leaf_a1 = Node("A1", isMax=False, value=1)
leaf_a2 = Node("A2", isMax=False, value=5)
leaf_a3 = Node("A3", isMax=False, value = 1)

leaf_b1 = Node("B1", isMax=False, value=2)
leaf_b2 = Node("B2", isMax=False, value=9)

# Intermediate nodes
child_a = Node("A", isMax=False, allchildren=[leaf_a1, leaf_a2])
child_b = Node("B", isMax=False, allchildren=[leaf_b1, leaf_b2])

# Root node
root = Node("Root", isMax=True, allchildren=[child_a, child_b])

val, path = minimax(root)

print(val)

print(path)
```

 **Testing Minimax:**
```bash
2
('B', ('B1', None))
```

**Algorithm Minimax - αβ-pruning:**
```bash
Input: node, alpha, beta
Output: score, path

function minimax(node, alpha, beta):
    if node is a leaf then
        return evaluate(node), None

    if node is a maximizing node then
        max_path ← None
        for each child in node.children() do
            score, path ← minimax(child, alpha, beta)
            if score >= beta then
                return score, None 
            if score > alpha then
                alpha ← score
                max_path ← (child.name, path)
        return alpha, max_path

    else  // node is a minimizing node
        min_path ← None
        for each child in node.children() do
            score, path ← minimax(child, alpha, beta)
            if score <= alpha then
                return score, None  
            if score < beta then
                beta ← score
                min_path ← (child.name, path)
        return beta, min_path

```

**Code Minimax - αβ-pruning:** 
```python
class Node:
    def __init__(self,name,isMax,value = None,allchildren = None):
        self.name = name
        self.isMax = isMax
        self.value = value
        self.allchildren = allchildren
    

    def isLeaf(self):
        if self.allchildren is None:
            return True
        
    def children(self):
        return self.allchildren
    
    def evaluate(self):
        return self.value
    

def minimax(node,alpha,beta):
    if node.isLeaf():
        return node.evaluate(),None
    
    elif node.isMax:
        max_path = None
        for c in node.children():
            score,path = minimax(c,alpha,beta)
            if score >= beta:
                return score, None
            if score > alpha:
                alpha = score
                max_path = c.name,path
        return alpha,max_path
    
    else:
        min_path = None
        for c in node.children():
            score,path = minimax(c,alpha,beta)
            if score <= alpha:
                return score,None

            if score < beta:
                beta = score
                min_path = c.name,path

        return beta,min_path
    

leaf_a1 = Node("A1", isMax=False, value=1)
leaf_a2 = Node("A2", isMax=False, value=5)
leaf_a3 = Node("A3", isMax=False, value = 1)

leaf_b1 = Node("B1", isMax=False, value=2)
leaf_b2 = Node("B2", isMax=False, value=9)

# Intermediate nodes
child_a = Node("A", isMax=False, allchildren=[leaf_a1, leaf_a2])
child_b = Node("B", isMax=False, allchildren=[leaf_b1, leaf_b2])

# Root node
root = Node("Root", isMax=True, allchildren=[child_a, child_b])

val, path = minimax(root,float('-inf'),float('inf'))
print(val)

print(path)
```

**Testing Minimax - αβ-pruning:**
```bash
2
('B', ('B1', None))
```


## Assignment 7
**Date :** 25/09/2024

**Problem Description :** 

**Algorithm Top-Down:**
```bash
Input: Knowledge Base (KB)
Output: Fixed point of the KB

function prove(KB, ans_body, indent=""):
    print(indent + 'yes <- ' + join(ans_body with " & "))

    if ans_body is not empty then
        selected ← ans_body[0]
        
        if selected is an askable in KB then
            ask user if selected is true
            if user confirms selected is true then
                return prove(KB, ans_body[1:], indent + " ")
            else
                return False
        
        else
            for each clause in KB.clauses_for_atom(selected) do
                if prove(KB, clause.body + ans_body[1:], indent + " ") then
                    return True

            return False

    else  
        return True
```

**Code Top-Down:** 
```python
class Clause:
    def __init__(self,head,body = []):
        self.head = head
        self.body = body

    def __repr__(self):
        return f"{self.head} <- {'&'.join(str(a) for a in self.body)}"
    
class Askable:
    def __init__(self,atom):
        self.atom = atom

    def __repr___(self):
        return f"{self.atom}"
    
def yes(ans):
    return True if ans.lower() in ['yes','y'] else False


class KB(object):
    def __init__(self,statements = []):
        self.statements = statements
        self.askables = [a.atom for a in statements if isinstance(a,Askable)]
        self.clauses = [c for c in statements if isinstance(c,Clause)]

        self.atom_to_clauses = {}

        for c in self.clauses:
            self.add_clause(c)

    def add_clause(self,c):
        if c.head in self.atom_to_clauses:
            self.atom_to_clauses[c.head].append(c)

        else:
            self.atom_to_clauses[c.head] = [c]

    def clauses_for_atom(self,a):
        return self.atom_to_clauses[a] if a in self.atom_to_clauses else []
    

elect = KB([
Clause('light_l1'),
Clause('light_l2'),
Clause('ok_l1'),
Clause('ok_l2'),
Clause('ok_cb1'),
Clause('ok_cb2'),
Clause('live_outside'),
Clause('live_l1', ['live_w0']),
Clause('live_w0', ['up_s2','live_w1']),
Clause('live_w0', ['down_s2','live_w2']),
Clause('live_w1', ['up_s1', 'live_w3']),
Clause('live_w2', ['down_s1','live_w3' ]),
Clause('live_l2', ['live_w4']),
Clause('live_w4', ['up_s3','live_w3' ]),
Clause('live_p_1', ['live_w3']),
Clause('live_w3', ['live_w5', 'ok_cb1']),
Clause('live_p_2', ['live_w6']),
Clause('live_w6', ['live_w5', 'ok_cb2']),
Clause('live_w5', ['live_outside']),
Clause('lit_l1', ['light_l1', 'live_l1', 'ok_l1']),
Clause('lit_l2', ['light_l2', 'live_l2', 'ok_l2']),
Askable('up_s1'),
Askable('down_s1'),
Askable('up_s2'),
Askable('down_s2'),
Askable('up_s3'),
Askable('down_s2')
])




def prove(kb,ans_body,indent=""):
    print(indent + 'yes <- ' + ' & '.join(ans_body))
    if ans_body:
        selected = ans_body[0]
        if selected in kb.askables:
            return yes(input(f"Is {selected} True?: ")) and prove(kb,ans_body[1:],indent + " ")
        
        else:
            return any(prove(kb,cl.body + ans_body[1:],indent + " ") for cl in kb.clauses_for_atom(selected))
        
    else:
        return True


prove(elect, ['live_w0'])

```
**Testing Top-Down:**
```bash
yes <- live_w0
 yes <- up_s2 & live_w1
Is up_s2 True?: yes
  yes <- live_w1
   yes <- up_s1 & live_w3
Is up_s1 True?: no
 yes <- down_s2 & live_w2
Is down_s2 True?: yes
  yes <- live_w2
   yes <- down_s1 & live_w3
Is down_s1 True?: no
```


**Algorithm Bottom-Up:**
```bash
Input: Knowledge Base (KB)
Output: Fixed point of the KB

function fixed_point(KB):
    fp ← ask_askables(KB)
    added ← True

    while added do
        added ← False  // Indicates if an atom was added this iteration

        for each clause in KB.clauses do
            if clause.head is not in fp and all elements of clause.body are in fp then
                add clause.head to fp
                added ← True
                print(clause.head, "added to fixed point due to clause:", clause)

    return fp
```

**Code Bottom-Up:** 
```python
class Clause(object):
    """A definite clause"""

    def __init__(self, head, body=[]):
        """Clause with atom head and list of atoms body"""
        self.head = head
        self.body = body

    def __repr__(self):
        """Returns the string representation of a clause."""
        if self.body:
            return f"{self.head} <- {' & '.join(str(a) for a in self.body)}."
        else:
            return f"{self.head}."


class Askable(object):
    """An askable atom"""

    def __init__(self, atom):
        """Clause with atom head and list of atoms body"""
        self.atom = atom

    def __str__(self):
        """Returns the string representation of a clause."""
        return "askable " + self.atom + "."

    @staticmethod
    def yes(ans):
        """Returns true if the answer is yes in some form"""
        return ans.lower() in ['yes', 'oui', 'y']  # bilingual


class KB:
    """A knowledge base consists of a set of clauses.
    This also creates a dictionary to give fast access to the clauses with an atom in head.
    """

    def __init__(self, statements=[]):
        self.statements = statements
        self.clauses = [c for c in statements if isinstance(c, Clause)]
        self.askables = [c.atom for c in statements if isinstance(c, Askable)]
        self.atom_to_clauses = {}  # Dictionary giving clauses with atom as head
        for c in self.clauses:
            self.add_clause(c)

    def add_clause(self, c):
        if c.head in self.atom_to_clauses:
            self.atom_to_clauses[c.head].append(c)
        else:
            self.atom_to_clauses[c.head] = [c]

    def clauses_for_atom(self, a):
        """Returns list of clauses with atom a as the head"""
        return self.atom_to_clauses.get(a, [])

    def __str__(self):
        """Returns a string representation of this knowledge base."""
        return '\n'.join([str(c) for c in self.statements])


triv_KB = KB([
    Clause('i_am', ['i_think']),
    Clause('i_think'),
    Clause('i_smell', ['i_exist'])
])


def fixed_point(kb):
    """Returns the fixed point of knowledge base kb."""
    fp = ask_askables(kb)
    added = True
    while added:
        added = False  # added is true when an atom was added to fp this iteration
        for c in kb.clauses:
            if c.head not in fp and all(b in fp for b in c.body):
                fp.add(c.head)
                added = True
                print(f"{c.head} added to fixed point due to clause: {c}")
    return fp


def ask_askables(kb):
    return {at for at in kb.askables if Askable.yes(input(f"Is {at} true? "))}


def test(kb=triv_KB, fixedpt={'i_am', 'i_think'}):
    fp = fixed_point(kb)
    assert fp == fixedpt, f"kb gave result {fp}"
    print("Passed unit test")


if __name__ == "__main__":
    test()
```

**Testing Bottom-Up:**
```bash
i_think added to fixed point due to clause: i_think.
i_am added to fixed point due to clause: i_am <- i_think.
Passed unit test
```

## Assignment 8
**Date :** 25/09/2024

**Problem Description :** 

Inference using Bayesian Network (BN) – Joint Probability Distribution
The given Bayesian Network has 5 variables with the dependency between the variables as shown below:
 

1. The marks (M) of a student depends on:
- Exam level (E): This is a discrete variable that can take two values, (difficult, easy) and
- IQ of the student (I): A discrete variable that can take two values (high, low)
2. The marks (M) will, in turn, predict whether he/she will get admitted (A) to a university.
3. The IQ (I) will also predict the aptitude score (S) of the student.

Write functions to

1. Construct the given DAG representation using appropriate libraries.
2. Read and print the Conditional Probability Table (CPT) for each variable.
3. Calculate the joint probability distribution of the BN using 5 variables.
Observation: Write the formula for joint probability distribution and explain each parameter.
Justify the answer with the advantage of BN.

**Algorithm :**
```bash

Input: Prior probabilities for hypotheses H
Output: Posterior Probability P(H|E)

function bayes_algorithm(P_H, P_E_given_H):
    P_E ← 0
    for each hypothesis H in P_H:
        P_E ← P_E + P(E | H) * P(H)  \

    for each hypothesis H in P_H:
        P_H_given_E[H] ← (P(E | H) * P(H)) / P_E   

    return P_H_given_E
```

**Code :** 
```python
P_e = {0: 0.7, 1: 0.3}  
P_i = {0: 0.8, 1: 0.2}  

P_m_given_e_i = {
    (0, 0): {0: 0.6, 1: 0.4},
    (0, 1): {0: 0.1, 1: 0.9},
    (1, 0): {0: 0.5, 1: 0.5},
    (1, 1): {0: 0.2, 1: 0.8}
}  

P_a_given_m = {
    0: {0: 0.6, 1: 0.4},
    1: {0: 0.9, 1: 0.1}
}  

P_s_given_i = {
    0: {0: 0.75, 1: 0.25},
    1: {0: 0.4, 1: 0.6}
}  

def print_cpd_exam_level():
    print("CPD for Exam Level (e):")
    print("+----------------+------+")
    print("| e              | P(e) |")
    print("+----------------+------+")
    for e_state, prob in P_e.items():
        print(f"| {e_state:<14} | {prob:<4} |")
    print("+----------------+------+\n")

def print_cpd_iq():
    print("CPD for IQ (i):")
    print("+----------------+------+")
    print("| i              | P(i) |")
    print("+----------------+------+")
    for i_state, prob in P_i.items():
        print(f"| {i_state:<14} | {prob:<4} |")
    print("+----------------+------+\n")

def print_cpd_marks():
    print("CPD for Marks (m):")
    print("+----------------+----------------+----------------+----------------+")
    print("| e              | i              | P(m=0)        | P(m=1)        |")
    print("+----------------+----------------+----------------+----------------+")
    for (e_state, i_state), m_probs in P_m_given_e_i.items():
        print(f"| {e_state:<14} | {i_state:<14} | {m_probs[0]:<14} | {m_probs[1]:<14} |")
    print("+----------------+----------------+----------------+----------------+\n")

def print_cpd_admission():
    print("CPD for Admission (a):")
    print("+----------------+----------------+----------------+")
    print("| m              | P(a=0)        | P(a=1)        |")
    print("+----------------+----------------+----------------+")
    for m_state, a_probs in P_a_given_m.items():
        print(f"| {m_state:<14} | {a_probs[0]:<14} | {a_probs[1]:<14} |")
    print("+----------------+----------------+----------------+\n")

def print_cpd_aptitude_score():
    print("CPD for Aptitude Score (s):")
    print("+----------------+----------------+----------------+")
    print("| i              | P(s=0)        | P(s=1)        |")
    print("+----------------+----------------+----------------+")
    for i_state, s_probs in P_s_given_i.items():
        print(f"| {i_state:<14} | {s_probs[0]:<14} | {s_probs[1]:<14} |")
    print("+----------------+----------------+----------------+\n")

def calculate_jpd(e_state, i_state, m_state, a_state, s_state):
    P_e_val = P_e[e_state]
    P_i_val = P_i[i_state]
    P_m_val = P_m_given_e_i[(e_state, i_state)][m_state]
    P_a_val = P_a_given_m[m_state][a_state]
    P_s_val = P_s_given_i[i_state][s_state]
    jpd = P_e_val * P_i_val * P_m_val * P_a_val * P_s_val
    return jpd

def print_jpd_table():
    print("Joint Probability Distribution Table:")
    print("+----------------+----------------+----------------+----------------+----------------+----------------+")
    print("| e              | i              | m              | a              | s              | P(e, i, m, a, s)|")
    print("+----------------+----------------+----------------+----------------+----------------+----------------+")
    for e_state in P_e.keys():
        for i_state in P_i.keys():
            for m_state in [0, 1]:
                for a_state in [0, 1]:
                    for s_state in [0, 1]:
                        jpd = calculate_jpd(e_state, i_state, m_state, a_state, s_state)
                        print(f"| {e_state:<14} | {i_state:<14} | {m_state:<14} | {a_state:<14} | {s_state:<14} | {jpd:<14.4f} |")
    print("+----------------+----------------+----------------+----------------+----------------+----------------+")

def print_jpd_formula():
    print("Joint Probability Distribution Formula:")
    print("P(e, i, m, a, s) = P(e) * P(i) * P(m | e, i) * P(a | m) * P(s | i)\n")
    print("Where:")
    print(" P(e): Probability of Exam Level")
    print(" P(i): Probability of IQ")
    print(" P(m | e, i): Probability of Marks given Exam Level and IQ")
    print(" P(a | m): Probability of Admission given Marks")
    print(" P(s | i): Probability of Aptitude Score given IQ\n")

def get_input_and_print_probability():
    print("Enter the states for the following variables (leave blank for unknown):")
    e_state = input("Exam Level (e) [0=easy/1=difficult]: ").strip() or None
    i_state = input("IQ (i) [0=low/1=high]: ").strip() or None
    m_state = input("Marks (m) [0=low/1=high]: ").strip() or None
    a_state = input("Admission (a) [0=no/1=yes]: ").strip() or None
    s_state = input("Aptitude Score (s) [0=poor/1=good]: ").strip() or None

    e_state = int(e_state) if e_state is not None else None
    i_state = int(i_state) if i_state is not None else None
    m_state = int(m_state) if m_state is not None else None
    a_state = int(a_state) if a_state is not None else None
    s_state = int(s_state) if s_state is not None else None

    valid_states_e = list(P_e.keys())
    valid_states_i = list(P_i.keys())
    valid_states_m = [0, 1]
    valid_states_a = [0, 1]
    valid_states_s = [0, 1]

    states_to_check = {
        'e': valid_states_e if e_state is None else [e_state],
        'i': valid_states_i if i_state is None else [i_state],
        'm': valid_states_m if m_state is None else [m_state],
        'a': valid_states_a if a_state is None else [a_state],
        's': valid_states_s if s_state is None else [s_state],
    }

    total_jpd = 0
    print("\nCalculating JPD for the following combinations:")
    for e in states_to_check['e']:
        for i in states_to_check['i']:
            for m in states_to_check['m']:
                for a in states_to_check['a']:
                    for s in states_to_check['s']:
                        jpd = calculate_jpd(e, i, m, a, s)
                        total_jpd += jpd
                        print(f"P(e={e}, i={i}, m={m}, a={a}, s={s}) = {jpd:.4f}")

    print(f"\nTotal Joint Probability for the given states = {total_jpd:.4f}")

print_cpd_exam_level()
print_cpd_iq()
print_cpd_marks()
print_cpd_admission()
print_cpd_aptitude_score()

print_jpd_table()
print_jpd_formula()

get_input_and_print_probability()
```

**Testing :**
```bash
CPD for Exam Level (e):
+----------------+------+
| e              | P(e) |
+----------------+------+
| 0              | 0.7  |
| 1              | 0.3  |
+----------------+------+

CPD for IQ (i):
+----------------+------+
| i              | P(i) |
+----------------+------+
| 0              | 0.8  |
| 1              | 0.2  |
+----------------+------+

CPD for Marks (m):
+----------------+----------------+----------------+----------------+
| e              | i              | P(m=0)        | P(m=1)        |
+----------------+----------------+----------------+----------------+
| 0              | 0              | 0.6            | 0.4            |
| 0              | 1              | 0.1            | 0.9            |
| 1              | 0              | 0.5            | 0.5            |
| 1              | 1              | 0.2            | 0.8            |
+----------------+----------------+----------------+----------------+

CPD for Admission (a):
+----------------+----------------+----------------+
| m              | P(a=0)        | P(a=1)        |
+----------------+----------------+----------------+
| 0              | 0.6            | 0.4            |
| 1              | 0.9            | 0.1            |
+----------------+----------------+----------------+

CPD for Aptitude Score (s):
+----------------+----------------+----------------+
| i              | P(s=0)        | P(s=1)        |
+----------------+----------------+----------------+
| 0              | 0.75           | 0.25           |
| 1              | 0.4            | 0.6            |
+----------------+----------------+----------------+

Joint Probability Distribution Table:
+----------------+----------------+----------------+----------------+----------------+----------------+
| e              | i              | m              | a              | s              | P(e, i, m, a, s)|
+----------------+----------------+----------------+----------------+----------------+----------------+
| 0              | 0              | 0              | 0              | 0              | 0.1512         |
| 0              | 0              | 0              | 0              | 1              | 0.0504         |
| 0              | 0              | 0              | 1              | 0              | 0.1008         |
| 0              | 0              | 0              | 1              | 1              | 0.0336         |
| 0              | 0              | 1              | 0              | 0              | 0.1512         |
| 0              | 0              | 1              | 0              | 1              | 0.0504         |
| 0              | 0              | 1              | 1              | 0              | 0.0168         |
| 0              | 0              | 1              | 1              | 1              | 0.0056         |
| 0              | 1              | 0              | 0              | 0              | 0.0034         |
| 0              | 1              | 0              | 0              | 1              | 0.0050         |
| 0              | 1              | 0              | 1              | 0              | 0.0022         |
| 0              | 1              | 0              | 1              | 1              | 0.0034         |
| 0              | 1              | 1              | 0              | 0              | 0.0454         |
| 0              | 1              | 1              | 0              | 1              | 0.0680         |
| 0              | 1              | 1              | 1              | 0              | 0.0050         |
| 0              | 1              | 1              | 1              | 1              | 0.0076         |
| 1              | 0              | 0              | 0              | 0              | 0.0540         |
| 1              | 0              | 0              | 0              | 1              | 0.0180         |
| 1              | 0              | 0              | 1              | 0              | 0.0360         |
| 1              | 0              | 0              | 1              | 1              | 0.0120         |
| 1              | 0              | 1              | 0              | 0              | 0.0810         |
| 1              | 0              | 1              | 0              | 1              | 0.0270         |
| 1              | 0              | 1              | 1              | 0              | 0.0090         |
| 1              | 0              | 1              | 1              | 1              | 0.0030         |
| 1              | 1              | 0              | 0              | 0              | 0.0029         |
| 1              | 1              | 0              | 0              | 1              | 0.0043         |
| 1              | 1              | 0              | 1              | 0              | 0.0019         |
| 1              | 1              | 0              | 1              | 1              | 0.0029         |
| 1              | 1              | 1              | 0              | 0              | 0.0173         |
| 1              | 1              | 1              | 0              | 1              | 0.0259         |
| 1              | 1              | 1              | 1              | 0              | 0.0019         |
| 1              | 1              | 1              | 1              | 1              | 0.0029         |
+----------------+----------------+----------------+----------------+----------------+----------------+
Joint Probability Distribution Formula:
P(e, i, m, a, s) = P(e) * P(i) * P(m | e, i) * P(a | m) * P(s | i)

Where:
 P(e): Probability of Exam Level
 P(i): Probability of IQ
 P(m | e, i): Probability of Marks given Exam Level and IQ
 P(a | m): Probability of Admission given Marks
 P(s | i): Probability of Aptitude Score given IQ

Enter the states for the following variables (leave blank for unknown):
Exam Level (e) [0=easy/1=difficult]: 0
IQ (i) [0=low/1=high]: 1
Marks (m) [0=low/1=high]: 0
Admission (a) [0=no/1=yes]:
Aptitude Score (s) [0=poor/1=good]: 1

Calculating JPD for the following combinations:
P(e=0, i=1, m=0, a=0, s=1) = 0.0050
P(e=0, i=1, m=0, a=1, s=1) = 0.0034

Total Joint Probability for the given states = 0.0084

```
## Game Report - Super Mario: The Adventure Quest

<div align="center">

Name: Nivetha Dhanakoti, Pranav Moorthi
Reg No.: 3122 22 5001 087, 3122 22 5001 089
Date: 10-11-2024

</div>

**Date :** 12/11/2024

### Problem Description

Super Mario: The Adventure Quest is a MarioBros-style platformer where players navigate through a series of obstacles, collect coins, and avoid enemies and hazards. The objective is to reach the endpoint of each level after collecting all available coins, represented by a flag or specific location. The game includes standard platformer mechanics, such as moving left and right, jumping, and interacting with objects in the environment. 

The game world is filled with classic platformer elements, including breakable and unbreakable blocks, moving platforms, and hidden passages, adding layers of complexity and exploration to each level. Players must master timing and precision to dodge enemies and hazards, which are strategically placed to challenge their progress. Enemies have unique behaviors, such as patrolling specific areas, following set paths, or actively pursuing the player when in close range. Hazards like fire, spikes, or pits add an extra layer of difficulty, often requiring the player to pause and observe patterns before advancing.

**Features of Game**

**1) Collecting All Coins**  
The primary objective of the game is to collect all coins present in each level. Coins are strategically scattered throughout the environment, often guarded by non-player character (NPC) enemies or positioned in challenging locations that require precise jumping and movement to access. Successfully gathering every coin is essential for player progression and completion of the level.

**2) Navigating NPC Enemies**  
Players must also navigate and overcome various NPC enemies that exhibit distinct behaviors. These enemies may patrol specific areas, chase the player, or block pathways, each requiring different strategies to either avoid or engage. Mastering the interactions with these enemies is crucial for advancing through the levels and achieving success.

**3) Environmental Obstacles**  
The game features a variety of environmental obstacles that challenge the player’s platforming skills. Players must maneuver around platforms, walls, pits, and spike traps while avoiding harmful hazards such as fire, spikes, and falling rocks. Success in this aspect of the game hinges on careful timing and precise movements to dodge these dangers effectively.

**4) Reaching the Level Endpoint**  
Ultimately, the game culminates in the player reaching the endpoint of each level. After successfully collecting all coins, navigating obstacles, and overcoming hazards, the player must arrive at this endpoint, which could be represented by a flag, door, or designated location. The completion of this objective signifies the successful conclusion of the level.


### Classes Implemented

**1. Player Class**
   
The Player class is responsible for managing the player's behavior and interactions within the game world. It handles the player's movement, gravity effects, collisions, and animation states (e.g., jumping, walking, idle). This class is central to the game, as it directly represents the character the player controls.
•	Movement: The class controls the player's movement left and right using keyboard input. It ensures that the player can only move within the boundaries of the game environment.
•	Jumping: The player can jump by applying an upward velocity. The class handles jump initiation and gravity effects, ensuring the player moves up and falls back down.
•	Collision Detection: The player detects and responds to collisions with other objects, such as platforms or walls. If the player lands on a platform, it stops falling and can walk on it.
•	Animation: Based on the player’s state (e.g., walking, jumping), the class updates the sprite to show the correct animation.
Health and Coin Collection: The class tracks the player's health and coin count. The player's health decreases if they hit an obstacle or enemy, and coins are collected when the player touches them.
- **Variables**:
  - `COLOR`: Color of the player for debugging or other purposes.
  - `GRAVITY`: Gravity constant for calculating fall speed.
  - `SPRITES`: Holds the loaded animations for the player.
  - `ANIMATION_DELAY`: Delay between frames in animation.
  - `rect`: Position and dimensions of the player.
  - `x_vel`, `y_vel`: Horizontal and vertical velocities.
  - `mask`: Used for collision detection.
  - `direction`: Direction the player is facing (`left` or `right`).
  - `animation_count`: Counter for tracking animation frames.
  - `fall_count`: Tracks the duration of falling.
  - `jump_count`: Tracks the number of jumps.
  - `hit`: Boolean indicating if the player is hit.
  - `hit_count`: Counter for how long the player remains in the hit state.
  - `coins`: Count of coins collected.
  - `health`: Player's health points.

- **Methods**:
  - `jump()`: Initiates a jump, updating the vertical velocity and resetting fall counters.
  - `make_hit(hit_type)`: Reduces health based on the type of hit (`fire` or other).
  - `move(dx, dy)`: Moves the player by `dx` and `dy`.
  - `move_left(vel)`, `move_right(vel)`: Moves the player left or right, changing the direction and resetting animation if necessary.
  - `loop(fps)`: Updates the player’s position, checks for hits, and animates per frame.
  - `update()`: Updates the player’s collision mask and position.
  - `landed()`: Resets fall and jump counters when the player lands.
  - `hit_head()`: Inverts vertical velocity if the player hits a ceiling.
  - `update_sprite()`: Updates the current sprite based on the player's state (e.g., running, jumping).
  - `all_coins_collected(coins)`: Checks if all coins have been collected.
  - `draw(win, offset_x)`: Draws the player on the screen at the current position with an offset.


**2. Object Class**
   
The Object class serves as the foundation for all in-game objects like platforms, blocks, coins, etc. It provides a common structure for defining the properties and behaviors of objects in the game world.
•	Positioning: The class tracks an object’s position using a rectangle (rect), which defines where the object is located on the screen and its size.
•	Rendering: Objects can be drawn to the screen using an image or shape associated with them.
Collision Detection: The rect of each object is used to detect collisions with the player or other objects in the game.
- **Variables**:
  - `rect`: Position and dimensions.
  - `image`: Surface to represent the object visually.
  - `width`, `height`: Dimensions of the object.
  - `name`: Identifier for the object type.

- **Methods**:
  - `draw(win, offset_x)`: Draws the object on the screen with a horizontal offset.


**3. Block Class**
   
The Coin class is a specialized version of the Object class, designed to represent collectible coins within the game. When the player collects a coin, their coin count increases.
•	Rendering: Coins are drawn on the screen in their respective positions.
•	Collection: The class detects when the player touches or collects the coin. Upon collection, it adds to the player's coin total.
•	Display: The coin’s visual representation may be an image that shows up as a coin icon or shape within the game world.

- **Variables**:
  - Inherits variables from `Object`.
  - `image`: Surface with block texture from the `get_block()` function.
  - `mask`: Mask for collision detection.

- **Methods**:
  - Inherits `draw()` from `Object`.
 
**4. Fire Class**
   
Represents a fire hazard that can harm the player.

- **Variables**:
  - `ANIMATION_DELAY`: Delay between frames in animation.
  - `fire`: Contains fire animation frames.
  - `image`: Current frame of the fire animation.
  - `mask`: Mask for collision detection.
  - `animation_count`: Counter to track frames for animation.
  - `animation_name`: Indicates whether fire is "on" or "off".

- **Methods**:
  - `on()`: Turns the fire animation to the "on" state.
  - `off()`: Turns the fire animation to the "off" state.
  - `loop()`: Updates the fire animation based on `animation_name`, resets `animation_count` if necessary.
 
**5. Coins Class**
   
Represents collectible coins in the game.

- **Variables**:
  - `ANIMATION_DELAY`: Delay between frames for coin animation.
  - `rect`: Position and dimensions of the coin.
  - `coin_sprites`: List of coin animation frames.
  - `image`: Current image for the coin.
  - `mask`: Mask for collision detection.
  - `animation_count`: Counter for tracking animation frames.
  - `collected`: Boolean indicating if the coin is collected.

- **Methods**:
  - `collect()`: Sets the coin as collected.
  - `loop()`: Animates the coin if it hasn’t been collected, resets `animation_count` if necessary.
  - `draw(win, offset_x)`: Draws the coin on the screen if it hasn’t been collected.
 
**Algorithms**

1. Local Search
```bash
Input: current position (node), target position (goal_state), speed, player, obstacles (objects)
Output: Updated position

function move_local(node, goal_state, speed, objects):
    moves ← {
        "right": (node.x + speed, node.y),
        "left": (node.x - speed, node.y),
        "down": (node.x, node.y + speed),
        "up": (node.x, node.y - speed)
    }
    
    best_move ← None
    current_heuristic ← ∞

    for each move, position in moves do
        if no collision at position with objects then
            heuristic ← calculate_heuristic(position, goal_state)
            
            if heuristic < current_heuristic then
                best_move ← position
                current_heuristic ← heuristic

    if best_move is not None then
        update node.x, node.y to best_move

    return node
```

2. A* Path Finding
```bash
Input: Start position, goal position, movement step, obstacle list (objects)
Output: Path from start to goal

function heuristic(pos1, pos2):
    return abs(pos1.x - pos2.x) + abs(pos1.y - pos2.y)


function find_path(start, goal_state, objects):
    frontier ← PriorityQueue with (0, start, start)
    explored ← empty set
    path ← empty

    while frontier is not empty do
        g_cost, node, path ← frontier.pop_with_lowest_cost()
        add node to explored
        neighbours ← [
            (node.x + step, node.y), 
            (node.x - step, node.y), 
            (node.x, node.y + step), 
            (node.x, node.y - step)
        ]

        for each neighbour in neighbours do
            if neighbour equals goal_state then
                add neighbour to path
                return path

            else if neighbour in explored or collision(neighbour, objects) or is_out_of_bounds(neighbour) then
                continue

            else
                score ← g_cost + heuristic(node, goal_state)
                new_path ← path + (node)
                add (score, neighbour, new_path) to frontier

    return failure  
```

**Code :**
```python
import os
import random
import math
import pygame
from os import listdir
from os.path import isfile, join
import heapq
import math
import random

pygame.init()

pygame.display.set_caption("MarioBros")

BG_COLOR = (255,255,255)
WIDTH, HEIGHT  = 1000,800
FPS = 60
PLAYER_VEL = 5


# Load start image
def get_start_screen():
    start_image = pygame.image.load("assets/background/start.png")  # Load your start image from path
    start_image = pygame.transform.scale(start_image, (WIDTH, HEIGHT))  # Scale to fit the screen
    return start_image

# Function to draw start text
def draw_start_text(win):
    x = WIDTH // 2  # Center the image horizontally
    y = HEIGHT // 2 + 50
    start_img = pygame.image.load("assets/addons/start.png")
    start_text_rect = start_img.get_rect(topleft=(x - 110, y - 36))  # Get rectangle for detecting clicks
    win.blit(start_img, (x - 300, y - 10))
    return start_text_rect

def game_end_check(player, last_block, coins):
    if player.rect.colliderect(last_block.rect) and player.all_coins_collected(coins):
        # Show the game ending screen
        display_end_screen()

def display_end_screen():
    # Load and display the "end.png" image
    end_img = pygame.image.load("assets/background/end.png")
    end_img = pygame.transform.scale(end_img, (WIDTH, HEIGHT))
    window.blit(end_img, (0, 0))
    pygame.display.update()
    pygame.time.wait(3000)  # Wait for 3 seconds before closing or restarting
    pygame.quit()
    exit()  # End the game


window = pygame.display.set_mode((WIDTH,HEIGHT))

def flip(sprites):
    return [pygame.transform.flip(sprite,True,False) for sprite in sprites]


def load_sprite_sheets(dir1, dir2, width, height, direction = False):
    path = join("assets" , dir1 ,dir2)
    images = [f for f in listdir(path) if isfile(join(path,f))]

    all_sprites = {}

    for image in images:
        sprite_sheet = pygame.image.load(join(path,image)).convert_alpha()

        sprites = []
        for i in range(sprite_sheet.get_width() // width): #divide to find number of images
            surface = pygame.Surface((width,height) , pygame.SRCALPHA , 32)
            rect = pygame.Rect(i * width, 0 ,width, height)
            surface.blit(sprite_sheet,(0,0),rect)
            sprites.append(pygame.transform.scale2x(surface))

        if direction:
            all_sprites[image.replace(".png","") + "_right"] = sprites
            all_sprites[image.replace(".png","") + "_left"] = flip(sprites)

        else:
            all_sprites[image.replace(".png","")] = sprites
    return all_sprites


def get_block(size):
    path = join("assets","Terrain","Terrain.png")
    image = pygame.image.load(path).convert_alpha()
    surface = pygame.Surface((size,size), pygame.SRCALPHA,32)
    rect = pygame.Rect(96, 0 ,size, size)
    surface.blit(image, (0,0) ,rect)
    return pygame.transform.scale2x(surface)



class Player(pygame.sprite.Sprite): #Sprite object
    COLOR = (255,0,0)
    GRAVITY = 1
    SPRITES = load_sprite_sheets("MainCharacters","NinjaFrog", 32, 32, True)
    ANIMATION_DELAY = 3
    

    def __init__(self,x,y,width,height):
        super().__init__()
        self.rect = pygame.Rect(x,y,width,height)
        self.x_vel = 0
        self.y_vel = 0
        self.mask = None
        self.direction = 'left'
        self.animation_count = 0
        self.fall_count = 0 #for gravity
        self.jump_count = 0
        self.hit = False
        self.hit_count = 0
        self.coins = 0
        self.health = 100

    def jump(self):
        self.y_vel = -self.GRAVITY * 8
        self.animation_count = 0
        self.jump_count += 1
        if self.jump_count == 1:
            self.fall_count = 0

        
    def make_hit(self,hit_type):
        self.hit = True
        self.hit_count = 0
        if hit_type == 'fire':
            if self.health > 0:
                self.health -= 1 
        else:
            if self.health > 0:
                self.health -= 2


    
    def move(self,dx,dy):
        self.rect.x += dx #dist move to x axis
        self.rect.y += dy #dist move to y axis

    def move_left(self,vel):
        self.x_vel = -vel
        if self.direction != 'left':
            self.direction = 'left'
            self.animation_count = 0


    def move_right(self,vel):
        self.x_vel = vel
        if self.direction != 'right':
            self.direction = 'right'
            self.animation_count = 0
    
    def loop(self,fps): #update animation and stuff per frame
        self.y_vel += min(1,(self.fall_count / fps) * self.GRAVITY)
        self.move(self.x_vel,self.y_vel)

        if self.hit:
            self.hit_count +=1
        if self.hit_count > fps * 2:
            self.hit = False
            self.hit_count = 0
        self.fall_count += 1
        self.update_sprite()
    
    def update(self):
        self.rect = self.sprite.get_rect(topleft=(self.rect.x,self.rect.y))
        self.mask = pygame.mask.from_surface(self.sprite)

    def landed(self):
        self.fall_count = 0
        self.y_vel = 0
        self.jump_count = 0

    def hit_head(self):
        self.count = 0
        self.y_vel *= -1


    def update_sprite(self):
        sprite_sheet = 'idle'
        if self.hit:
            sprite_sheet = 'hit'
        elif self.y_vel < 0:
            if self.jump_count == 1:
                sprite_sheet = 'jump'
            elif self.jump_count == 2:
                sprite_sheet = "double_jump"
        elif self.y_vel > self.GRAVITY * 2:
            sprite_sheet = 'fall'
        elif self.x_vel != 0:
            sprite_sheet = 'run'


        sprite_sheet_name = sprite_sheet + "_" + self.direction
        sprites = self.SPRITES[sprite_sheet_name]
        sprite_index = (self.animation_count // self.ANIMATION_DELAY) % len(sprites)
        self.sprite = sprites[sprite_index]
        self.animation_count+=1
        self.update()

    def all_coins_collected(self, coins):
        # Check if all coins are collected
        for coin in coins:
            if not coin.collected:
                return False
        return True

    def draw(self,win,offset_x):
        win.blit(self.sprite,(self.rect.x - offset_x,self.rect.y))

class Object(pygame.sprite.Sprite):
    def __init__(self,x,y,width,height,name = None):
        super().__init__()
        self.rect = pygame.Rect(x,y,width,height)
        self.image = pygame.Surface((width, height), pygame.SRCALPHA)
        self.width = width
        self.height = height
        self.name = name

    def draw(self, win, offset_x):
        win.blit(self.image, (self.rect.x - offset_x,self.rect.y))

class Block(Object):
    def __init__(self, x,y,size):
        super().__init__(x,y,size,size)
        block = get_block(size)
        self.image.blit(block,(0,0))
        self.mask = pygame.mask.from_surface(self.image)

class Fire(Object):
    ANIMATION_DELAY = 3

    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height, "fire")
        self.fire = load_sprite_sheets("Traps", "Fire", width, height)
        self.image = self.fire["off"][0]
        self.mask = pygame.mask.from_surface(self.image)
        self.animation_count = 0
        self.animation_name = "off"

    def on(self):
        self.animation_name = "on"

    def off(self):
        self.animation_name = "off"

    def loop(self):
        sprites = self.fire[self.animation_name]
        sprite_index = (self.animation_count //
                        self.ANIMATION_DELAY) % len(sprites)
        self.image = sprites[sprite_index]
        self.animation_count += 1

        self.rect = self.image.get_rect(topleft=(self.rect.x, self.rect.y))
        self.mask = pygame.mask.from_surface(self.image)

        if self.animation_count // self.ANIMATION_DELAY > len(sprites):
            self.animation_count = 0

class Coins:
    ANIMATION_DELAY = 10  # Adjust delay for coin animation speed

    def __init__(self, x, y, width=16, height=28):
        self.rect = pygame.Rect(x, y, width, height)
        self.coin_sprites = [
            pygame.transform.scale2x(pygame.image.load(join("assets/coins", f"coin{i}.png")).convert_alpha())
            for i in range(2)
        ]
        self.image = self.coin_sprites[0]
        self.mask = pygame.mask.from_surface(self.image)
        self.animation_count = 0
        self.collected = False  

    def collect(self):
        self.collected = True

    def loop(self):
        if not self.collected:
            # Animate the coin if it hasn't been collected
            sprite_index = (self.animation_count // self.ANIMATION_DELAY) % len(self.coin_sprites)
            self.image = self.coin_sprites[sprite_index]
            self.animation_count += 1

            if self.animation_count // self.ANIMATION_DELAY >= len(self.coin_sprites):
                self.animation_count = 0

    def draw(self, win, offset_x):
        if not self.collected:
            win.blit(self.image, (self.rect.x - offset_x, self.rect.y))

class Enemy(pygame.sprite.Sprite):
    ANIMATION_DELAY = 10  # Delay between frames for animation

    def __init__(self, x, y, width=32, height=32):
        super().__init__()
        self.rect = pygame.Rect(x, y, width, height)
        self.enemy_sprites = load_sprite_sheets("MainCharacters", "MaskDude", width, height, True)
        self.image = self.enemy_sprites["idle_right"][0]  # Assuming "fall" animation is available
        self.mask = pygame.mask.from_surface(self.image)
        self.animation_count = 0
        self.direction = 'right'
        self.speed = 2
        self.path = None
        self.frame_count = 0
        self.path_interval = 5 * FPS
        self.path_counter = 1000
        self.previous_moves = []

        self.temperature = 100  # Initial "temperature" for annealing
        self.cooling_rate = 0.995

    def move_left(self,vel):
        self.x_vel = -vel
        if self.direction != 'left':
            self.direction = 'left'
            self.animation_count = 0


    def move_right(self,vel):
        self.x_vel = vel
        if self.direction != 'right':
            self.direction = 'right'
            self.animation_count = 0

    def heuristic(self,pos1,pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


    def manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two points"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def collision(self, node, objects):
        """Check if moving to a position causes a collision."""
        prev_x, prev_y = self.rect.x, self.rect.y
        self.rect.x = node[0]
        self.rect.y = node[1]
        
        for obj in objects:
            if pygame.sprite.collide_mask(self, obj):
                self.rect.x, self.rect.y = prev_x, prev_y
                return True  # Blocked by an object
                
        self.rect.x, self.rect.y = prev_x, prev_y
        return False  # No collision

    def move_local(self, player, objects):
        node = (self.rect.x, self.rect.y)
        goal_state = (player.rect.x, player.rect.y)
        
        moves = {
            'right': (node[0] + self.speed, node[1]),
            'left': (node[0] - self.speed, node[1]),
            'down': (node[0], node[1] + self.speed),
            'up': (node[0], node[1] - self.speed)
        }
        
        best_move = None
        current_heuristic = float('inf')
        
        for move, neigh in moves.items():
            if not self.collision(neigh, objects):
                heuristic = self.heuristic(neigh, goal_state)
                if heuristic < current_heuristic:
                    best_move = neigh
                    current_heuristic = heuristic
        
        if best_move:
            self.rect.x = best_move[0]
            self.rect.y = best_move[1]
    
    def loop(self, player,objects):
        # Handle animation frame update
        sprite_sheet_name = f"run_{self.direction}"
        sprites = self.enemy_sprites[sprite_sheet_name]
        sprite_index = (self.animation_count // self.ANIMATION_DELAY) % len(sprites)
        self.image = sprites[sprite_index]
        self.animation_count += 1

        if self.animation_count // self.ANIMATION_DELAY >= len(sprites):
            self.animation_count = 0

        # Move the enemy
        #self.move(player,objects)
        self.move_local(player,objects)

    def draw(self, win, offset_x):
        win.blit(self.image, (self.rect.x - offset_x, self.rect.y))

def get_background(name):
    pass
    image = pygame.image.load(join("assets/background/",name))
    image = pygame.transform.scale(image, (WIDTH, HEIGHT)) # scaling an image to fit display window
    return image


def draw_health_bar(window, health):
    
    max_health = 100  
    bar_width = 200
    bar_height = 20
    border_color = (0, 0, 0)  
    fill_color = (255, 0, 0)  

    

    # Calculate bar position (top-right corner)
    x = WIDTH - bar_width - 20  # 20px padding from the right
    y = 20  # 20px padding from the top

    pygame.draw.rect(window, border_color, (x - 4, y - 4, bar_width + 8, bar_height + 8),4)

    # Calculate filled portion width
    filled_width = int((health / max_health) * bar_width)

    # Draw the filled red health bar inside the border
    pygame.draw.rect(window, fill_color, (x, y, filled_width, bar_height))

    health_img = pygame.image.load("assets/addons/health2.png")
    window.blit(health_img,(x - 110, y - 36))



    


def handle_vertical_collision(player, objects,dy):
    collided_objects = []
    for obj in objects:
        if pygame.sprite.collide_mask(player, obj):
            if dy > 0:
                player.rect.bottom = obj.rect.top
                player.landed()

            elif dy < 0:
                player.rect.top = obj.rect.bottom
                player.hit_head()

            collided_objects.append(obj)

    return collided_objects

def collide(player, objects, dx):
    player.move(dx , 0)
    player.update()
    collided_object = None
    for obj in objects:
        if pygame.sprite.collide_mask(player, obj):
            collided_object = obj
            break

    player.move(-dx,0)
    player.update()
    return collided_object

def collide_with_enemy(player, enemy):
    if pygame.sprite.collide_mask(player, enemy):
        player.make_hit("enemy")  # Reduce health on collision with enemy


def coins_collect(player, coins):
    for coin in coins:
        if pygame.sprite.collide_mask(player, coin) and coin.collected == False:
            coin.collected = True
            return coin

    return None       
            



def check_death(player):
    if player.health <= 0:
        return True
    
    if player.rect.y > 1010:
        return True
    
    return False

# Update draw function to render enemies
def draw(window, background, player, objects, coins, enemy, offset_x):
    window.blit(background, (0, 0))

    for obj in objects:
        obj.draw(window, offset_x)
    for coin in coins:
        coin.draw(window, offset_x)
    
    enemy.draw(window, offset_x)
    draw_health_bar(window, player.health)
    player.draw(window, offset_x)

    pygame.display.update()

def manhattan(pos1,pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def collision(node,objects):
    #prev_x, prev_y = self.rect.x, self.rect.y
    x = node[0]
    y = node[1]
    
    for obj in objects:
        if obj.rect.collidepoint(node):
            return True

    return False

def find_local_path(player_pos,enemy,objects):
    start = (enemy.rect.x,enemy.rect.y)
    goal_state = player_pos
    path = [start]
    explored = set()

    found = False
    while not(found):
        node = path[len(path) - 1]
        explored.add(node)

        neighbours = [(node[0] + 1,node[1]),(node[0],node[1] + 1),(node[0] - 1,node[1]),(node[0],node[1] - 1)]
        current_heuristic = float('inf')
        best_move = None
        for neigh in neighbours:
            if neigh not in explored and not(collision(neigh,objects)):

                if neigh == goal_state:
                    path.append(neigh)
                    break
                heuristic = manhattan(neigh,goal_state)
                if heuristic < current_heuristic:
                    current_heuristic = heuristic
                    best_move = neigh
        if best_move:
            path.append(best_move)
        else:
            break

    return path




def handle_move(player, objects, enemy):
    keys = pygame.key.get_pressed()

    collide_left = collide(player, objects, -PLAYER_VEL * 2)
    collide_right = collide(player, objects, PLAYER_VEL * 2)
    player.x_vel = 0
    if  keys[pygame.K_a] and not collide_left:
        player.move_left(PLAYER_VEL)

    if keys[pygame.K_d] and not collide_right:
        player.move_right(PLAYER_VEL)

    vertical_collide = handle_vertical_collision(player, objects,player.y_vel)
    to_check = [collide_left,collide_right,*vertical_collide]
    for obj in to_check:
        if obj and obj.name == "fire":
            player.make_hit("fire")

    if coins_collect(player,coins) != None:
        player.coins += 1
        if player.coins == 5:
            print("All coins") # here is where you add function after getting all coins

    collide_with_enemy(player,enemy)
        

block_size = 96
player = Player(100,100,50,50)

floor = [
    Block(i * block_size, HEIGHT - block_size, block_size)
    for i in range(-WIDTH // block_size, (WIDTH * 3) // block_size)
    if i != 19 and i != 20   # Skip blocks at i = 3, 4, and 5 after the last air block
]

fire_list=[
    Fire(400,HEIGHT - block_size - 64,16,32),
    Fire(900,HEIGHT - block_size - 64,16,32),
    Fire(1200,HEIGHT - block_size - 64,16,32),
    Fire(2150,HEIGHT - block_size - 64,16,32),
    Fire(2800,HEIGHT - block_size - 64,16,32)
]

blocks_list = [
    Block(0, HEIGHT - block_size * 2,block_size),
    Block(0, HEIGHT - block_size * 3,block_size),
    Block(0, HEIGHT - block_size * 4,block_size),
    Block(0, HEIGHT - block_size * 5,block_size),
    Block(0, HEIGHT - block_size * 6,block_size),
    Block(0, HEIGHT - block_size * 7,block_size),
    Block(0, HEIGHT - block_size * 8,block_size)
    ]

air_blocks = [
    Block(200, HEIGHT - block_size * 3, block_size),  
    Block(500, HEIGHT - block_size * 5, block_size),  
    Block(600, HEIGHT - block_size * 5, block_size),  
    Block(700, HEIGHT - block_size * 5, block_size),  
    Block(1000, HEIGHT - block_size * 7, block_size),  
    Block(1400, HEIGHT - block_size * 5, block_size), 
    Block(1500, HEIGHT - block_size * 5, block_size), 
    Block(2400, HEIGHT - block_size * 3, block_size), 
    Block(2600, HEIGHT - block_size * 5, block_size), 
    Block(2700, HEIGHT - block_size * 5, block_size)
]

objects = [*floor,*blocks_list,*air_blocks,*fire_list]

coins = [
    Coins(500, HEIGHT - block_size  - 60, 16, 28),
    Coins(600, HEIGHT - block_size *5 - 60, 16, 28),
    Coins(1500, HEIGHT - block_size *5 - 60, 16, 28),
    Coins(2500, HEIGHT - block_size *5 - 60, 16, 28),
    Coins(2700, HEIGHT - block_size *5 - 60, 16, 28)
]

enemy = Enemy(300, 50)

def main(window):

    clock = pygame.time.Clock()
    run = True
    # Start screen loop
    while run:
        clock.tick(FPS)
        window.fill(BG_COLOR)

        # Draw start screen
        window.blit(get_start_screen(), (0, 0))

        # Draw START text
        start_text_rect = draw_start_text(window)

        # Check for events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False  # Exit the game if the window is closed

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button click
                    if start_text_rect.collidepoint(event.pos):  # Check if click was on the "START" text
                        run = False  # Exit the start screen to start the game

            if run == False:  
                pass

        pygame.display.update()

    background = get_background("image.png") # call function to get bg image from 
    game_counter = 0
    #defining a player
    for fire in fire_list:
        fire.on()

    scroll_area_width = 300
    offset_x = 0
    run = True
    while run:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
        
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w and player.jump_count < 2:
                    player.jump()
        player.loop(FPS) #continually set movement 
        
        enemy.loop(player,objects)
        for fire in fire_list:
            fire.loop()
        for coin in coins:
            coin.loop()
        handle_move(player, objects, enemy) #set direction of movement
        last_block = floor[-1]
        game_end_check(player, last_block, coins)
        
        if (check_death(player)):
            break
        draw(window,background,player,objects,coins,enemy,offset_x)

        if ((player.rect.right - offset_x >= WIDTH - scroll_area_width) and player.x_vel > 0) or ((player.rect.left - offset_x <= scroll_area_width) and player.x_vel < 0):
            offset_x += player.x_vel



    pygame.quit()
    quit() 

if __name__  == '__main__':
    main(window)

```

**Testing :**

<div align="center">
    <img width="500" alt="1" src="https://github.com/user-attachments/assets/7272c00f-b50b-4602-bf1b-adb47b4c3964">
    <p>Image 1 : Start Page</p>
</div>
<hr>

<div align="center">
    <img width="500" alt="2" src="https://github.com/user-attachments/assets/af85d135-4820-4362-a882-557f463330a5">
    <p>Image 2 : Initial Page - This is shown once the “START” button is clicked</p>
</div>
<hr>

<div align="center">
    <img width="500" alt="3" src="https://github.com/user-attachments/assets/1eb002e1-c2c0-491d-a53b-2140e2e5f601">
    <p>Image 3 : If Player collides with fire, the player turns to white and health meter decreases</p>
</div>
<hr>

<div align="center">
    <img width="500" alt="4" src="https://github.com/user-attachments/assets/c773c230-55ab-487d-b873-569264fc7ddf">
    <p>Image 4 : If Enemy touches the player, the player turns white and health meter decreases in a faster rate</p>
</div>
<hr>

<div align="center">
    <img width="500" alt="7" src="https://github.com/user-attachments/assets/8be20db4-0bf8-4588-a541-6572b5c4ad9f">
    <p>Image 5 : If player misses to jump when there’s a pit, the game ends</p>
</div>
<hr>

<div align="center">
    <img width="500" alt="6" src="https://github.com/user-attachments/assets/30e3e428-55d1-4b69-bec3-b2431ddc0c97">
    <p>Image 6 : When player reaches the last block of floor after collecting all the coins, then the End page is displayed. If not, the player has missed coins to collect and has to collect all of them for the quest to get over</p>
</div>
<hr>

<div align="center">
    <img width="500" alt="5" src="https://github.com/user-attachments/assets/98107d31-00e6-4279-b06c-fbfa7ea7fa91">
    <p>Image 7 : End Page - This is displayed only when all the coins are collected and the player should reach the last block of the floor.</p>
</div>
<hr>

