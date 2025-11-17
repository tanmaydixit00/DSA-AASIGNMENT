# campus_planner.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import heapq

# -------------------------
# Building ADT
# -------------------------
@dataclass
class Building:
    id: int
    name: str
    location: str

    def __str__(self):
        return f"[ID:{self.id} | {self.name} | {self.location}]"

# -------------------------
# Binary Search Tree (BST)
# -------------------------
class BSTNode:
    def __init__(self, building: Building):
        self.building = building
        self.left: Optional['BSTNode'] = None
        self.right: Optional['BSTNode'] = None

class BST:
    def __init__(self):
        self.root: Optional[BSTNode] = None

    def insert(self, building: Building):
        def _insert(node, b):
            if not node:
                print(f"BST: inserting {b}")
                return BSTNode(b)
            if b.id < node.building.id:
                node.left = _insert(node.left, b)
            elif b.id > node.building.id:
                node.right = _insert(node.right, b)
            else:
                print(f"BST: duplicate id {b.id} ignored")
            return node
        self.root = _insert(self.root, building)

    def search(self, building_id: int) -> Optional[Building]:
        node = self.root
        while node:
            if building_id == node.building.id:
                return node.building
            node = node.left if building_id < node.building.id else node.right
        return None

    def inorder(self) -> List[Building]:
        res = []
        def _in(node):
            if not node: return
            _in(node.left); res.append(node.building); _in(node.right)
        _in(self.root); return res

    def preorder(self) -> List[Building]:
        res=[]
        def _pre(node):
            if not node: return
            res.append(node.building); _pre(node.left); _pre(node.right)
        _pre(self.root); return res

    def postorder(self) -> List[Building]:
        res=[]
        def _post(node):
            if not node: return
            _post(node.left); _post(node.right); res.append(node.building)
        _post(self.root); return res

    def height(self) -> int:
        def _h(node):
            if not node: return 0
            return 1 + max(_h(node.left), _h(node.right))
        return _h(self.root)

# -------------------------
# AVL Tree
# -------------------------
class AVLNode:
    def __init__(self, building: Building):
        self.building = building
        self.left: Optional['AVLNode'] = None
        self.right: Optional['AVLNode'] = None
        self.height: int = 1

class AVLTree:
    def __init__(self):
        self.root: Optional[AVLNode] = None

    def insert(self, building: Building):
        def _height(node: Optional[AVLNode]) -> int:
            return node.height if node else 0

        def _update_height(node: AVLNode):
            node.height = 1 + max(_height(node.left), _height(node.right))

        def _balance_factor(node: AVLNode) -> int:
            return _height(node.left) - _height(node.right)

        def _right_rotate(y: AVLNode) -> AVLNode:
            x = y.left
            T2 = x.right
            x.right = y
            y.left = T2
            _update_height(y); _update_height(x)
            print(f"AVL: RR rotation on {y.building.id} -> new root {x.building.id}")
            return x

        def _left_rotate(x: AVLNode) -> AVLNode:
            y = x.right
            T2 = y.left
            y.left = x
            x.right = T2
            _update_height(x); _update_height(y)
            print(f"AVL: LL rotation on {x.building.id} -> new root {y.building.id}")
            return y

        def _insert(node: Optional[AVLNode], b: Building) -> AVLNode:
            if not node:
                print(f"AVL: inserting {b}")
                return AVLNode(b)
            if b.id < node.building.id:
                node.left = _insert(node.left, b)
            elif b.id > node.building.id:
                node.right = _insert(node.right, b)
            else:
                print(f"AVL: duplicate id {b.id} ignored")
                return node

            _update_height(node)
            balance = _balance_factor(node)

            # LL
            if balance < -1 and b.id > node.right.building.id:
                # Right Right (RR) case in some naming; here we do left rotate (LL in earlier naming)
                return _left_rotate(node)
            # RR
            if balance > 1 and b.id < node.left.building.id:
                return _right_rotate(node)
            # LR
            if balance > 1 and b.id > node.left.building.id:
                print(f"AVL: LR rotation triggered at {node.building.id}")
                node.left = _left_rotate(node.left)
                return _right_rotate(node)
            # RL
            if balance < -1 and b.id < node.right.building.id:
                print(f"AVL: RL rotation triggered at {node.building.id}")
                node.right = _right_rotate(node.right)
                return _left_rotate(node)

            return node

        self.root = _insert(self.root, building)

    def _traverse_order(self, order='in') -> List[Building]:
        res = []
        def _in(n):
            if not n: return
            _in(n.left); res.append(n.building); _in(n.right)
        def _pre(n):
            if not n: return
            res.append(n.building); _pre(n.left); _pre(n.right)
        def _post(n):
            if not n: return
            _post(n.left); _post(n.right); res.append(n.building)
        if order=='in': _in(self.root)
        elif order=='pre': _pre(self.root)
        else: _post(self.root)
        return res

    def inorder(self): return self._traverse_order('in')
    def preorder(self): return self._traverse_order('pre')
    def postorder(self): return self._traverse_order('post')

    def height(self) -> int:
        return self.root.height if self.root else 0

# -------------------------
# Graph (Adjacency List & Matrix)
# -------------------------
class GraphAdjList:
    def __init__(self):
        self.adj: Dict[int, List[Tuple[int, float]]] = {}  # id -> list of (neighbor_id, weight)
        self.buildings: Dict[int, Building] = {}

    def add_building(self, b: Building):
        if b.id not in self.adj:
            self.adj[b.id] = []
            self.buildings[b.id] = b

    def add_edge(self, id1: int, id2: int, weight: float = 1.0, bidirectional=True):
        self.add_building(self.buildings.get(id1, Building(id1,f"Building_{id1}","Unknown")))
        self.add_building(self.buildings.get(id2, Building(id2,f"Building_{id2}","Unknown")))
        self.adj[id1].append((id2, weight))
        if bidirectional:
            self.adj[id2].append((id1, weight))

    def bfs(self, start_id: int) -> List[int]:
        visited = set()
        q = [start_id]
        order = []
        visited.add(start_id)
        while q:
            cur = q.pop(0)
            order.append(cur)
            for (nbr, _) in self.adj.get(cur, []):
                if nbr not in visited:
                    visited.add(nbr); q.append(nbr)
        return order

    def dfs(self, start_id: int) -> List[int]:
        visited = set()
        order = []
        def _dfs(u):
            visited.add(u); order.append(u)
            for (v, _) in self.adj.get(u, []):
                if v not in visited:
                    _dfs(v)
        _dfs(start_id)
        return order

class GraphAdjMatrix:
    def __init__(self):
        self.index_map: Dict[int,int] = {}
        self.rev_map: Dict[int,int] = {}
        self.matrix: List[List[float]] = []

    def add_nodes(self, ids: List[int]):
        for idd in ids:
            if idd in self.index_map: continue
            idx = len(self.index_map)
            self.index_map[idd] = idx
            self.rev_map[idx] = idd
            # expand matrix
            for row in self.matrix:
                row.append(float('inf'))
            new_row = [float('inf')] * (len(self.index_map))
            self.matrix.append(new_row)
            self.matrix[idx][idx] = 0.0

    def add_edge(self, id1: int, id2: int, weight: float = 1.0, bidirectional=True):
        ids = [id1, id2]
        self.add_nodes(ids)
        i = self.index_map[id1]; j = self.index_map[id2]
        self.matrix[i][j] = weight
        if bidirectional:
            self.matrix[j][i] = weight

    def neighbors(self, idd: int) -> List[Tuple[int, float]]:
        if idd not in self.index_map: return []
        i = self.index_map[idd]
        res = []
        for j,w in enumerate(self.matrix[i]):
            if w != float('inf') and j != i:
                res.append((self.rev_map[j], w))
        return res

# -------------------------
# Dijkstra (on adj list)
# -------------------------
def dijkstra(adj_graph: GraphAdjList, source_id: int) -> Tuple[Dict[int,float], Dict[int,Optional[int]]]:
    dist = {nid: float('inf') for nid in adj_graph.adj}
    parent = {nid: None for nid in adj_graph.adj}
    dist[source_id] = 0
    heap = [(0, source_id)]
    while heap:
        d,u = heapq.heappop(heap)
        if d > dist[u]: continue
        for v,w in adj_graph.adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(heap, (nd, v))
    return dist, parent

def reconstruct_path(parent: Dict[int, Optional[int]], target: int) -> List[int]:
    path=[]
    cur=target
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path

# -------------------------
# Kruskal (MST)
# -------------------------
class DSU:
    def __init__(self):
        self.parent={}
        self.rank={}

    def make(self, x):
        self.parent[x]=x; self.rank[x]=0

    def find(self,x):
        if self.parent[x]!=x:
            self.parent[x]=self.find(self.parent[x])
        return self.parent[x]

    def union(self,a,b):
        ra, rb = self.find(a), self.find(b)
        if ra==rb: return False
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra]=rb
        else:
            self.parent[rb]=ra
            if self.rank[ra]==self.rank[rb]:
                self.rank[ra]+=1
        return True

def kruskal(adj_graph: GraphAdjList) -> Tuple[List[Tuple[int,int,float]], float]:
    edges=[]
    for u, nbrs in adj_graph.adj.items():
        for v,w in nbrs:
            if u < v:
                edges.append((u,v,w))
    edges.sort(key=lambda x: x[2])
    dsu=DSU()
    for node in adj_graph.adj.keys():
        dsu.make(node)
    mst=[]
    total=0.0
    for u,v,w in edges:
        if dsu.union(u,v):
            mst.append((u,v,w)); total += w
    return mst, total

# -------------------------
# Expression Tree (postfix -> tree -> evaluate)
# -------------------------
class ExprNode:
    def __init__(self, val):
        self.val = val
        self.left: Optional['ExprNode'] = None
        self.right: Optional['ExprNode'] = None

def build_expr_tree_from_postfix(tokens: List[str]) -> Optional[ExprNode]:
    stack=[]
    operators = set(['+','-','*','/','^'])
    for tok in tokens:
        if tok not in operators:
            stack.append(ExprNode(tok))
        else:
            node = ExprNode(tok)
            right = stack.pop(); left = stack.pop()
            node.right = right; node.left = left
            stack.append(node)
    return stack[0] if stack else None

def eval_expr_tree(node: ExprNode, variables: Dict[str,float]={}) -> float:
    if not node: return 0.0
    if node.val not in ['+','-','*','/','^']:
        try:
            return float(node.val)
        except:
            return variables.get(node.val, 0.0)
    L = eval_expr_tree(node.left, variables)
    R = eval_expr_tree(node.right, variables)
    if node.val == '+': return L+R
    if node.val == '-': return L-R
    if node.val == '*': return L*R
    if node.val == '/': return L/R
    if node.val == '^': return L**R
    raise ValueError("Unknown operator")

# -------------------------
# Demo and sample usage
# -------------------------
def demo():
    # Sample building data
    sample = [
        Building(10, "Admin Block", "Central"),
        Building(20, "Library", "North Wing"),
        Building(5, "CSE Dept", "Block C"),
        Building(15, "ECE Dept", "Block B"),
        Building(25, "Cafeteria", "East Wing"),
        Building(2, "Gym", "South"),
        Building(8, "Auditorium", "Main Hall"),
    ]

    print("\n=== BST Demo ===")
    bst = BST()
    for b in sample:
        bst.insert(b)
    print("BST Inorder:", [str(x) for x in bst.inorder()])
    print("BST Preorder:", [str(x) for x in bst.preorder()])
    print("BST Postorder:", [str(x) for x in bst.postorder()])
    print("BST Height:", bst.height())

    print("\n=== AVL Demo with rotation logging ===")
    avl = AVLTree()
    # choose an insertion order that will trigger rotations (e.g., create unbalanced inserts)
    avl_inserts = [Building(30,"B30","x"), Building(20,"B20","x"), Building(10,"B10","x"),  # will cause rotation
                   Building(40,"B40","x"), Building(50,"B50","x"), Building(25,"B25","x")]
    for b in avl_inserts:
        avl.insert(b)
    print("AVL Inorder:", [str(x) for x in avl.inorder()])
    print("AVL Height:", avl.height())

    # Compare heights using same data
    bst2 = BST()
    for b in avl_inserts:
        bst2.insert(b)
    print("BST Height with same AVL data:", bst2.height())

    print("\n=== Graph Demo (Adj List & Matrix), BFS & DFS ===")
    g = GraphAdjList()
    # add nodes explicitly so building names are preserved
    for b in sample:
        g.add_building(b)
    # add edges (bidirectional) with weights (distances)
    edges = [
        (10,20,5.0),(10,5,3.0),(5,8,2.0),(20,25,4.0),(8,2,6.0),(25,50,7.0) # include an external node 50 to show isolated handling
    ]
    for u,v,w in edges:
        g.add_edge(u,v,w)

    start = 10
    print(f"BFS from {start}:", g.bfs(start))
    print(f"DFS from {start}:", g.dfs(start))

    # adjacency matrix representation
    mat = GraphAdjMatrix()
    nodes_list = [b.id for b in sample] + [50]
    mat.add_nodes(nodes_list)
    for u,v,w in edges:
        mat.add_edge(u,v,w)
    print("Adjacency Matrix neighbors for 10:", mat.neighbors(10))

    print("\n=== Dijkstra Demo ===")
    dist, parent = dijkstra(g, 10)
    for nid in sorted(dist.keys()):
        print(f"Distance 10 -> {nid}: {dist[nid]}")
    target = 8
    if dist.get(target, float('inf')) < float('inf'):
        print("Path 10 -> 8:", reconstruct_path(parent, target))
    else:
        print("No path 10 -> 8")

    print("\n=== Kruskal (MST) Demo ===")
    mst, total = kruskal(g)
    print("MST edges:", mst)
    print("Total MST cost:", total)

    print("\n=== Expression Tree Demo (Energy bill calculation) ===")
    # Example: energy bill formula: base + (units * rate) + surcharge
    # Suppose: base=50, units=350, rate=0.12, surcharge=20 -> expression: base + (units*rate) + surcharge
    # Postfix tokens: base units rate * + surcharge +
    tokens = ["base","units","rate","*","+","surcharge","+"]
    tree = build_expr_tree_from_postfix(tokens)
    vars_map = {"base":50, "units":350, "rate":0.12, "surcharge":20}
    val = eval_expr_tree(tree, vars_map)
    print("Energy bill computed:", val)

if __name__ == "__main__":
    demo()
