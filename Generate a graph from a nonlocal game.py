import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
from itertools import product


def predicate_asym(a,b,x,y, predicate):
    if x == y:
        value = 1 if a == b else 0
    elif x > y:
        value = 1
    elif x<y:
        value = predicate(a,b,x,y) * predicate(b,a,y,x)
    
    return value


def predicate_inv(I, O, predicate, predicate_mermin_peres):
    predicate_inv = []
    for a,b,x,y in itertools.product(O,O,I,I):
        if predicate(a,b,x,y, predicate_mermin_peres) == 0:
            predicate_inv.append([a,b,x,y])
    return predicate_inv


def generate_variable_table(function, predicate):
    data = []
    
    for a, b, x, y in itertools.product(O,O,I,I):
        #if x != y and x < y:
        result = predicate(a, b, x, y)
        data.append({"a": a, "b": b, "x": x, "y": y, "λ(a,b,x,y)": result})
            

    df = pd.DataFrame(data)
    return df



def predicate_mermin_peres(a,b,x,y):
    # a,b = 1,2,3,4; x,y = 1,2,3,4,5,6
    # Mapping from question to which bits are set by that answer
    assignment_rules = {
        1: [0, 1, 2],  # x1 + x2 + x3 = 0
        2: [3, 4, 5],  # x4 + x5 + x6 = 0
        3: [6, 7, 8],  # x7 + x8 + x9 = 0
        4: [0, 3, 6],  # x1 + x4 + x7 = 1
        5: [1, 4, 7],  # x2 + x5 + x8 = 1
        6: [2, 5, 8],  # x3 + x6 + x9 = 1
    }

    # Answer encodings for questions 1 to 5 (right-hand side = 0)
    encoding_0 = {
        1: [0, 0, 0],
        2: [0, 1, 1],
        3: [1, 0, 1],
        4: [1, 1, 0],
    }

    # Answer encodings for question 6 (right-hand side = 1)
    encoding_1 = {
        1: [1, 1, 1],
        2: [1, 0, 0],
        3: [0, 1, 0],
        4: [0, 0, 1],
    }
    
    alice = np.full(9,-1)
    bob = np.full(9,-1)
    
    for i in range(len(assignment_rules[x])):
        if x == 6:
            alice[assignment_rules[x][i]] = encoding_1[a][i]
        else:
            alice[assignment_rules[x][i]] = encoding_0[a][i]
            
            
    for j in range(len(assignment_rules[y])):
        if y == 6:
            bob[assignment_rules[y][j]] = encoding_1[b][j]
        else:
            bob[assignment_rules[y][j]] = encoding_0[b][j]
    
    for i in range(len(alice)):
        if alice[i] < 0 or bob[i] < 0:
            pass
        elif alice[i] != bob[i]:
            return 0
        else:
            pass
    return 1




def v_name(i: int, j: int, alpha: int, x: int) -> str:
    return f"v({i},{j},{alpha},{x})"

def t_name(i: int, alpha: int, x: int) -> str:
    return f"t({i},{alpha},{x})"

def q_name(i: int, j: int, a: int, b: int, x: int, y: int) -> str:
    return f"q({i},{j},{a},{b},{x},{y})"

def v_hat(a: int, x: int, k: int) -> str:
    if a == 1:
        return v_name(1, 1, 1, x)
    if 2 <= a <= k - 1:
        return v_name(2, 1, a - 1, x)
    if a == k:
        return v_name(2, 2, k - 2, x)
    raise ValueError("a out of range in v̂")


def _merge_or_rename(G: nx.Graph, src: str, dst: str) -> None:
    if src == dst or src not in G:
        return
    if dst not in G:
        nx.relabel_nodes(G, {src: dst}, copy=False)
        return
    # merge
    for nbr in list(G.neighbors(src)):
        if nbr != dst:
            G.add_edge(dst, nbr)
    G.remove_node(src)

def generate_G_lambda(I, O, predicate_inv):
    """
    Construct and return the graph G_λ.

    Parameters
    ----------
    I : iterable[int]    # inputs
    O : iterable[int]    # outputs
    predicate_inv : iterable or dict
        Quadruples (a,b,x,y) with λ(x,y,a,b)=0.  Lists OK.

    Returns
    -------
    nx.Graph
    """
    I = tuple(I)
    O = tuple(O)
    n, k = len(I), len(O)
    if n < 2 or k < 3:
        raise ValueError("Need |I| ≥ 2 and |O| ≥ 3.")
    
    # normalise λ⁻¹({0}) to a *set of 4-tuples*
    if isinstance(predicate_inv, dict):
        lambda_zeros = {tuple(q) for q, val in predicate_inv.items() if val == 0}
    else:
        lambda_zeros = {tuple(q) for q in predicate_inv}

    #  Initialise base triangle  {A,B,C}
    G = nx.Graph()
    G.add_nodes_from(["A", "B", "C"])
    G.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])

    #  Copies R_{α,x}  (α = 1 … k−2,  x = 1 … n)
    for alpha in range(1, k - 1):
        for x in range(1, n + 1):

            # -- add vertices (with identification v(1,2)=B) and K3×K3 edges
            for (i1, j1), (i2, j2) in product(product(range(1, 4), repeat=2),
                                              repeat=2):
                if (i1, j1) == (i2, j2):
                    continue
                if (i1 == i2) ^ (j1 == j2):         # XOR adjacency rule
                    u = "B" if (i1, j1) == (1, 2) else v_name(i1, j1, alpha, x)
                    v = "B" if (i2, j2) == (1, 2) else v_name(i2, j2, alpha, x)
                    G.add_edge(u, v)

            # extra adjacencies (4.4) and (4.5)
            G.add_edge("A", v_name(3, 3, alpha, x))
            G.add_edge("C", v_name(2, 1, alpha, x))

            # identification (4.3) across successive α
            if alpha <= k - 3:
                _merge_or_rename(
                    G,
                    v_name(3, 2, alpha, x),
                    v_name(1, 1, alpha + 1, x),
                )

            # triangular prism T_{α,x}
            tri1 = [v_name(1, 1, alpha, x), "B", v_name(1, 3, alpha, x)]
            tri2 = [t_name(1, alpha, x), "A", t_name(2, alpha, x)]
            for tri in (tri1, tri2):
                G.add_edges_from([(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])])
            G.add_edge(v_name(1, 1, alpha, x), t_name(1, alpha, x))
            G.add_edge(v_name(1, 3, alpha, x), t_name(2, alpha, x))
            # edge B–A already exists in base triangle

    #  Partition λ⁻¹({0}) into E_λ and F_λ
    E_lambda, F_lambda = set(), set()
    for a, b, x, y in lambda_zeros:
        if x == y:
            continue
        if (a, b) in {(1, 1), (1, k), (k, 1), (k, k)}:
            E_lambda.add((a, b, x, y))
        elif 2 <= a <= k - 1 and 2 <= b <= k - 1:
            F_lambda.add((a, b, x, y))

    #  Helper to add gadget Q_{a,b,x,y}  (Case 2
    def add_Q(a: int, b: int, x: int, y: int) -> None:
        is_E = (a, b, x, y) in E_lambda      # else it belongs to F_λ

        def q_alias(i: int, j: int) -> str:
            if (i, j) == (1, 1):
                return v_hat(a, x, k)
            if (i, j) == (2, 2):
                return v_hat(b, y, k)
            if (i, j) == (1, 2):
                return "B" if is_E else "C"
            return q_name(i, j, a, b, x, y)

        # build K3×K3 with XOR rule
        for (i1, j1), (i2, j2) in product(product(range(1, 4), repeat=2),
                                          repeat=2):
            if (i1, j1) == (i2, j2):
                continue
            if (i1 == i2) ^ (j1 == j2):
                G.add_edge(q_alias(i1, j1), q_alias(i2, j2))

        G.add_edge("A", q_name(3, 3, a, b, x, y))

    #  Add Case 1 edges or Case 2 gadgets
    for a, b, x, y in lambda_zeros:
        if x == y:
            continue
        if (a, b, x, y) in E_lambda or (a, b, x, y) in F_lambda:
            add_Q(a, b, x, y)
        else:
            G.add_edge(v_hat(a, x, k), v_hat(b, y, k))

    return G



def draw_graph(G):
    plt.figure(figsize=(10, 10))
    nx.draw_networkx(G, with_labels=False, node_size=100, edge_color='gray')
    plt.show()

I = [1,2,3,4,5,6]
O = [1,2,3,4]
    
# Generate the table
variable_table = generate_variable_table(predicate_asym, predicate_mermin_peres)
    
G_lambda = generate_G_lambda(I, O, predicate_inv(I,O,predicate_asym, predicate_mermin_peres))


#draw_graph(G_lambda)
#print(G_lambda)


plt.figure(figsize=(10, 10))

pos = nx.spring_layout(G_lambda, seed=7123)        

nx.draw_networkx_nodes(G_lambda, pos, node_size=200, edgecolors="black")
nx.draw_networkx_edges(G_lambda, pos, width=1)


plt.axis("off")
plt.tight_layout()
plt.show()
