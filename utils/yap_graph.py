# -*- coding: utf-8 -*-
# @Author: Yuval Weiss
import copy
import math
from typing import Dict, List, NamedTuple, Union

import pandas as pd

class Edge(NamedTuple):
    From: int
    To: int

class YapGraph:
    _graph: Dict[int, List[int]]
    def __init__(self, adj:Union[Dict[int, List[int]], None] = None) -> None:
        if adj:
            self._graph = copy.deepcopy(adj)
        else:
            self._graph = {}
        
    @classmethod
    def from_df(cls, df: pd.DataFrame, from_col = 'FROM', to_col = 'TO'):
        '''
        Creates a YapGraph from a pandas DataFrame containing a lattice
        '''
        adj: Dict[int, List[int]] = {}
        for f, t in df[[from_col, to_col]].itertuples(index=False):
            if f not in adj:
                adj[f] = []
            if t not in adj:
                adj[t] = []
            if t not in adj[f]:
                adj[f].append(t)
        return cls(adj)     
    
    def __str__(self) -> str:
        rep = [f"\t{x: >4}:\t{self._graph.get(x, [])}" for x in self._graph]
        return "Graph(\n" + "\n".join(rep) + "\n)" 
        
    
    def nodes(self):
        return set(self._graph.keys())
    
    def all_edges(self):
        for k in self._graph:
            for j in self._graph[k]:
                yield Edge(k, j) 
    
    def adj(self, node: int):
        return iter(self._graph.get(node, []))
    
    def __getitem__(self, x):
        return self.adj(x)
    
    def get_all_paths(self, start: int, end: int, limit=math.inf):
        '''
        Returns an iterator over all paths, each of which is a list of ints, using the iterator version of DFS.
        Source: Wikipedia
        '''
        path = [start]
        s = [self.adj(start)]
        while s:
            peek = s[-1]
            w = next(peek, None)
            if w == None: # backtrack
                s.pop()
                path.pop()
            elif len(path) < limit:
                # implicitly backtracks after this due to iterator
                if w == end:
                    yield path + [end]
                elif w not in path:
                    path.append(w)
                    s.append(self.adj(w))
            else: # len(path) == limit
                if w == end or end in peek: 
                    # note `in` consumes the iterator, but this is fine, since we are at max depth, so want to check if reachable from here 
                    yield path + [end]
                # backtrack
                s.pop()
                path.pop()