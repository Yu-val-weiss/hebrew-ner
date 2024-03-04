from typing import Set, Tuple
from utils import yap, ner
from utils.yap_graph import YapGraph
import pandas as pd
from fastapi import FastAPI
import uvicorn

app = FastAPI(debug=True)

def prune_lattices(lattice_df: pd.DataFrame, multi_df: pd.DataFrame, multi_label_delim='^', fallback=False):
    splitting = ner.make_multi_splitting_df(multi_df, multi_label_delim)
    valid_edges: Set[Tuple[int, int, int, int]] = set()
    for ((sent_id, tok_id), sub_lattice),(_, split) in zip(lattice_df.groupby(['SENTNUM', 'TOKEN']), splitting.groupby(['SentNum', 'WordIndex'])):
        # tok_id in lattice starts from 1, so may need to offset
        g = YapGraph.from_df(sub_lattice)
        source, target = sub_lattice['FROM'].iat[0], sub_lattice['TO'].iat[-1]
        path_len = split['Splitting'].iat[0] + 1
        paths = list(g.get_all_paths(source, target, limit=path_len))
        pruned_paths = [p for p in paths if len(p) == path_len]
        if fallback and len(pruned_paths) == 0:
            pruned_paths = [p for p in paths if abs(len(p) - path_len) <= 1]
        if len(pruned_paths) > 0:
            paths = pruned_paths
        for p in paths:
            for f, t in zip(p[:-1], p[1:]):
                valid_edges.add((sent_id, tok_id, f, t))
                
    cols_to_filter = ['SENTNUM', 'TOKEN', 'FROM', 'TO']
    return lattice_df[lattice_df[cols_to_filter].apply(tuple, axis=1).isin(valid_edges)].reset_index(drop=True)
    

@app.get('/')
def home():
    return 200


if __name__ == '__main__':
    uvicorn.run(app, port=5000, reload=True)