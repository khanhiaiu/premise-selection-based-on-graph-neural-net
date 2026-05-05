import os
import sys
import json
import torch
from data.processor import ExprGraphProcessor

def convert_tree_to_dag(tree):
    """
    Converts a JSON state tree to a flattened DAG using hash-consing.
    Based on ntp-toolkit/convert_tree_to_graph.py
    """
    nodes = []
    memo = {}
    
    def process(node):
        if not isinstance(node, dict): return None
        kind = node.get("kind")
        if not kind: return None

        # 1. BÓC VỎ MDATA
        if kind == "mdata":
            return process(node.get("expr"))

        # --- TÍNH TOÁN KEY CHO HASH-CONSING ---
        if kind == "app":
            fn_id = process(node.get("fn"))
            arg_id = process(node.get("arg"))
            if fn_id is None or arg_id is None: return None
            key = ("app", fn_id, arg_id)
            
        elif kind in ["forall", "lam"]:
            type_id = process(node.get("type"))
            body_id = process(node.get("body"))
            bi = node.get("bi", "default")
            key = (kind, node.get("name"), type_id, body_id, bi)
            
        elif kind == "let":
            type_id = process(node.get("type"))
            val_id = process(node.get("value"))
            body_id = process(node.get("body"))
            key = ("let", node.get("name"), type_id, val_id, body_id)
            
        elif kind == "const":
            levels = tuple(node.get("levels", []))
            key = ("const", node.get("name"), levels)
            
        elif kind == "sort":
            key = ("sort", node.get("level"))
            
        elif kind == "bvar": key = ("bvar", node.get("index"))
        elif kind == "fvar": key = ("fvar", node.get("id"))
        elif kind == "mvar": key = ("mvar", node.get("id"))
        elif kind == "lit": key = ("lit", node.get("val"))
        elif kind == "proj":
            expr_id = process(node.get("expr"))
            key = ("proj", expr_id, node.get("index"), node.get("type"))
        else:
            return None 

        if key in memo: return memo[key]
        
        new_id = len(nodes)
        dag_node = {"kind": kind}
        
        if kind == "app":
            dag_node["fn"] = key[1]; dag_node["arg"] = key[2]
        elif kind in ["forall", "lam"]:
            dag_node["name"] = key[1]; dag_node["type"] = key[2]; dag_node["body"] = key[3]; dag_node["bi"] = key[4]
        elif kind == "let":
            dag_node["name"] = key[1]; dag_node["type"] = key[2]; dag_node["value"] = key[3]; dag_node["body"] = key[4]
        elif kind == "const":
            dag_node["name"] = key[1]; dag_node["levels"] = list(key[2])
        elif kind == "sort":
            dag_node["level"] = key[1]
        elif kind == "bvar": dag_node["index"] = key[1]
        elif kind in ["fvar", "mvar"]: dag_node["id"] = key[1]
        elif kind == "lit": dag_node["val"] = key[1]
        elif kind == "proj":
            dag_node["expr"] = key[1]; dag_node["index"] = key[2]; dag_node["type"] = key[3]
            
        nodes.append(dag_node)
        memo[key] = new_id
        return new_id

    root_id = process(tree)
    return nodes

class StateGraphBuilder:
    def __init__(self, vocab_path="datatrain/symbol_vocab.json"):
        from data.symbol_manager import SymbolManager
        self.symbol_manager = SymbolManager(vocab_path=vocab_path)
        self.processor = ExprGraphProcessor(symbol_to_id=self.symbol_manager.symbol_to_id, max_nodes=512)
        
    def build_graph_from_tree(self, tree_json):
        """
        Parses a state tree JSON into PyG HeteroData.
        """
        if isinstance(tree_json, str):
            tree_json = json.loads(tree_json)
            
        # 1. Parse tree JSON to DAG
        dag_nodes = convert_tree_to_dag(tree_json)
        if dag_nodes is None or len(dag_nodes) == 0:
            raise ValueError("Failed to convert tree to DAG or graph is empty.")
            
        # 2. Process DAG to PyG HeteroData
        hetero_data = self.processor.process_json_graph(dag_nodes)
        
        return hetero_data, dag_nodes
