import torch
import hashlib
from torch_geometric.data import HeteroData
from typing import List, Dict, Any, Tuple, Set

class ExprGraphProcessor:
    def __init__(self, symbol_to_id: Dict[str, int], max_nodes: int = 512):
        self.symbol_to_id = symbol_to_id
        self.max_nodes = max_nodes
        self.kinds = [
            'app', 'bvar', 'const', 'forall', 'fvar', 
            'lam', 'let', 'lit', 'mvar', 'proj', 'sort'
        ]
        self.kind_to_idx = {k: i for i, k in enumerate(self.kinds)}

    def _get_real_index(self, nodes: List[Dict], idx: int) -> int:
        """Recursive function to skip mdata nodes."""
        if nodes[idx]['kind'] == 'mdata':
            return self._get_real_index(nodes, nodes[idx]['expr'])
        return idx

    def process_json_graph(self, json_nodes: List[Dict]) -> HeteroData:
        # 1. Flatten mdata
        # Create a mapping from old indices to non-mdata indices
        real_indices = {}
        valid_indices = []
        for i, node in enumerate(json_nodes):
            if node['kind'] != 'mdata':
                valid_indices.append(i)
                real_indices[i] = i
            else:
                real_indices[i] = self._get_real_index(json_nodes, i)

        # 2. Pruning
        # We need to keep Root (usually the last node), Const nodes, and their ancestors.
        # Then fill with others based on distance from root.
        
        # Build adjacency for pruning analysis (children to parents)
        # Note: In Lean JSON, parent points to children indices.
        parents = {i: [] for i in range(len(json_nodes))}
        for i in valid_indices:
            node = json_nodes[i]
            for child_key in ['fn', 'arg', 'type', 'body', 'value', 'expr']:
                if child_key in node:
                    child_old = node[child_key]
                    if isinstance(child_old, int):
                        child_real = real_indices[child_old]
                        parents[child_real].append(i)

        root_idx = valid_indices[-1] if valid_indices else None
        
        # Identify critical nodes (Consts and their ancestors)
        critical_nodes = set()
        if root_idx is not None:
            critical_nodes.add(root_idx)
            
        for i in valid_indices:
            if json_nodes[i]['kind'] == 'const':
                curr = i
                while curr is not None and curr not in critical_nodes:
                    critical_nodes.add(curr)
                    # For simplicity, just trace one parent (usually trees)
                    curr = parents[curr][0] if parents[curr] else None

        # If too many nodes, prune by BFS from root
        final_kept_nodes = set(critical_nodes)
        if len(final_kept_nodes) > self.max_nodes:
            # Too many critical nodes? Prune by distance from root
            # (Unlikely but for safety)
            sorted_critical = sorted(list(critical_nodes), key=lambda x: self._get_depth(json_nodes, real_indices, root_idx, x))
            final_kept_nodes = set(sorted_critical[:self.max_nodes])
        elif len(valid_indices) > self.max_nodes:
            # Fill with BFS
            queue = [root_idx]
            visited = {root_idx}
            while queue and len(final_kept_nodes) < self.max_nodes:
                curr = queue.pop(0)
                # Find children
                node = json_nodes[curr]
                for child_key in ['fn', 'arg', 'type', 'body', 'value', 'expr']:
                    if child_key in node:
                        c_real = real_indices[node[child_key]]
                        if c_real not in visited:
                            visited.add(c_real)
                            final_kept_nodes.add(c_real)
                            queue.append(c_real)
        else:
            final_kept_nodes = set(valid_indices)

        # 3. Re-indexing
        # final_kept_nodes should be sorted to preserve some order (e.g., topological)
        sorted_kept = sorted(list(final_kept_nodes))
        old_to_new = {old: new for new, old in enumerate(sorted_kept)}

        # 4. Feature Construction
        expr_features = []
        depths = self._compute_depths(json_nodes, real_indices, root_idx, sorted_kept)
        
        for i, old_idx in enumerate(sorted_kept):
            node = json_nodes[old_idx]
            feat = self._construct_expr_features(node, depths.get(old_idx, 0))
            expr_features.append(feat)

        # 5. Build HeteroData
        data = HeteroData()
        data['expr'].x = torch.stack(expr_features)
        
        # Virtual node
        data['virtual'].x = torch.zeros((1, 1)) # Placeholder for batching

        # Symbols
        symbols_found = []
        symbol_to_new = {}
        
        # Edge storage
        edges = {
            'has_fn': ([], []), 'has_arg': ([], []),
            'has_type': ([], []), 'has_body': ([], []),
            'has_value': ([], []), 'has_expr': ([], []),
            'is_instance_of': ([], [])
        }

        for new_idx, old_idx in enumerate(sorted_kept):
            node = json_nodes[old_idx]
            
            # AST Edges
            self._add_edge(node, 'fn', new_idx, real_indices, old_to_new, edges['has_fn'])
            self._add_edge(node, 'arg', new_idx, real_indices, old_to_new, edges['has_arg'])
            self._add_edge(node, 'type', new_idx, real_indices, old_to_new, edges['has_type'])
            self._add_edge(node, 'body', new_idx, real_indices, old_to_new, edges['has_body'])
            self._add_edge(node, 'value', new_idx, real_indices, old_to_new, edges['has_value'])
            self._add_edge(node, 'expr', new_idx, real_indices, old_to_new, edges['has_expr'])
            
            # Const to Symbol
            if node['kind'] == 'const':
                sym_name = node['name']
                sym_id = self.symbol_to_id.get(sym_name, self.symbol_to_id.get("<UNK>", 0))
                if sym_id not in symbol_to_new:
                    symbol_to_new[sym_id] = len(symbols_found)
                    symbols_found.append(sym_id)
                
                edges['is_instance_of'][0].append(new_idx)
                edges['is_instance_of'][1].append(symbol_to_new[sym_id])

        # Assign AST edges and reverse
        for e_name, (src, dst) in edges.items():
            if not src: continue
            if e_name == 'is_instance_of':
                data['expr', 'is_instance_of', 'symbol'].edge_index = torch.tensor([src, dst], dtype=torch.long)
                data['symbol', 'rev_is_instance_of', 'expr'].edge_index = torch.tensor([dst, src], dtype=torch.long)
            else:
                data['expr', e_name, 'expr'].edge_index = torch.tensor([src, dst], dtype=torch.long)
                data['expr', f'rev_{e_name}', 'expr'].edge_index = torch.tensor([dst, src], dtype=torch.long)

        # Symbols Features
        if symbols_found:
            data['symbol'].x = torch.tensor(symbols_found, dtype=torch.long)
        else:
            data['symbol'].x = torch.empty((0,), dtype=torch.long)

        # Global/Virtual edges
        num_expr = len(sorted_kept)
        num_sym = len(symbols_found)
        
        if num_expr > 0:
            data['expr', 'to_virtual', 'virtual'].edge_index = torch.stack([
                torch.arange(num_expr), torch.zeros(num_expr, dtype=torch.long)
            ])
            data['virtual', 'from_virtual', 'expr'].edge_index = torch.stack([
                torch.zeros(num_expr, dtype=torch.long), torch.arange(num_expr)
            ])
            
        if num_sym > 0:
            data['symbol', 'sym_to_virtual', 'virtual'].edge_index = torch.stack([
                torch.arange(num_sym), torch.zeros(num_sym, dtype=torch.long)
            ])
            data['virtual', 'sym_from_virtual', 'symbol'].edge_index = torch.stack([
                torch.zeros(num_sym, dtype=torch.long), torch.arange(num_sym)
            ])

        return data

    def _add_edge(self, node, key, src_new, real_indices, old_to_new, edge_list):
        if key in node:
            child_old = node[key]
            if isinstance(child_old, int):
                child_real = real_indices[child_old]
                child_new = old_to_new.get(child_real) # This might be None if pruned
                if child_new is not None:
                    edge_list[0].append(src_new)
                    edge_list[1].append(child_new)

    def _get_depth(self, nodes, real_indices, root, target):
        # Dummy depth implementation for sorting, compute_depths is more accurate
        return 0 

    def _compute_depths(self, nodes, real_indices, root_idx, kept_indices) -> Dict[int, int]:
        if root_idx is None: return {}
        depths = {root_idx: 0}
        queue = [root_idx]
        visited = {root_idx}
        
        while queue:
            curr = queue.pop(0)
            node = nodes[curr]
            for child_key in ['fn', 'arg', 'type', 'body', 'value', 'expr']:
                if child_key in node:
                    child_old = node[child_key]
                    if isinstance(child_old, int):
                        c_real = real_indices[child_old]
                        if c_real not in visited:
                            visited.add(c_real)
                            depths[c_real] = depths[curr] + 1
                            queue.append(c_real)
        return depths

    def _construct_expr_features(self, node: Dict, depth: int) -> torch.Tensor:
        # 47-dim feature vector
        features = torch.zeros(47)
        
        # 1. One-hot kind (11 dims)
        kind_idx = self.kind_to_idx.get(node['kind'], 10) # Default to 'sort' if unknown
        features[kind_idx] = 1.0
        
        # 2. Index/Proj (1 dim)
        idx_val = 0
        if node['kind'] == 'bvar':
            idx_val = node.get('index', 0)
        elif node['kind'] == 'proj':
            idx_val = node.get('index', 0)
        features[11] = min(idx_val, 127) / 127.0
        
        # 3. Depth (1 dim)
        features[12] = min(depth, 255) / 255.0
        
        # 4. Hash (32 dims)
        # Identifiers are in: name (const, fvar, mvar, forall, lam), val (lit)
        ident = ""
        if 'name' in node: ident = node['name']
        elif 'val' in node: ident = str(node['val'])
        elif 'level' in node and node['kind'] == 'sort': ident = str(node['level'])
        
        if ident:
            h = hashlib.md5(ident.encode()).digest()
            # Take first 4 bytes as a 32-bit int
            hash_int = int.from_bytes(h[:4], 'big')
            for i in range(32):
                if (hash_int >> i) & 1:
                    features[13 + i] = 1.0
                    
        # 5. Is Leaf (1 dim)
        is_leaf = 1.0
        for key in ['fn', 'arg', 'type', 'body', 'value', 'expr']:
            if key in node:
                is_leaf = 0.0
                break
        features[45] = is_leaf
        
        # 6. Padding (1 dim) - features[46] is already 0.0
        
        return features
