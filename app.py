import gradio as gr
import json
import traceback

from utils.graph_builder import StateGraphBuilder
from utils.retriever import LeanRetriever
from utils.lean_parser import parse_lean_to_graph

# Initialize models globally
builder = None
retriever = None

def init_models():
    global builder, retriever
    if builder is not None and retriever is not None:
        return
        
    print("Initializing models on CPU...")
    builder = StateGraphBuilder(vocab_path="datatrain/symbol_vocab.json")
    retriever = LeanRetriever(
        model_path="checkpoints/hgt_epoch_29_val_loss_1.858.pt",
        vocab_path="datatrain/symbol_vocab.json",
        embeddings_path="datatrain/symbol_embeddings.pt",
        precomputed_premises_path="datatrain/precomputed_50k/premises_dict.pt",
        device="cpu",
        max_premises=5
    )
    print("Models initialized successfully!")

def retrieve_premises(lean_code_str, top_k):
    global builder, retriever
    try:
        if builder is None or retriever is None:
            init_models()
            
        # Parse Lean Code
        if not lean_code_str.strip():
            return "Error: Empty input state."
        
        try:
            state_tree = parse_lean_to_graph(lean_code_str)
        except Exception as e:
            return f"Error: Lean compilation failed.\n{str(e)}"
            
        # Build graph
        try:
            hetero_data, dag_nodes = builder.build_graph_from_tree(state_tree)
            graph_stats = f"Graph built successfully! Nodes: {len(dag_nodes)}\n"
            
            # Count edges
            num_edges = 0
            for edge_type in hetero_data.edge_types:
                num_edges += hetero_data[edge_type].edge_index.size(1)
            graph_stats += f"Total Edges: {num_edges}"
        except Exception as e:
            return f"Error during graph construction:\n{traceback.format_exc()}"
            
        # Retrieve
        try:
            results = retriever.retrieve(hetero_data, top_k=int(top_k))
            output_str = f"=== GRAPH STATS ===\n{graph_stats}\n\n=== RETRIEVAL RESULTS (Top {top_k}) ===\n"
            for i, (pid, score) in enumerate(results):
                output_str += f"{i+1}. {pid} (Score: {score:.4f})\n"
            return output_str
        except Exception as e:
            return f"Error during retrieval:\n{traceback.format_exc()}"
            
    except Exception as e:
        return f"Unexpected error:\n{traceback.format_exc()}"

# Create Gradio interface
with gr.Blocks(title="Lean 4 Premise Selection GNN") as demo:
    gr.Markdown("# Lean 4 Premise Selection GNN Demo")
    gr.Markdown("Paste a raw Lean expression (e.g., `∀ n : Nat, n + 0 = n`) below to retrieve relevant premises.")
    
    with gr.Row():
        with gr.Column(scale=2):
            input_state = gr.Textbox(
                label="Raw Lean Expression", 
                lines=10, 
                placeholder='∀ n m : Nat, n + m = m + n'
            )
            top_k_slider = gr.Slider(minimum=1, maximum=50, value=5, step=1, label="Top K Premises")
            submit_btn = gr.Button("Parse Lean & Retrieve", variant="primary")
            
        with gr.Column(scale=2):
            output_text = gr.Textbox(label="Results", lines=20)
            
    submit_btn.click(
        fn=retrieve_premises,
        inputs=[input_state, top_k_slider],
        outputs=output_text
    )

if __name__ == "__main__":
    init_models()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
