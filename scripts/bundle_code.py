import os
import argparse
from pathlib import Path

def bundle_code(input_paths, output_file, extensions=None, exclude_dirs=None):
    if extensions is None:
        extensions = ['.py', '.json', '.sh', '.txt', '.md', '.lean']
    if exclude_dirs is None:
        exclude_dirs = ['__pycache__', '.git', '.venv', 'build', 'dist', 'node_modules']

    output_path = Path(output_file)
    
    with open(output_path, 'w', encoding='utf-8') as outfile:
        outfile.write(f"# Code Context Bundle\n")
        outfile.write(f"# Generated on: {os.popen('date').read().strip()}\n\n")
        
        for input_path in input_paths:
            path = Path(input_path)
            if not path.exists():
                print(f"Warning: Path {input_path} does not exist. Skipping.")
                continue
                
            if path.is_file():
                process_file(path, outfile)
            elif path.is_dir():
                for root, dirs, files in os.walk(path):
                    # Filter excluded directories in-place
                    dirs[:] = [d for d in dirs if d not in exclude_dirs]
                    
                    for file in files:
                        file_path = Path(root) / file
                        if any(file_path.suffix == ext for ext in extensions):
                            process_file(file_path, outfile)

def process_file(file_path, outfile):
    try:
        print(f"Processing: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        relative_path = os.path.relpath(file_path, os.getcwd())
        outfile.write(f"### File: {relative_path}\n")
        
        # Determine language for markdown block
        lang = ""
        if file_path.suffix == ".py":
            lang = "python"
        elif file_path.suffix == ".json":
            lang = "json"
        elif file_path.suffix == ".sh":
            lang = "bash"
        elif file_path.suffix == ".lean":
            lang = "lean"
            
        outfile.write(f"```{lang}\n")
        outfile.write(content)
        if not content.endswith('\n'):
            outfile.write('\n')
        outfile.write(f"```\n\n")
        outfile.write("-" * 80 + "\n\n")
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bundle code files into a single text file for LLM prompts.")
    parser.add_argument("paths", nargs="+", help="Files or directories to include.")
    parser.add_argument("-o", "--output", default="code_context.txt", help="Output file path (default: code_context.txt)")
    parser.add_argument("-e", "--extensions", nargs="+", help="File extensions to include (e.g., .py .json)")
    
    args = parser.parse_args()
    
    bundle_code(args.paths, args.output, extensions=args.extensions)
    print(f"\nDone! Context saved to {args.output}")
