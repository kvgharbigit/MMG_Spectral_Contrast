import os

def combine_py_and_yaml_files_recursively(output_file="combined_output.py"):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    collected_files = []

    # Recursively collect .py, .yaml, .yml files
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(('.py', '.yaml', '.yml')):
                full_path = os.path.join(dirpath, filename)
                collected_files.append(full_path)

    # Sort for consistency
    collected_files.sort()

    with open(os.path.join(root_dir, output_file), 'w', encoding='utf-8') as outfile:
        for filepath in collected_files:
            # Skip output file itself
            if os.path.abspath(filepath) == os.path.join(root_dir, output_file):
                continue

            relative_path = os.path.relpath(filepath, root_dir)
            outfile.write(f"# ===== File: {relative_path} =====\n\n")

            try:
                with open(filepath, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())
                    outfile.write("\n\n")
            except Exception as e:
                print(f"⚠️ Error reading {filepath}: {e}")

    print(f"✅ Combined {len(collected_files)} files into: {output_file}")

if __name__ == "__main__":
    combine_py_and_yaml_files_recursively()
