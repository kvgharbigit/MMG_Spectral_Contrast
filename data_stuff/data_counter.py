import os
import glob


def count_h5_files_in_top_level_dirs(parent_dir, output_file=None, sample_depth=2):
    """
    Count the number of H5 files in each top-level directory within the parent directory,
    draw a sample directory structure, and optionally save the results to a text file.

    Args:
        parent_dir (str): Path to the parent directory
        output_file (str, optional): Path to save the results as a text file
        sample_depth (int): How many levels deep to show in the directory structure sample

    Returns:
        dict: Dictionary with top-level directory names as keys and counts of H5 files as values
    """
    # Check if parent directory exists
    if not os.path.isdir(parent_dir):
        raise ValueError(f"Parent directory '{parent_dir}' does not exist")

    # Get all top-level directories
    top_level_dirs = [d for d in os.listdir(parent_dir)
                      if os.path.isdir(os.path.join(parent_dir, d))]

    # Initialize result dictionary
    h5_counts = {}
    # Store directory structure samples
    dir_structures = {}

    # Count H5 files in each top-level directory and sample directory structure
    for dir_name in top_level_dirs:
        dir_path = os.path.join(parent_dir, dir_name)

        # Use glob to recursively find all .h5 files
        h5_files = glob.glob(os.path.join(dir_path, "**", "*.h5"), recursive=True)

        # Store the count
        h5_counts[dir_name] = len(h5_files)

        # Get directory structure sample
        dir_structures[dir_name] = get_directory_structure(dir_path, sample_depth)

    # Save results to a text file if specified
    if output_file:
        try:
            with open(output_file, 'w') as f:
                f.write(f"H5 File Counts in {parent_dir}:\n")
                f.write("-" * 50 + "\n")

                for dir_name, count in h5_counts.items():
                    f.write(f"{dir_name}: {count} H5 files\n")

                    # Add directory structure visualization
                    f.write("\nDirectory Structure Sample:\n")
                    structure_text = dir_structures[dir_name]
                    f.write(structure_text + "\n\n")
                    f.write("-" * 50 + "\n")

                # Add a total count at the end
                total_count = sum(h5_counts.values())
                f.write(f"Total: {total_count} H5 files\n")

            print(f"Results saved to {output_file}")
        except Exception as e:
            print(f"Error saving results to file: {e}")

    return h5_counts


def get_directory_structure(directory, max_depth=2, current_depth=0, max_items=5, prefix=""):
    """
    Generate a text representation of the directory structure up to max_depth levels.
    Limits the number of items shown at each level to avoid extremely long outputs.

    Args:
        directory (str): The directory to visualize
        max_depth (int): Maximum depth to traverse
        current_depth (int): Current depth in the traversal
        max_items (int): Maximum number of items to show at each level
        prefix (str): Prefix for the current line

    Returns:
        str: Text representation of the directory structure
    """
    if current_depth > max_depth:
        return ""

    result = []

    try:
        # Get all items in the directory
        items = os.listdir(directory)

        # Sort items so directories come first
        items.sort(key=lambda x: not os.path.isdir(os.path.join(directory, x)))

        # Limit the number of items shown
        if len(items) > max_items:
            displayed_items = items[:max_items]
            items_remaining = len(items) - max_items
        else:
            displayed_items = items
            items_remaining = 0

        for i, item in enumerate(displayed_items):
            item_path = os.path.join(directory, item)
            is_last = (i == len(displayed_items) - 1) and items_remaining == 0

            # Choose the appropriate connector
            if is_last:
                connector = "└── "
                next_prefix = prefix + "    "
            else:
                connector = "├── "
                next_prefix = prefix + "│   "

            # Add the current item to the result
            result.append(f"{prefix}{connector}{item}")

            # If it's a directory, recursively add its contents
            if os.path.isdir(item_path):
                result.append(get_directory_structure(
                    item_path,
                    max_depth,
                    current_depth + 1,
                    max_items,
                    next_prefix
                ))

        # Add a note if we didn't show all items
        if items_remaining > 0:
            if items_remaining == 1:
                result.append(f"{prefix}└── ... 1 more item not shown")
            else:
                result.append(f"{prefix}└── ... {items_remaining} more items not shown")

    except PermissionError:
        result.append(f"{prefix}└── [Permission denied]")
    except Exception as e:
        result.append(f"{prefix}└── [Error: {str(e)}]")

    return "\n".join(result)


# Example usage
if __name__ == "__main__":
    # Replace with your parent directory path
    parent_directory = "Z:/Projects/Ophthalmic neuroscience/Projects/Kayvan/data_500"

    # Specify output file path
    output_file = "h5_file_counts.txt"

    try:
        result = count_h5_files_in_top_level_dirs(parent_directory, output_file)

        print("H5 File Counts:")
        for dir_name, count in result.items():
            print(f"{dir_name}: {count} H5 files")

        print(f"Total: {sum(result.values())} H5 files")
        print(f"Detailed results including directory structures saved to {output_file}")

    except Exception as e:
        print(f"Error: {e}")