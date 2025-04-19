import os
import glob


def count_h5_files_in_top_level_dirs(parent_dir, output_file=None):
    """
    Count the number of H5 files in each top-level directory within the parent directory
    and optionally save the results to a text file.

    Args:
        parent_dir (str): Path to the parent directory
        output_file (str, optional): Path to save the results as a text file

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

    # Count H5 files in each top-level directory
    for dir_name in top_level_dirs:
        dir_path = os.path.join(parent_dir, dir_name)

        # Use glob to recursively find all .h5 files
        h5_files = glob.glob(os.path.join(dir_path, "**", "*.h5"), recursive=True)

        # Store the count
        h5_counts[dir_name] = len(h5_files)

    # Save results to a text file if specified
    if output_file:
        try:
            with open(output_file, 'w') as f:
                f.write(f"H5 File Counts in {parent_dir}:\n")
                f.write("-" * 50 + "\n")
                for dir_name, count in h5_counts.items():
                    f.write(f"{dir_name}: {count} H5 files\n")

                # Add a total count at the end
                total_count = sum(h5_counts.values())
                f.write("-" * 50 + "\n")
                f.write(f"Total: {total_count} H5 files\n")

            print(f"Results saved to {output_file}")
        except Exception as e:
            print(f"Error saving results to file: {e}")

    return h5_counts


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
    except Exception as e:
        print(f"Error: {e}")