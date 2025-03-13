# check_data.py
import os
import glob


def find_h5_files(directory):
    """Find all .h5 files in the directory tree."""
    print(f"Searching for .h5 files in: {directory}")

    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return

    # Get absolute path
    abs_path = os.path.abspath(directory)
    print(f"Absolute path: {abs_path}")

    # List immediate contents
    print("\nImmediate directory contents:")
    try:
        contents = os.listdir(directory)
        for item in contents:
            full_path = os.path.join(directory, item)
            if os.path.isdir(full_path):
                print(f"  DIR: {item}")
            else:
                print(f"  FILE: {item}")
    except Exception as e:
        print(f"Error listing directory: {e}")

    # Find all .h5 files
    print("\nSearching for .h5 files recursively...")
    h5_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.h5'):
                h5_files.append(os.path.join(root, file))

    # Print results
    if h5_files:
        print(f"\nFound {len(h5_files)} .h5 files:")
        for i, file in enumerate(h5_files[:5]):  # Show first 5
            print(f"  {i + 1}. {file}")
        if len(h5_files) > 5:
            print(f"  ... and {len(h5_files) - 5} more files.")

        # Print potential patient directories
        print("\nPotential patient directories (containing .h5 files):")
        patient_dirs = set()
        for file in h5_files:
            patient_dirs.add(os.path.dirname(file))

        for i, dir_path in enumerate(sorted(patient_dirs)[:5]):
            print(f"  {i + 1}. {dir_path}")
        if len(patient_dirs) > 5:
            print(f"  ... and {len(patient_dirs) - 5} more directories.")
    else:
        print("\nNo .h5 files found in the directory tree.")

    # Check how find_patient_dirs would work
    print("\nSimulating find_patient_dirs function:")
    patient_dirs = []
    for root, _, files in os.walk(directory):
        if any(file.endswith('.h5') for file in files):
            patient_dirs.append(root)

    if patient_dirs:
        print(f"Found {len(patient_dirs)} patient directories:")
        for i, dir_path in enumerate(patient_dirs[:5]):
            print(f"  {i + 1}. {dir_path}")
        if len(patient_dirs) > 5:
            print(f"  ... and {len(patient_dirs) - 5} more directories.")
    else:
        print("No patient directories found.")


if __name__ == "__main__":
    find_h5_files("dummydata")