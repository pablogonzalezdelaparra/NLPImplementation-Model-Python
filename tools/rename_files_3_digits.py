import os

def rename_files_with_three_digits(directory):
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            # Split the filename by hyphens
            parts = filename.split('-')
            print(parts)
            # Check if the filename has two parts and the second part is a two-digit number
            if parts[1].isdigit() and len(parts[1]) == 2:
                # Generate new filename with three digits
                new_filename = f"{parts[0]}-{int(parts[1]):03d}-5.txt"
                # Rename the file
                os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
                print(f"Renamed {filename} to {new_filename}")

# Replace 'directory_path' with the path to your directory containing the files
directory_path = './generated_data/paraphrase_data'
rename_files_with_three_digits(directory_path)
