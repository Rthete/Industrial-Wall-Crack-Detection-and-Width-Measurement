import os

def rename_files_in_directory(directory_path):
    if not os.path.exists(directory_path):
        print(f"The directory '{directory_path}' does not exist.")
        return

    count = 161

    for filename in os.listdir(directory_path):
        if filename.startswith("negative"):
            old_path = os.path.join(directory_path, filename)
            new_filename = f"negative_sample{count}.jpg"
            new_path = os.path.join(directory_path, new_filename)
            
            os.rename(old_path, new_path)
            
            print(f"Renamed: {filename} to {new_filename}")
            
            count += 1
            
if __name__ == "__main__":

    directory_path = "/mnt/d/Projects/project-0624/HJD-VOC/JPEGImages-clean"
    rename_files_in_directory(directory_path)
