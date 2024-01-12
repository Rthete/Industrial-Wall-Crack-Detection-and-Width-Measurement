import os
import shutil

def get_file_names(directory):
    files = [os.path.splitext(file)[0] for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]
    return files

def copy_matching_files(dir_a, dir_b, dir_c):
    a_files = get_file_names(dir_a)
    b_files = get_file_names(dir_b)

    for file_name in a_files:
        if file_name.endswith("_result"):
            file_name = file_name[:-7]

        if file_name in b_files:
            source_path = os.path.join(dir_b, file_name + os.path.splitext(os.path.join(dir_a, file_name))[1] + ".png")
            destination_path = os.path.join(dir_c, file_name + os.path.splitext(source_path)[1])
            shutil.copy(source_path, destination_path)

if __name__ == "__main__":
    # Viz中人工筛选数据后，将PNG标记文件一并过滤
    directory_a = "/mnt/d/Projects/project-0624/HJD-VOC/Viz"
    directory_b = "/mnt/d/Projects/project-0624/HJD-VOC/SegmentationClassPNG"
    directory_c = "/mnt/d/Projects/project-0624/HJD-VOC/SegmentationClassPNG-clean"

    copy_matching_files(directory_a, directory_b, directory_c)
