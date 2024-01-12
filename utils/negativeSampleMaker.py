from PIL import Image
import os

def convert_jpg_to_empty_png(input_directory, output_directory):
    if not os.path.exists(input_directory):
        print(f"The input directory '{input_directory}' does not exist.")
        return

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.lower().endswith(".jpg") and filename.lower().startswith("negative") :
            jpg_path = os.path.join(input_directory, filename)
            png_filename = os.path.splitext(filename)[0] + ".png"
            png_path = os.path.join(output_directory, png_filename)

            img = Image.open(jpg_path)
            width, height = img.size

            # 创建一个尺寸相同，像素值全部为0的png图片
            empty_img = Image.new("L", (width, height), 0)
            empty_img.save(png_path)

            print(f"Converted: {filename} to {png_filename}")

if __name__ == "__main__":
    # 对负样本数据制作label
    input_directory = "/mnt/d/Projects/project-0624/HJD-VOC/JPEGImages-clean"
    output_directory = "/mnt/d/Projects/project-0624/HJD-VOC/SegmentationClassPNG-clean"
    convert_jpg_to_empty_png(input_directory, output_directory)
