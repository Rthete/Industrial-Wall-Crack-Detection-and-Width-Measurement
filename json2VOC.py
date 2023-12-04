from __future__ import print_function

import argparse
import glob
import os
import os.path as osp
import sys

import imgviz
import numpy as np

import labelme

# python json2VOC.py --input_dir "D:\\projects\\project-0624\\项目组\\231028-758-all\\HJD" --output_dir "D:\\projects\\project-0624\\项目组\\231028-758-all\\HJD-VOC"
# python json2VOC.py --input_dir "D:\\projects\\project-0624\\项目组\\231028-758-all\\TB" --output_dir "D:\\projects\\project-0624\\项目组\\231028-758-all\\TB-VOC"
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_dir",default="D:\\projects\\project-0624\\项目组\\231013-本科生汇总\\231013-all", help="input annotated directory")
    parser.add_argument("--output_dir",default="D:\\projects\\project-0624\\项目组\\231013-本科生汇总\\231013-all-VOC", help="output dataset directory")
    parser.add_argument("--labels",default="label.txt", help="labels file")
    parser.add_argument(
        "--noviz", default=True, help="no visualization"
    )
    args = parser.parse_args()

    # if osp.exists(args.output_dir):
    #     print("Output directory already exists:", args.output_dir)
    #     sys.exit(1)
    # os.makedirs(args.output_dir)
    os.makedirs(osp.join(args.output_dir, "JPEGImages"))
    os.makedirs(osp.join(args.output_dir, "SegmentationClass"))
    os.makedirs(osp.join(args.output_dir, "SegmentationClassPNG"))
    if not args.noviz:
        os.makedirs(
            osp.join(args.output_dir, "SegmentationClassVisualization")
        )
    print("Creating dataset:", args.output_dir)

    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        elif class_id == 0:
            assert class_name == "_background_"
        class_names.append(class_name)
    class_names = tuple(class_names)
    print("class_names:", class_names)
    out_class_names_file = osp.join(args.output_dir, "class_names.txt")
    with open(out_class_names_file, "w") as f:
        f.writelines("\n".join(class_names))
    print("Saved class_names:", out_class_names_file)

    for filename in glob.glob(osp.join(args.input_dir, "*.json")):
        print("Generating dataset from:", filename)

        label_file = labelme.LabelFile(filename=filename)

        base = osp.splitext(osp.basename(filename))[0]
        out_img_file = osp.join(args.output_dir, "JPEGImages", base + ".jpg")
        out_lbl_file = osp.join(
            args.output_dir, "SegmentationClass", base + ".npy"
        )
        out_png_file = osp.join(
            args.output_dir, "SegmentationClassPNG", base + ".png"
        )
        if not args.noviz:
            out_viz_file = osp.join(
                args.output_dir,
                "SegmentationClassVisualization",
                base + ".jpg",
            )

        with open(out_img_file, "wb") as f:
            f.write(label_file.imageData)
        img = labelme.utils.img_data_to_arr(label_file.imageData)

        lbl, _ = labelme.utils.shapes_to_label(
            img_shape=img.shape,
            shapes=label_file.shapes,
            label_name_to_value=class_name_to_id,
        )
        labelme.utils.lblsave(out_png_file, lbl)

        np.save(out_lbl_file, lbl)

        if not args.noviz:
            viz = imgviz.label2rgb(
                label=lbl,
                #img改成image，labelme接口的问题不然会报错
                #img=imgviz.rgb2gray(img),
                # image=imgviz.rgb2gray(img),
                image = img,
                font_size=15,
                label_names=class_names,
                loc="rb",
            )
            imgviz.io.imsave(out_viz_file, viz)


if __name__ == "__main__":
    main()
