import xml.etree.ElementTree as ET
import numpy as np


def write2XML(crack_areas, crack_locations, file_name):
    # 假设这是你的nparray结果
    result_nparray = np.array([[0, 1, 2], [3, 4, 5]])

    # 创建XML树的根元素
    root = ET.Element("TUNNEL")

    # 创建GENERALINFO部分
    general_info = ET.SubElement(root, "GENERALINFO")
    file_name = ET.SubElement(general_info, "filename")
    file_name.text = "mock_file_name.jpg"
    number = ET.SubElement(general_info, "number")
    number.text = "mock_number"
    details_num = ET.SubElement(general_info, "detailsNum")
    details_num.text = "0"

    # 创建TEXT部分
    text = ET.SubElement(root, "TEXT")

    for crack_area, crack_location in zip(crack_areas, crack_locations):
        item = ET.SubElement(text, "ITEM")

        # 补全crack_location元素
        indices = np.where(crack_area == 255)
        indices = np.array(indices)
        indices[0] += crack_location[0]
        indices[1] += crack_location[1]

        crack_location = ET.SubElement(item, "crack_location")
        crack_location.text = "({})".format(
            ")(".join(
                [",".join(map(str, point)) for point in zip(indices[0], indices[1])]
            )
        )

    # 将根元素包装到ElementTree中
    tree = ET.ElementTree(root)

    # 将XML保存到文件中
    tree.write(f"output/xml/{file_name}.xml", encoding="utf-8", xml_declaration=True)
