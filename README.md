<!--
 * @FilePath: \crack-0921\README.md
 * @Description:  
 * @Author: rthete
 * @Date: 2023-09-21 18:43:59
 * @LastEditTime: 2023-09-21 18:51:33
-->
# 墙面裂缝分割及宽度测量

```
crack-0921
├─ .gitignore
├─ predict_concat.py    # 分割推理脚本
├─ measure_crack.py     # 宽度测量脚本
├─ nets
│  ├─ resnet.py
│  ├─ unet.py
│  ├─ unet_training.py
│  ├─ vgg.py
│  ├─ __init__.py
├─ pth
│  ├─ resnet50_unet_best_epoch_weights.pth
│  └─ vgg_unet_best_epoch_weights.pth
├─ README.md
├─ unet.py      # 基于ResNet50的UNet
├─ unet2.py     # 基于VGG的UNet
└─ utils
```

运行`predict_concat.py`脚本时，需要设置数据集路径`folder_path`，并设置`measure_flag`。