# 上班摸鱼系列———2 把你的脸变成Emoji
<div align=center>

![copy_0EAC1BFE-AD61-426E-BD9B-D32E10EAD362](https://github.com/user-attachments/assets/eae3bdc8-e6e1-4d5e-ae5e-76384a327e72)

</div>

当你上班觉得很无聊的时候你就启动它，把它挂在右上角，当你上班偷偷跟朋友扯皮的时候，只要你一微笑，你的脑袋就会变成emoji。

# 介绍
直接运行show.py脚本就可以启动了，其中best.pt是我用YOLO训练出来的情绪识别的模型，本质是目标检测，数据集用的是[Facial Expression Image Data AFFECTNET YOLO Format](https://www.kaggle.com/datasets/cubeai/facial-expression-detection-for-yolov8?select=test)，给想自己训练的小伙伴一个思路。Happy.png是没有背景的emoji（在淘宝斥巨资买的），这个模型还可以检测其他的情绪，可以自己换，我是只用了快乐的表情。



