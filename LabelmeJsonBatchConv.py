# E:\anaconda3\envs\labelme\Scripts\labelme_json_to_dataset .\APHU6833504.json
import os

dataPath = "I:/ContainerNumber/label/reLabel/"  # 放json文件的地方(最终存放转换后数据的地方)
envsPath = 'E:/anaconda3/envs/labelme/Scripts/labelme_json_to_dataset'  # labelme位置
json = os.listdir(dataPath)
with open(dataPath + "conv.cmd", "w") as f:
    for i in json:
        f.write(envsPath + " " + dataPath + i + "\n")
os.system(dataPath + "conv.cmd")
os.remove(dataPath + "conv.cmd")
