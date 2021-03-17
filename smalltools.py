import os
import shutil
from json import load


def labelme2Json(dataPath, envsPath):
    """
    labelme文件批量转jsonBatch
    envsPath:labelme位置
    dataPath:放json文件的地方(最终存放转换后数据的地方)
    """
    json = os.listdir(dataPath)
    with open(dataPath + "conv.cmd", "w") as f:
        for i in json:
            f.write(envsPath + " " + dataPath + i + "\n")
    os.system(dataPath + "conv.cmd")
    os.remove(dataPath + "conv.cmd")
    "/ media / oldzhang / Data & Model & Course / data / ContainerNumber"


def getAlljsonFromFolder(FolderPath):
    """
    获取文件夹中所有json文件名
    :param FolderPath:文件夹路径
    :return:Json文件列表
    """
    files = os.listdir(FolderPath)
    jsonFiles = []
    for i in files:
        if i[-4:] == "json":
            jsonFiles.append(i)
    return jsonFiles


def moveFiles(dataPath, toPath):
    """
    移动文件
    @param dataPath:要移动的文件路径
    @param toPath:目的地路径(be sure don't exist “/” in the last)
    """
    shutil.move(dataPath, toPath)


def getCoordinateFormJson(jsonPath):
    """
    从labelme格式的json文件中获取坐标
    :param jsonPath:labelme格式的json文件路径
    :return:<list>所有坐标
    """
    res = []
    with open(jsonPath) as f:
        json = load(f)
        for i in json["shapes"]:
            res.append(i["points"])
    return res


def saveJson2Txt(jsonPath, savePath):
    """
    将提取的坐标以ICDR2015的格式存入txt
    :param jsonPath:
    :param savePath:
    """
    files = os.listdir(jsonPath)
    files = sorted(files)
    for i in files:
        imgName = i[:-5] + ".txt"
        with open(savePath + imgName, "w") as f:
            writeData = getCoordinateFormJson(jsonPath)
            for j in writeData:
                f.write(str(j).replace("[", "").replace("]", "").replace(" ", ""))
                f.write("\n")
