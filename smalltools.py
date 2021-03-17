import os
import shutil


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

    @param dataPath:要移动的文件路径
    @param toPath:目的地路径(be sure don't exist “/” in the last)
    """
    shutil.move(dataPath, toPath)


