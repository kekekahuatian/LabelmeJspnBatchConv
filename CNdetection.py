# -*- encoding: utf-8 -*-
"""
@File : CNdetection.py
@Modify Time : 2021/3/20 下午3:45
@Author : oldzhang
@Description ：检测相关小工具
"""
import os
import shutil
from json import load
import matplotlib.pyplot as plt
import json
import cv2
from tqdm import tqdm

basePath = "/media/oldzhang/Data&Model&Course/data/ContainerNumber/"


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


def moveFile(dataPath, toPath):
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


def imageCrop(rate, imgPath):
    """
    按高度缩放裁减
    :param rate:缩放比例
    :param imgPath:
    :return:CropImg
    """
    img = cv2.imread(imgPath)
    h, w, _ = img.shape
    img = img[0:int(h * (1 - rate)), 0:w]
    return img


def getCoordinateFromTxt(dataPath):
    """
    从ICDR2015格式的txt中获取数据
    :param dataPath:
    :return:按文件返回<list>
    """
    files = sorted(os.listdir(dataPath))
    res = []
    for i in files:
        with open(dataPath + i, "r") as f:
            res.append(f.readlines())
    return res


def drawCoordinate(dataPath, imgPath, savePath=None):
    """
    将标签画到原图上测试
    :param savePath: 保存路径，默认不保存
    :param dataPath:
    :param imgPath:标签数据位置
    """
    imgs = sorted(os.listdir(imgPath))
    coors = getCoordinateFromTxt(dataPath)
    court = 0
    for i in range(0, len(coors)):
        img = cv2.imread(imgPath + imgs[i])
        plt.imshow(img)
        for j in coors[i]:
            # change type of content in the list
            a = list(map(float, j.split(",")))
            plt.plot((a[0], a[2]), (a[1], a[3]))
            plt.plot((a[2], a[4]), (a[3], a[5]))
            plt.plot((a[4], a[6]), (a[5], a[7]))
            plt.plot((a[6], a[0]), (a[7], a[1]))
        if savePath:
            plt.savefig(savePath + imgs[i], dpi=300)
        plt.show()
        print(court)
        court += 1


def saveToOCRTxt(imgPath, labelPath, savePath):
    """
    生成数据标签对应txt
    :param imgPath:
    :param labelPath:
    :param savePath:
    """
    imgs = sorted(os.listdir(imgPath))
    labels = sorted(os.listdir(labelPath))
    with open(savePath + "pre.txt", "w") as f:
        for i in range(0, len(imgs)):
            f.write(imgPath + imgs[i])
            f.write("\t")
            f.write(labelPath + labels[i])
            f.write("\n")


def addTxt(txtPath, content):
    """
    在txt每行末尾追加content
    :param content: 追加的内容<list>
    :param txtPath:
    """
    files = sorted(os.listdir(txtPath))
    for file in files:
        with open(txtPath + file, "r") as f:
            lines = f.readlines()
            for i in range(0, len(lines)):
                lines[i] = lines[i].replace("\n", ",") + content[i]
        with open(txtPath + file, "w") as f:
            for i in lines:
                f.write(i)
                f.write("\n")


def preTest(testImgPath, testLabelPath, trainImgPath, trainLabelPath):
    """
    根据测试集中的图像，移动对应标签至测试集标签，并删除训练集对应图像
    :param testImgPath:
    :param testLabelPath:
    :param trainImgPath:
    :param trainLabelPath:
    """
    testImgs = sorted(os.listdir(testImgPath))
    for testImg in testImgs:
        if os.path.isfile(trainImgPath + testImg):
            os.remove(trainImgPath + testImg)
        testTxt = testImg[:-4] + ".txt"
        shutil.move(trainLabelPath + testTxt, testLabelPath)


def filesRename(filesPath):
    """
    批量文件改名
    :param filesPath:
    """
    count = 3081
    files = sorted(os.listdir(filesPath))
    for file in files:
        os.rename(filesPath + file, filesPath + "img_%d" % count + ".jpg")
        count += 1


def moveFiles(suffix, sourcePath, resPath):
    """
    根据文件类型移动文件
    :param suffix:文件后缀名(不用加.)
    :param sourcePath:源文件夹路径
    :param resPath:目标文件夹路径
    """
    files = os.listdir(sourcePath)
    for file in files:
        if file[file.rfind("."):] == "." + suffix:
            moveFile(sourcePath + file, resPath)


def compareList(pathA, pathB):
    """
    返回两个列表相同的元素<list>
    :param pathA:
    :param pathB:
    """
    filesA = sorted(os.listdir(pathA))
    filesB = sorted(os.listdir(pathB))
    res = []
    for a in filesA:
        for b in filesB:
            if a == b:
                res.append(a)
    return res


def compareFloder(pathA, pathB):
    """
    扩充文件b使于a同步并写入基础labelme json的格式
    :param pathA:
    :param pathB:
    :param suffix:扩充文件后缀
    """
    filesA = sorted(os.listdir(pathA))
    filesB = sorted(os.listdir(pathB))
    j = 0

    for i in range(0, len(filesA)):
        fileA = filesA[i]
        fileB = filesB[j]
        fileAName = fileA[:fileA.rfind(".")]
        fileBName = fileB[:fileB.rfind(".")]
        if fileAName != fileBName:
            j -= 1
            img = cv2.imread(pathA + fileA).shape
            structure = {
                "version": "4.5.7",
                "flags": {},
                "shapes": [],
                "imagePath": fileA,
                "imageData": "null",
                "imageHeight": img[0],
                "imageWidth": img[1]
            }

            with open(pathB + fileAName + ".json", "w") as f:
                res = json.dumps(structure)
                f.write(res)
        j += 1


compareFloder("/home/oldzhang/下载/test/imgs/", "/home/oldzhang/下载/test/json/")
