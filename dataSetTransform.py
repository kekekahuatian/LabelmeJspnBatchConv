# -*- encoding: utf-8 -*-
'''
@File : dataSetTransform.py    
@Modify Time : 2021/3/27 下午3:02        
@Author : oldzhang
@Description ： 数据集转换
'''
import datetime
import os
from json import load
import json
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import ElementTree
from xml.etree.ElementTree import SubElement

from cv2 import cv2

basePath = "/home/oldzhang/数据标注/菜品/"


def getMessageFormJson(jsonPath):
    """
    从labelme格式的json文件中获取坐标和类别
    :param jsonPath:labelme格式的json文件路径
    :return:<list>所有坐标和类别
    """
    res = []
    with open(jsonPath) as f:
        json = load(f)
        for i in json["shapes"]:
            res.append(i["label"])
            res.append(i["points"])
    return res


def prettyXml(element, indent, newline, level=0):
    # 判断element是否有子元素
    if element:
        # 如果element的text没有内容
        if element.text == None or element.text.isspace():
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
            # 此处两行如果把注释去掉，Element的text也会另起一行
    # else:
    # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
    temp = list(element)  # 将elemnt转成list
    for subelement in temp:
        # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
        if temp.index(subelement) < (len(temp) - 1):
            subelement.tail = newline + indent * (level + 1)
        else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
            subelement.tail = newline + indent * level
            # 对子元素进行递归操作
        prettyXml(subelement, indent, newline, level=level + 1)


def labelme2voc(jsonPath, resPath, imgPath):
    """
    labelme转voc
    :param jsonPath:labelme json文件夹路径
    :param resPath:最终保存的voc xml 路径
    :param imgPath:图片文件夹路径
    """
    jsons = sorted(os.listdir(jsonPath))
    imgs = sorted(os.listdir(imgPath))
    for i in range(0, len(jsons)):
        lData = getMessageFormJson(jsonPath + jsons[i])
        imgName = imgs[i]
        img = cv2.imread(imgPath + imgName)
        # create xml
        annotation = Element('annotation')
        folder = SubElement(annotation, 'folder')
        folder.text = "菜品数据标注"
        filename = SubElement(annotation, 'filename')
        filename.text = imgName

        size = SubElement(annotation, "size")
        depth = SubElement(size, "depth")
        height = SubElement(size, "height")
        width = SubElement(size, "width")
        depth.text = str(img.shape[2])
        width.text = str(img.shape[1])
        height.text = str(img.shape[0])

        segmented = SubElement(annotation, "segmented")
        segmented.text = "1"
        # create object
        for j in range(0, len(lData), 2):
            object = SubElement(annotation, "object")
            name = SubElement(object, "name")
            name.text = lData[j]
            pose = SubElement(object, "pose")
            pose.text = "top"
            truncated = SubElement(object, "truncated")
            truncated.text = "0"
            difficult = SubElement(object, "difficult")
            difficult.text = "0"

            bndbox = SubElement(object, "bndbox")
            xmin = SubElement(bndbox, "xmin")
            ymin = SubElement(bndbox, "ymin")
            xmax = SubElement(bndbox, "xmax")
            ymax = SubElement(bndbox, "ymax")
            xmin.text = str(lData[j + 1][0][0])
            ymin.text = str(lData[j + 1][0][1])
            xmax.text = str(lData[j + 1][1][0])
            ymax.text = str(lData[j + 1][1][1])

        tree = ElementTree(annotation)
        imgName = imgName[:imgName.rfind(".")]
        prettyXml(annotation, '\t', '\n')
        tree.write(resPath + imgName + ".xml", encoding='utf-8')


def labelme2coco(jsonPath, resPath, imgPath):
    # info和license暂时为空
    image = {"id": int,
             "width": int,
             "height": int,
             "file_name": str,
             "license": 0,
             "flickr_url": "null",
             "coco_url": "null",
             "date_captured": datetime}
    annotation = {"id": int,
                  "image_id": int,
                  "category_id": int,
                  "segmentation": "null",
                  "area": float,
                  "bbox": ["x", "y", "width", "height"],
                  "iscrowd": 0}
    categorize = {
        "id": int,
        "name": str,
        "supercategory": str,
    }
    structure = {
        "info": "null",
        "licenses": "null",
        "images": [image],
        "annotations": [annotation],
        "categories": [categorize]
    }

    jsons = sorted(os.listdir(jsonPath))
    imgs = sorted(os.listdir(imgPath))

    for i in range(0, len(jsons)):
        jsonData = getMessageFormJson(jsonPath + jsons[i])
        imgName = imgs[i]
        img = cv2.imread(imgPath + imgName)

        for i in range(0, len(jsonData), 2):

            print(1)


# labelme2voc(basePath + "json/", basePath + "transTest/", basePath + "imgs/")
labelme2coco(basePath + "json/", basePath + "coco/", basePath + "imgs/")
