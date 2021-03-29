# -*- encoding: utf-8 -*-
'''
@File : dataSetTransform.py    
@Modify Time : 2021/3/27 下午3:02        
@Author : oldzhang
@Description ： 数据集转换(图片和label文件的命名要一致)
'''
import json
import os
from json import load
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import ElementTree
from xml.etree.ElementTree import SubElement
import xml.etree.ElementTree as ET
from tqdm import tqdm
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
        if len(json) == 0:
            return
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


def getMessageFromVoc(vocPath):
    """
    从voc xml中获取数据
    :param vocPath:voc 路径
    :return:res[img,obj]
            img[filename,w,h]
            obj[bbox]
            bbox[label,xmin,ymin,xmax,ymax]
    """
    files = sorted(os.listdir(vocPath))
    res = []
    for file in files:
        fileData = []
        img = []
        tree = ET.parse(vocPath + file)
        root = tree.getroot()
        objs = root.findall("object")
        imgName = root.find("filename").text
        if len(objs) == 0:
            res.append(imgName)
            continue
        size = root.find("size")
        w = size.findtext("width")
        h = size.findtext("height")
        img.append(imgName)
        img.append(w)
        img.append(h)
        fileData.append(img)
        objList = []
        for obj in objs:
            objTemp = []
            label = obj.findtext("name")
            bbox = obj.find("bndbox")
            xmin = bbox.findtext("xmin")
            ymin = bbox.findtext("ymin")
            xmax = bbox.findtext("xmax")
            ymax = bbox.findtext("ymax")
            objTemp.append(label)
            objTemp.append(xmin)
            objTemp.append(ymin)
            objTemp.append(xmax)
            objTemp.append(ymax)
            objList.append(objTemp)
        fileData.append(objList)
        res.append(fileData)
    return res


def labelme2voc(jsonPath, resPath, imgPath):
    """
    labelme转voc
    :param jsonPath:labelme json文件夹路径
    :param resPath:最终保存的voc xml 路径
    :param imgPath:图片文件夹路径
    """
    jsons = sorted(os.listdir(jsonPath))
    imgs = sorted(os.listdir(imgPath))
    for i in tqdm(range(0, len(jsons))):
        lData = getMessageFormJson(jsonPath + jsons[i])

        imgName = imgs[i]
        img = cv2.imread(imgPath + imgName)
        # create xml
        annotation = Element('annotation')
        folder = SubElement(annotation, 'folder')
        folder.text = "菜品数据标注"
        filename = SubElement(annotation, 'filename')
        filename.text = imgName
        if lData == None:
            imgName = imgName[:imgName.rfind(".")]
            tree = ElementTree(annotation)
            prettyXml(annotation, '\t', '\n')
            tree.write(resPath + imgName + ".xml", encoding='utf-8')
            continue
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
    print("Finish!")


def labelme2coco(jsonPath, resPath, imgPath):
    """
    labelme到coco数据集的转换
    :param jsonPath:labelme 数据位置
    :param resPath:最后输出coco数据的位置，默认保存为res.json
    :param imgPath:图片位置
    """
    # info和license暂时为空
    structure = {
        "info": "null",
        "licenses": "null",
        "images": [],
        "annotations": [],
        "categories": []
    }

    jsons = sorted(os.listdir(jsonPath))
    imgs = sorted(os.listdir(imgPath))

    # categories
    lid = 0
    for i in range(0, len(jsons)):
        jsonData = getMessageFormJson(jsonPath + jsons[i])
        if jsonData == None:
            continue
        for j in range(0, len(jsonData), 2):
            flag = True
            categorize = {
                "id": int,
                "name": str,
                "supercategory": "null",
            }
            for k in structure["categories"]:
                if k["name"] == jsonData[j]:
                    flag = False
            if flag:
                categorize["id"] = lid
                lid += 1
                categorize["name"] = jsonData[j]
                structure["categories"].append(categorize)

    # image & annotation
    annotationId = 0
    for i in tqdm(range(0, len(jsons))):
        imgSize = cv2.imread(imgPath + imgs[i]).shape
        image = {"id": i,
                 "width": imgSize[1],
                 "height": imgSize[0],
                 "file_name": imgs[i],
                 "license": 0,
                 "flickr_url": "null",
                 "coco_url": "null",
                 "date_captured": "null"}
        structure["images"].append(image)

        # annotation

        jsonData = getMessageFormJson(jsonPath + jsons[i])
        if jsonData == None:
            continue
        for j in range(0, len(jsonData), 2):
            annotation = {"id": annotationId,
                          "image_id": i,
                          "category_id": int,
                          "segmentation": "null",
                          "area": float,
                          "bbox": [0, 0, 0, 0],
                          "iscrowd": 0}
            annotationId += 1
            for cat in structure["categories"]:
                if cat["name"] == jsonData[j]:
                    annotation["category_id"] = cat["id"]
                    break
            x = jsonData[j + 1][0][0]
            y = jsonData[j + 1][0][1]
            w = jsonData[j + 1][1][0] - x
            h = jsonData[j + 1][1][1] - y
            annotation["bbox"][0] = x
            annotation["bbox"][1] = y
            annotation["bbox"][2] = w
            annotation["bbox"][3] = h
            annotation["area"] = h * w
            structure["annotations"].append(annotation)
    res = json.dumps(structure)
    with open(resPath + "res.json", "w") as f:
        f.write(res)
    print("Finish!")


# labelme2voc(basePath + "json/", basePath + "voc/", basePath + "imgs/")
# labelme2coco(basePath + "json/", basePath + "coco/", basePath + "imgs/")
getMessageFromVoc(basePath + "voc/")
