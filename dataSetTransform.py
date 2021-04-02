# -*- encoding: utf-8 -*-
'''
@File : dataSetTransform.py    
@Modify Time : 2021/3/27 下午3:02        
@Author : oldzhang
@Description ： 数据集转换(图片和label文件的命名要一致)
'''
import json
import os
import time
from json import load
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import ElementTree
from xml.etree.ElementTree import SubElement
import xml.etree.ElementTree as ET
from tqdm import tqdm
from cv2 import cv2
import matplotlib.pyplot as plt
import pylab
from Utils import getMessageFromCoco, getMessageFromVoc, getMessageFormJson, prettyXml
import Utils

basePath = "/home/oldzhang/数据标注/菜品/"


def labelme2voc(jsonPath, resPath, imgPath, numWork=2):
    """
    labelme转voc
    :param numWork: 线程数
    :param jsonPath:labelme json文件夹路径
    :param resPath:最终保存的voc xml 路径
    :param imgPath:图片文件夹路径
    """
    start = time.time()
    jsons = sorted(os.listdir(jsonPath))
    imgs = sorted(os.listdir(imgPath))
    for i in range(0, len(jsons)):
        jsons[i] = jsonPath + jsons[i]
        imgs[i] = imgPath + imgs[i]
    # 创建新线程
    a = int(len(jsons) / numWork)
    bg = 0
    ed = a
    threads = []
    for i in range(0, numWork):
        temp = Utils.labelme2vocThread(str(i), jsons[bg:ed], resPath, imgs[bg:ed])
        bg = bg + a
        ed = ed + a
        threads.append(temp)
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print("退出主线程")
    print(time.time() - start)


def labelme2coco(jsonPath, resPath, imgPath, numWork=2):
    """
    labelme到coco数据集的转换
    :param numWork:线程数
    :param jsonPath:labelme 数据位置
    :param resPath:最后输出coco数据的位置，默认保存为resFromLabelme.json
    :param imgPath:图片位置
    """

    start = time.time()
    jsons = sorted(os.listdir(jsonPath))
    imgs = sorted(os.listdir(imgPath))
    for i in range(0, len(jsons)):
        jsons[i] = jsonPath + jsons[i]
        imgs[i] = imgPath + imgs[i]
    # 创建新线程
    a = int(len(jsons) / numWork)
    bg = 0
    ed = a
    threads = []
    for i in range(0, numWork):
        temp = Utils.labelme2vocThread(str(i), jsons[bg:ed], resPath, imgs[bg:ed])
        bg = bg + a
        ed = ed + a
        threads.append(temp)
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print("退出主线程")
    print(time.time() - start)
    print("Finish!")


def voc2coco(vocPath, resPath):
    """
    voc数据集转换为coco数据集
    :param vocPath: voc数据集地址
    :param resPath:结果存放地址(resFromVoc.json)
    """
    # info和license暂时为空
    structure = {
        "info": "null",
        "licenses": "null",
        "images": [],
        "annotations": [],
        "categories": []
    }
    vocDatas = getMessageFromVoc(vocPath)
    # categories
    lid = 0
    for vocData in vocDatas:
        if not isinstance(vocData, list):
            continue
        for bbox in vocData[1]:
            flag = True
            categorize = {
                "id": int,
                "name": str,
                "supercategory": "null",
            }
            for k in structure["categories"]:
                if k["name"] == bbox[0]:
                    flag = False
            if flag:
                categorize["id"] = lid
                lid += 1
                categorize["name"] = bbox[0]
                structure["categories"].append(categorize)

        # image & annotation
    annotationId = 0
    imageId = 0
    for vocData in tqdm(vocDatas):
        if not isinstance(vocData, list):
            continue
        image = {"id": imageId,
                 "width": vocData[0][2],
                 "height": vocData[0][1],
                 "file_name": vocData[0][0],
                 "license": 0,
                 "flickr_url": "null",
                 "coco_url": "null",
                 "date_captured": "null"}
        structure["images"].append(image)

        # annotation

        if not isinstance(vocData, list):
            continue
        for bbox in vocData[1]:
            annotation = {"id": annotationId,
                          "image_id": imageId,
                          "category_id": int,
                          "segmentation": "null",
                          "area": float,
                          "bbox": [0, 0, 0, 0],
                          "iscrowd": 0}
            annotationId += 1
            for cat in structure["categories"]:
                if cat["name"] == bbox[0]:
                    annotation["category_id"] = cat["id"]
                    break
            x = bbox[1]
            y = bbox[2]
            w = abs(bbox[3]-x)
            h = abs(bbox[4]-y)
            annotation["bbox"][0] = x
            annotation["bbox"][1] = y
            annotation["bbox"][2] = w
            annotation["bbox"][3] = h
            annotation["area"] = h * w
            structure["annotations"].append(annotation)
        imageId += 1
    res = json.dumps(structure)
    with open(resPath + "resFromVoc.json", "w") as f:
        f.write(res)
    print("Finish!")


def voc2txt(vocPath, resPath):
    """
    voc-xml转darknet-txt
    :param vocPath:
    :param resPath:
    """
    vocDatas = getMessageFromVoc(vocPath)

    for vocData in vocDatas:
        if not isinstance(vocData, list):
            Utils.createDrakNetTxt([], [vocData, [0, 0, 0]], resPath)
            continue
        imgData = [vocData[0][0], [vocData[0][1], vocData[0][2], 3]]
        objs = vocData[1]
        bboxs = []
        for obj in objs:
            temp = [obj[0], [obj[1], obj[2]], [obj[3], obj[4]]]
            bboxs.append(temp)
        Utils.createDrakNetTxt(bboxs, imgData, resPath)


def coco2voc(cocoPath, resPath):
    """
    coco-json 转voc-xml
    :param cocoPath:
    :param resPath:
    """
    cocoDatas = getMessageFromCoco(cocoPath)
    for cocoData in cocoDatas:
        img = cocoData[0]
        objs = cocoData[1]
        imgData = [img["file_name"], [img["height"], img["width"], 3]]
        imgName = img["file_name"][:img["file_name"].rfind(".")]
        bboxs = []
        for obj in objs:
            obj[1][2] = obj[1][2] + obj[1][0]
            obj[1][3] = obj[1][3] + obj[1][1]
            temp = [obj[0], [obj[1][0], obj[1][1]], [obj[1][2], obj[1][3]]]
            bboxs.append(temp)
        annotation = Utils.createVocXml(bboxs, imgData)
        tree = ElementTree(annotation)
        prettyXml(annotation, '\t', '\n')
        tree.write(resPath + imgName + ".xml", encoding='utf-8')
    print("Finish!")


def coco2txt(cocoPath, resPath):
    cocoDatas = getMessageFromCoco(cocoPath)
    for cocoData in cocoDatas:
        img = cocoData[0]
        objs = cocoData[1]
        imgData = [img["file_name"], [img["height"], img["width"], 3]]
        bboxs = []
        for obj in objs:
            obj[1][2] = obj[1][2] + obj[1][0]
            obj[1][3] = obj[1][3] + obj[1][1]
            temp = [obj[0], [obj[1][0], obj[1][1]], [obj[1][2], obj[1][3]]]
            bboxs.append(temp)
        Utils.createDrakNetTxt(bboxs, imgData, resPath)

labelme2voc(basePath + "json/0-2999/", basePath + "dataTrans/voc/", basePath + "imgs/0-2999/")
labelme2coco(basePath + "json/0-2999/", basePath + "dataTrans/coco/", basePath + "imgs/0-2999/")
voc2coco(basePath + "dataTrans/vocfromcoco/", basePath + "dataTrans/coco/")
voc2txt(basePath + "dataTrans/vocfromcoco/", basePath + "dataTrans/darknet/")
coco2voc("/home/oldzhang/数据标注/菜品/dataTrans/coco/resFromVoc.json", "/home/oldzhang/数据标注/菜品/dataTrans/vocfromcoco/")
coco2txt("/home/oldzhang/数据标注/菜品/dataTrans/coco/resFromLabelme.json", "/home/oldzhang/数据标注/菜品/dataTrans"
                                                                       "/darknetfromcoco/")
