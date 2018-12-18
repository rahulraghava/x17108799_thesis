from pycocotools.coco import COCO

import numpy as np
import skimage.io as io
import cv2
import sys
import os
import argparse
import random

parser = argparse.ArgumentParser(description="Download images form COCO dataset for specfic category")
parser.add_argument("file", help="File with URL for download")
parser.add_argument("category", help="Category to download")
parser.add_argument("-l","--len",help="For just number of images", action="store_true")
args = parser.parse_args()

ver = args.file.split('/')[-1].find('val') >= 0

path = "data/" + ("val" if args.file.split('/')[-1].find('val') >= 0 else "train") 

if not os.path.exists(path):
    os.makedirs(path)

cl = ['person','car','airplane','bus','train','dog']

coco = COCO(args.file)
imids = []
if args.category == "not":
    cats = coco.loadCats(coco.getCatIds())
    ids = [cat['id'] for cat in cats]
    for c in cl:
        ids.remove(coco.getCatIds(catNms = c)[0])
    for i in range(0,300 if ver else 10000):
        t = []
        while len(t) < 100:
            t = coco.getImgIds(catIds = random.sample(ids, 2))
        e = True
        while e:
            v = random.choice(t)
            if v not in imids:
                imids.append(v)
                e = False
    print(len(imids))

else:
    cid = coco.getCatIds(catNms = args.category)
    imids = coco.getImgIds(catIds=cid)
    imids = imids[:10000]

idet = coco.loadImgs(imids)
print("The number of image:-", len(imids), " of ", args.category)
if args.len:
    exit()
k = 0
for i in idet:
    img = io.imread(i["coco_url"])
    p = path + "/" + args.category + "_" + str(i['id']) + ".jpg"
    cv2.imwrite(p, img )
    print("Downloading-->", k , " of ", len(imids),end='\r')
    k += 1
