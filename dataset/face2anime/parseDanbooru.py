import cv2 as cv
import os
import sys
import numpy as np
import requests as rq
from bs4 import BeautifulSoup
import json
import sys

WIDTH = 800
#TAGS = "computer+solo"
TAGS="solo"
#BASE_URL = "http://104.24.110.222/" #http://danbooru.idanbooru.com
BASE_URL = "http://danbooru.idanbooru.com"
START_URL = BASE_URL + "/posts?ms=1&page={}&tags={}&utf8=%E2%9C%93"
PAGES = 200
PADDING = 1.3
MIN_AREA = 1000

SIZE = 512

DIRTY_TAGS = ["nude", "pornography", "untied", "cum"]

os.chdir(os.path.dirname(__file__))

face_cascade = cv.CascadeClassifier("lbpcascade_animeface.xml")

def crop_face(image):
    faces = face_cascade.detectMultiScale(image, 1.2, 5)

    h0, w0 = image.shape[:2]

    if w0 > h0:
        w = h0
        h = h0
        y = h // 2
        x = (w0 - w) // 2 + w // 2
    else:
        h = w0
        w = w0
        x = w // 2
        y = h // 2

    if len(faces) == 0:
        return None

    if (len(faces) != 0):
        faces = sorted(faces, key=(lambda f: f[2] * f[3]), reverse=True)
        x1, y1, w1, h1 = faces[0]
        if (w1 * h1 < MIN_AREA):
            return None
        x = x1 + w1 // 2
        y = y1 + h1 // 2
        w = w1
        h = h1

    #     x -= int(w * (PADDING-1) / 2)
    #     y -= int(h * (PADDING-1) / 2)

    w = int(w * PADDING)
    h = int(h * PADDING)

    if (x < w // 2):
        x = w // 2
    if (y < h // 2):
        y = h // 2
    if x > w0 - w // 2:
        x = w0 - w // 2
    if y > h0 - h // 2:
        y = h0 - h // 2

    x -= w // 2
    y -= h // 2

    face = image[y:y + h, x:x + w]
    face = cv.resize(face, (SIZE, SIZE))
    return face

srcs = []
for page in range(1,PAGES):
    url = START_URL.format(page, TAGS)
    START = rq.get(url)
    parsed = BeautifulSoup(START.text)

    imgs = parsed.body.find_all('a')
    for img in imgs:
        if img.find('picture') == None:
            continue
        if img.has_key("href") and "posts" in img["href"]:
            srcs.append(img["href"])

    sys.stdout.write("\r{}/{}".format(page, PAGES))
    sys.stdout.flush()

alltags = []

all_len = len(srcs)
cnt = 0
taken = 0
for src in srcs:
    taken += 1
    res = rq.get(BASE_URL + src)
    parsed = BeautifulSoup(res.text)
    imgs = parsed.find_all("img")
    if len(imgs) == 0:
        continue
    img = imgs[0]
    if not img.has_attr("src"):
        continue

    if img.has_attr("class"):
        continue

    src = img["src"]
    tags = img["data-tags"]
    tags = tags.split(" ")

    #     print(tags)

    br = False
    for tag in DIRTY_TAGS:
        if tag in tags:
            #             print("dirty: " + tag)
            br = True
            break
    if br:
        continue

    alltags += tags

    resp = rq.get(src)
    try:
        image = np.asarray(bytearray(resp.content), dtype="uint8")
        image = cv.imdecode(image, cv.IMREAD_COLOR)
        if image is None:
            continue
    except:
        continue

    face = crop_face(image)
    if face is None:
        continue

    # cv.imshow("Image", image)
    cv.imshow("Face", face)
    cv.waitKey(10)
    #     clear_output(True)
    #     plt.imshow(cv.cvtColor(face, cv.COLOR_BGR2RGB))
    #     plt.axis('off')
    #     plt.show()
    #     plt.subplot(1, 5, 2)
    #     plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    #     plt.show()

    cv.imwrite("trainA/{}.png".format(cnt), face)
    cnt += 1
    sys.stdout.write("\r{}/{}".format(taken, all_len))
    sys.stdout.flush()

alltags = sorted(list(set(alltags)))