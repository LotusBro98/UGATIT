import os
import sys

import numpy as np
import vk
import cv2 as cv

from PIL import Image
import requests
from io import BytesIO

groups = ["oldlentach"]
token = "15e05b6515e05b6515e05b65e2158c70f6115e015e05b6548acce299743657f96c48dd2"

PADDING = 1.3
MIN_AREA = 1000
SIZE = 512

MAX_USERS = 2000


api = vk.api.API(vk.Session(token), v="5.101")

users = []
for group in groups:
    mem = api.groups.getMembers(group_id=group)
    n_users = mem['count']
    if n_users + len(users) >= MAX_USERS:
        n_users = MAX_USERS - len(users)
        if (n_users <= 0):
            continue
    if n_users < 1000:
        n_users = 1000
    for i in range(n_users // 1000):
        mem = api.groups.getMembers(group_id=group, offset=i * 1000)
        users += mem['items']



os.chdir(os.path.dirname(__file__))

face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

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

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def load_photo(user):
    url = user["photo_200"]
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = np.array(img)
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    face = crop_face(img)
    if face is not None:
        cv.imshow("Face", face)
        cv.waitKey(100)
    return face

cnt = 0
taken = 0
for userbatch in batch(users, 1000):
    photos = api.users.get(user_ids=userbatch, fields="photo_200")
    photos = list(filter(lambda x: "photo_200" in x, photos))
    for photo in  photos:
        cnt += 1
        face  = load_photo(photo)
        if face is None:
            continue

        taken += 1
        sys.stdout.write("\r{} / {} / {}".format(taken, cnt, MAX_USERS))
        sys.stdout.flush()
        cv.imwrite("trainB/{}.png".format(taken), face)