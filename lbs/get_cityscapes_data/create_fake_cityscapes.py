import os
from os.path import join, split
import sys
import json
import random
import numpy as np
from scipy.misc import imsave
import matplotlib
from PIL import Image

data_dir = "/data/nikita/lbs/data"
cityscapes_dir = join(data_dir, "cityscapes")


def make_img_text_file(phase="train"):
    with open(join(cityscapes_dir, phase + "_images.txt"), "w") as f:
        for location in ["loc1", "loc2"]:
            for img_num in range(3):
                f.write("leftImg8bit/" + phase + "/" + location + "/" +
                        location + "_00000" + str(img_num) +
                        "_000019_leftImg8bit.png\n")


def make_labels_text_file(phase="train"):
    with open(join(cityscapes_dir, phase + "_labels.txt"), "w") as f:
        for location in ["loc1", "loc2"]:
            for img_num in range(3):
                f.write("gtFine/" + phase + "/" + location + "/" + location +
                        "_00000" + str(img_num) +
                        "_000019_gtFine_trainIds.png\n")


def drawImage(w=2048, h=1024, min_color=0, max_color=256, channels=3):
    if channels > 1:
        imarray = np.random.randint(min_color, max_color, (h, w, channels))
    else:
        imarray = np.random.randint(min_color, max_color, (h, w))
    #im = Image.fromarray(imarray.astype('uint8'))
    return imarray


def make_leftImg8bit():
    main_folder_dir = join(cityscapes_dir, "leftImg8bit")
    os.mkdir(main_folder_dir)
    for phase in ["train", "val", "test"]:
        os.mkdir(join(main_folder_dir, phase))
        for location in ["loc1", "loc2"]:
            os.mkdir(
                join(cityscapes_dir, "leftImg8bit/" + phase + "/" + location))
            for img_num in range(3):
                im = drawImage(w=2048, h=1024)
                imsave(
                    join(
                        cityscapes_dir, "leftImg8bit/" + phase + "/" + location
                        + "/" + location + "_00000" + str(img_num) +
                        "_000019_leftImg8bit.png"), im)


def make_gtFine():
    main_folder_dir = join(cityscapes_dir, "gtFine")
    os.mkdir(main_folder_dir)
    for phase in ["train", "val", "test"]:
        os.mkdir(join(main_folder_dir, phase))
        for location in ["loc1", "loc2"]:
            os.mkdir(join(cityscapes_dir, "gtFine/" + phase + "/" + location))
            for img_num in range(3):
                for ids in ["color", "instanceIds", "labelIds"]:
                    im = drawImage(w=2048, h=1024, max_color=1, channels=1)
                    imsave(
                        join(
                            cityscapes_dir, "gtFine/" + phase + "/" + location
                            + "/" + location + "_00000" + str(img_num) +
                            "_000019_gtFine_" + ids + ".png"), im)
                im = drawImage(w=2048, h=1024, min_color=255, channels=1)
                imsave(
                    join(
                        cityscapes_dir, "gtFine/" + phase + "/" + location +
                        "/" + location + "_00000" + str(img_num) +
                        "_000019_gtFine_trainIds.png"), im)

                data = {
                    "imgHeight":
                    1024,
                    "imgWidth":
                    2048,
                    "objects": [{
                        "label": "ego vehicle",
                        "polygon": [[271, 1023], [387, 1009]]
                    }, {
                        "label": "out of roi",
                        "polygon": [[0, 0], [2048, 0]]
                    }]
                }
                with open(
                        join(
                            cityscapes_dir, "gtFine/" + phase + "/" + location
                            + "/" + location + "_00000" + str(img_num) +
                            "_000019_gtFine_polygons.json"), "w") as f:
                    json.dump(data, f)


def get_fake_cityscapes(data_root):
    global data_dir, cityscapes_dir
    data_dir = data_root
    cityscapes_dir = join(data_dir, "cityscapes")
    os.mkdir(cityscapes_dir)
    make_img_text_file("train")
    make_img_text_file("val")
    make_img_text_file("test")
    make_labels_text_file("train")
    make_labels_text_file("val")
    make_leftImg8bit()
    make_gtFine()
    info = {
        "std": [0.1829540508368939, 0.18656561047509476, 0.18447508988480435],
        "mean":
        [0.29010095242892997, 0.32808144844279574, 0.28696394422942517]
    }
    with open(join(cityscapes_dir, "info.json"), "w") as f:
        json.dump(info, f)
