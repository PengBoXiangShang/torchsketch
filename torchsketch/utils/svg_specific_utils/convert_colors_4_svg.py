import xml
from xml.dom import minidom
import os
from torchsketch.utils.general_utils.color_table import *
from torchsketch.utils.general_utils.torchsketch import *

def convert_colors_4_svg(svg_url, output_folder, colors = ["red", ], verbose = True):

    doc = minidom.parse(svg_url)
    paths = doc.getElementsByTagName("path")

    assert type(paths) == xml.dom.minicompat.NodeList
    stroke_count = len(paths)
    
    if verbose == True:
        print("{} has {} storkes.".format(svg_url, len(paths)))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    produced_count = 0

    svg_filename = svg_url.split("/")[-1][:-4]

    for color in colors:

        if color not in COLOR_TABLE.keys():
            continue

        try:
            for i in range(stroke_count):
                doc.getElementsByTagName("path")[i].setAttribute("stroke", color)
        except Exception as e:
            print(e)
            continue


        output = open(output_folder + "/" + svg_filename + "_" + str(color) + ".svg", "w")
        doc.writexml(output)
        output.close()

        produced_count += 1

    if verbose == True:
        print("Details are as follows.")
        print("{} colors have been inputted.".format(len(colors)))
        print("{} colorful sketches have been produced.".format(produced_count))
        print("{} colors can not been recognized.".format(len(colors) - produced_count))
        print(torchsketch())

    return produced_count
