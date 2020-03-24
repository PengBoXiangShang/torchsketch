import xml
from xml.dom import minidom
import os
import numpy as np
import time
from torchsketch.utils.general_utils.torchsketch import *

def randomly_convert_stroke_width_4_svg(svg_url, output_folder, stroke_width = 6, selected_strokes = 3, verbose = True):

    doc = minidom.parse(svg_url)
    paths = doc.getElementsByTagName("path")

    assert type(paths) == xml.dom.minicompat.NodeList
    stroke_count = len(paths)
    
    if verbose == True:
        print("{} has {} storkes.".format(svg_url, len(paths)))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    svg_filename = svg_url.split("/")[-1][:-4]

    seed = int(time.time())
    np.random.seed(seed)

    index_list = np.arange(stroke_count)

    np.random.shuffle(index_list)

    random_list = index_list[ : selected_strokes]

    
    try:
        for i in random_list:
            doc.getElementsByTagName("path")[i].setAttribute("stroke-width", str(stroke_width))
    except Exception as e:
        print(e)


    output = open(output_folder + "/" + svg_filename + "_randomly_selected_" + str(selected_strokes) + "_strokes_stroke_width_" + str(stroke_width) + ".svg", "w")
    doc.writexml(output)
    output.close()


    if verbose == True:
        print("Details are as follows.")
        print("{} strokes have been selected randomly.".format(selected_strokes))
        print("The stroke widths have been modified as {}.".format(stroke_width))
        print(torchsketch())

    return stroke_width
