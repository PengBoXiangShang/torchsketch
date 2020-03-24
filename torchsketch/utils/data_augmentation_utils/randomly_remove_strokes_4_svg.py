import xml
from xml.dom import minidom
import os
import numpy as np
import time
from torchsketch.utils.general_utils.reset_dir import *
from torchsketch.utils.general_utils.torchsketch import *


def randomly_remove_strokes_4_svg(svg_url, output_folder, remove_ratio = 0.3, verbose = True):

    if remove_ratio <= 0.0 or remove_ratio >= 1.0:
        print("The input parameter of remove_ratio should be defined in (0.0, 1.0).")
        print("Please choose a proper remove_ratio value, and retry this API.")
        print(torchsketch())
        return 0

    doc = minidom.parse(svg_url)
    paths = doc.getElementsByTagName("path")

    assert type(paths) == xml.dom.minicompat.NodeList
    stroke_count = len(paths)

    
    if verbose == True:
        print("{} has {} storkes.".format(svg_url, stroke_count))

    reset_dir(output_folder)


    svg_filename = svg_url.split("/")[-1][:-4]

    seed = int(time.time())
    np.random.seed(seed)

    index_list = np.arange(stroke_count)

    np.random.shuffle(index_list)

    random_list = index_list[ : int(stroke_count * remove_ratio)]

    try:
        for i in random_list:            
            doc.getElementsByTagName("path")[int(i)].setAttribute("visibility", "hidden")


        output = open(output_folder + "/" + svg_filename + "_" + "randomly_partially_removed" + ".svg", "w")
        doc.writexml(output)
        output.close()
        	
    except Exception as e:
        print(e)

    

    if verbose == True:
        print("Details are as follows.")
        print("{} strokes have been removed RANDOMLY.".format(int(stroke_count * remove_ratio)))
        print("The produced sketch is stored in {}".format(output_folder))
        print(torchsketch())
	
    return stroke_count