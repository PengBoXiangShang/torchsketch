import xml
from xml.dom import minidom
import os
from torchsketch.utils.general_utils.reset_dir import *
from torchsketch.utils.general_utils.torchsketch import *

def convert_svg_2_accumulative_svgs(svg_url, output_folder, verbose = True):

    doc = minidom.parse(svg_url)
    paths = doc.getElementsByTagName("path")

    assert type(paths) == xml.dom.minicompat.NodeList
    stroke_count = len(paths)
    
    if verbose == True:
        print("{} has {} storkes.".format(svg_url, len(paths)))

    reset_dir(output_folder)


    svg_filename = svg_url.split("/")[-1][:-4]

    try:

        output = open(output_folder + "/" + svg_filename + "_" + str(stroke_count).zfill(6) + ".svg", "w")
        doc.writexml(output)
        output.close()

        for i in range(stroke_count - 1, 0, -1):

            doc.getElementsByTagName("path")[i].setAttribute("visibility", "hidden")

            output = open(output_folder + "/" + svg_filename + "_" + str(i).zfill(6) + ".svg", "w")
            doc.writexml(output)
            output.close()
        	
    except Exception as e:
        print(e)

    if verbose == True:
        print("Details are as follows.")
        print("{} svgs have been produced, and stored in {}.".format(stroke_count, output_folder))
        print(torchsketch())
	
    return stroke_count