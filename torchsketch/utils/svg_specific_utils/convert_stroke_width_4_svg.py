import xml
from xml.dom import minidom
import os
from torchsketch.utils.general_utils.torchsketch import *

def convert_stroke_width_4_svg(svg_url, output_folder, stroke_width = 6, verbose = True):

    doc = minidom.parse(svg_url)
    paths = doc.getElementsByTagName("path")

    assert type(paths) == xml.dom.minicompat.NodeList
    stroke_count = len(paths)
    
    if verbose == True:
        print("{} has {} storkes.".format(svg_url, len(paths)))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    svg_filename = svg_url.split("/")[-1][:-4]

    

    try:
        for i in range(stroke_count):
            doc.getElementsByTagName("path")[i].setAttribute("stroke-width", str(stroke_width))
    except Exception as e:
        print(e)


    output = open(output_folder + "/" + svg_filename + "_stroke_width_" + str(stroke_width) + ".svg", "w")
    doc.writexml(output)
    output.close()


    if verbose == True:
        print("Details are as follows.")
        print("The stroke widths have been modified as {}.".format(stroke_width))
        print(torchsketch())

    return stroke_width
