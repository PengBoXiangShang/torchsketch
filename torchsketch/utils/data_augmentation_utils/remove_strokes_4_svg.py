import xml
from xml.dom import minidom
import os
from torchsketch.utils.general_utils.reset_dir import *
from torchsketch.utils.general_utils.torchsketch import *

def remove_strokes_4_svg(svg_url, output_folder, remove_ratio = 0.3, verbose = True):

    if remove_ratio <= 0.0 or remove_ratio >= 1.0:
        print("The input parameter of remove_ratio should be defined in (0.0, 1.0).")
        print("Please choose a proper remove_ratio value, and retry this API.")
        print(torchsketch())
        return 0

    doc = minidom.parse(svg_url)
    paths = doc.getElementsByTagName("path")

    assert type(paths) == xml.dom.minicompat.NodeList
    stroke_count = len(paths)

    stroke_length_array = [len(path.getAttribute('d')) for path in paths]

    assert len(stroke_length_array) == stroke_count

    stroke_length_array.sort()

    threshold_value = stroke_length_array[int(stroke_count * remove_ratio)] 
    
    if verbose == True:
        print("{} has {} storkes.".format(svg_url, stroke_count))

    reset_dir(output_folder)


    svg_filename = svg_url.split("/")[-1][:-4]

    try:

        for i in range(stroke_count):

            if len(doc.getElementsByTagName("path")[i].getAttribute("d")) <= threshold_value:

                doc.getElementsByTagName("path")[i].setAttribute("visibility", "hidden")



        output = open(output_folder + "/" + svg_filename + "_" + "partially_removed" + ".svg", "w")
        doc.writexml(output)
        output.close()
        	
    except Exception as e:
        print(e)

    if verbose == True:
        print("Details are as follows.")
        print("{} strokes have been removed.".format(int(stroke_count * remove_ratio)))
        print("The produced sketch is stored in {}".format(output_folder))
        print(torchsketch())
	
    return stroke_count