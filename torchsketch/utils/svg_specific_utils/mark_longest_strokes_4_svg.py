import xml
from xml.dom import minidom
import os
from torchsketch.utils.general_utils.reset_dir import *
from torchsketch.utils.general_utils.color_table import *
from torchsketch.utils.general_utils.torchsketch import *

def mark_longest_strokes_4_svg(svg_url, output_folder, mark_longest = True, color = "red", verbose = True):


    if color not in color_table.keys():
        print("Please choose a recognizable color.")
        return 0

    doc = minidom.parse(svg_url)
    paths = doc.getElementsByTagName("path")

    assert type(paths) == xml.dom.minicompat.NodeList
    stroke_count = len(paths)

    stroke_length_array = [len(path.getAttribute('d')) for path in paths]

    assert len(stroke_length_array) == stroke_count

    stroke_length_array.sort()

    if mark_longest:
        threshold_value = stroke_length_array[-1]
    else:
    	threshold_value = stroke_length_array[0]
    
    if verbose == True:
        print("{} has {} storkes.".format(svg_url, stroke_count))

    reset_dir(output_folder)


    svg_filename = svg_url.split("/")[-1][:-4]

    marked_count = 0

    try:

        for i in range(stroke_count):

            if len(doc.getElementsByTagName("path")[i].getAttribute("d")) == threshold_value:

                doc.getElementsByTagName("path")[i].setAttribute("stroke", color)
                marked_count += 1



        if mark_longest:
            output = open(output_folder + "/" + svg_filename + "_" + "marked_longest_stroke" + ".svg", "w")
        else:
        	output = open(output_folder + "/" + svg_filename + "_" + "marked_shortest_stroke" + ".svg", "w")
        doc.writexml(output)
        output.close()
        	
    except Exception as e:
        print(e)

    if verbose == True:
        print("Details are as follows.")
        print("{} strokes have been marked.".format(marked_count))
        print("The produced sketch is stored in {}".format(output_folder))
        print(torchsketch())
	
    return marked_count