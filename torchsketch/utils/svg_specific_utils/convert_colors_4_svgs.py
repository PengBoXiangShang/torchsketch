from torchsketch.utils.svg_specific_utils.convert_colors_4_svg import *
from torchsketch.utils.general_utils.torchsketch import *

def convert_colors_4_svgs(svg_url_list, output_folder, colors = ["red", ], verbose = True):

    for svg_url in svg_url_list:
        convert_colors_4_svg(svg_url, output_folder, colors, verbose)



    print(torchsketch())