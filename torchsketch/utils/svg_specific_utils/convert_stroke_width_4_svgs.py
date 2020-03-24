from torchsketch.utils.svg_specific_utils.convert_stroke_width_4_svg import *
from torchsketch.utils.general_utils.torchsketch import *

def convert_stroke_width_4_svgs(svg_url_list, output_folder, stroke_width = 6, verbose = True):

    for svg_url in svg_url_list:
        convert_stroke_width_4_svg(svg_url, output_folder, stroke_width, verbose)

    print(torchsketch())
