from torchsketch.utils.svg_specific_utils.convert_svg_2_png import *
from torchsketch.utils.general_utils.torchsketch import *

def convert_svgs_2_pngs(svg_url_list, output_folder = None):

    for svg_url in svg_url_list:
        convert_svg_2_png(svg_url, output_folder)


    print(torchsketch())