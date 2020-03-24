import cairosvg
import os

def convert_svg_2_png(svg_url, output_folder = None):

    try:
        if output_folder is None:
            cairosvg.svg2png(url = svg_url, write_to = svg_url[:-4] + ".png")
        else:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            cairosvg.svg2png(url = svg_url, write_to = os.path.join(output_folder, (svg_url.split("/")[-1][:-4] + ".png")))
    except:
        print("{} threw an exception.".format(svg_url))