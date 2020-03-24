import cairosvg
import os

def convert_svg_2_pdf(svg_url, output_folder = None):

    try:
        if output_folder is None:
            cairosvg.svg2pdf(url = svg_url, write_to = svg_url[:-4] + ".pdf")
        else:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            cairosvg.svg2pdf(url = svg_url, write_to = os.path.join(output_folder, (svg_url.split("/")[-1][:-4] + ".pdf")))
    except:
        print("{} threw an exception.".format(svg_url))