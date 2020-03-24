import xml
from xml.dom import minidom
import os
import cairosvg
import imageio
from pdf2image import convert_from_path
from torchsketch.utils.svg_specific_utils.convert_svg_2_accumulative_svgs import *
from torchsketch.utils.general_utils.torchsketch import *

def convert_svg_2_gif(svg_url, output_folder, gif_fps = 3, verbose = True):

    
    convert_svg_2_accumulative_svgs(svg_url = svg_url, output_folder = output_folder, verbose = True)

    if verbose == True:
        print("Accumulative svgs have been produced.")
    
    accumulative_svg_list = os.listdir(output_folder)
    accumulative_svg_list.sort(key = lambda i : int(i[-10:-4]))

    gif_frames = list()

    for accum_svg in accumulative_svg_list:

        accum_svg_url = os.path.join(output_folder, accum_svg)

        converted_pdf_url = accum_svg_url[:-4] + ".pdf"

        cairosvg.svg2pdf(url = accum_svg_url, write_to = converted_pdf_url)

        gif_frames.append(convert_from_path(pdf_path = converted_pdf_url, dpi = 50)[0])

        os.remove(accum_svg_url)
        os.remove(converted_pdf_url)

    imageio.mimsave(output_folder + "/" + svg_url.split("/")[-1][:-4] + ".gif", gif_frames, fps = gif_fps)
    gif_frame_count = len(gif_frames)

    if verbose == True:
        print("Intermediate files have been well deleted.")
        print("The achieved gif file has {} frames in total.".format(gif_frame_count))
        print(torchsketch())

    return gif_frame_count



