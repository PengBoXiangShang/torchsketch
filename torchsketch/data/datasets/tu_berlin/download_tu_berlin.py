import wget
import os
import time
from torchsketch.utils.general_utils.print_banner import *
from torchsketch.utils.general_utils.torchsketch import *
from torchsketch.utils.general_utils.extract_files import *
from torchsketch.utils.general_utils.md5sum import *
from torchsketch.utils.general_utils.md5sum_table import *





def download_tu_berlin(output_folder = "./", remove_sourcefile = True):

    print(torchsketch())
    time.sleep(2)

    print_banner("Downloading started......")

    try:
        wget.download("http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_svg.zip", out = os.path.join(output_folder, "sketches_svg.zip"))
        wget.download("http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_png.zip", out = os.path.join(output_folder, "sketches_png.zip"))
        
    except Exception as e:
        print(e)
        return
	
    print("\n")
    print_banner("TU-Berlin is downloaded {}!".format('\033[32;1m'+ 'succesfully' + '\033[0m'))

    assert md5sum(os.path.join(output_folder, "sketches_svg.zip")) == MD5SUM_TABLE["sketches_svg.zip"], 'sketches_svg.zip md5 checksum error'
    assert md5sum(os.path.join(output_folder, "sketches_png.zip")) == MD5SUM_TABLE["sketches_png.zip"], 'sketches_png.zip md5 checksum error'

    
    print_banner("Md5 checksum passed {}!".format('\033[32;1m'+ 'succesfully' + '\033[0m'))

    print_banner("Extracting started......")

    try:
    	extract_files(file_name = os.path.join(output_folder, "sketches_svg.zip"), output_folder = output_folder, remove_sourcefile = remove_sourcefile)
    	extract_files(file_name = os.path.join(output_folder, "sketches_png.zip"), output_folder = output_folder, remove_sourcefile = remove_sourcefile)

    except Exception as e:
    	print(e)
    	return

	

    print_banner("TU-Berlin is extracted {}!".format('\033[32;1m'+ 'succesfully' + '\033[0m'))

    print(torchsketch())



