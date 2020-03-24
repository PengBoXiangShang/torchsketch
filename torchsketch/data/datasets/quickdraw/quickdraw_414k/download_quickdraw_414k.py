import gdown
import os
import time
from torchsketch.utils.general_utils.print_banner import *
from torchsketch.utils.general_utils.torchsketch import *
from torchsketch.utils.general_utils.extract_files import *
from torchsketch.utils.general_utils.md5sum import *
from torchsketch.utils.general_utils.md5sum_table import *





def download_quickdraw_414k(output_folder = "./", remove_sourcefile = True):

    print(torchsketch())
    time.sleep(2)

    print_banner("Downloading started......")

    try:
        gdown.download("https://drive.google.com/uc?id=1q933KpmJGkfStgIbwMfgfls1_1ZJVFyd", os.path.join(output_folder, "picture_files.tar.gz"), quiet=False)
        gdown.download("https://drive.google.com/uc?id=1Vrf1ouhtWYJp4XKa6jestLGY3aVlxfLC", os.path.join(output_folder, "coordinate_files.tar.gz"), quiet=False)
        
    except Exception as e:
        print(e)
        return
	
    print("\n")
    print_banner("quickdraw_414k is downloaded {}!".format('\033[32;1m'+ 'succesfully' + '\033[0m'))

    assert md5sum(os.path.join(output_folder, "picture_files.tar.gz")) == MD5SUM_TABLE["picture_files.tar.gz"], 'picture_files.tar.gz md5 checksum error'
    assert md5sum(os.path.join(output_folder, "coordinate_files.tar.gz")) == MD5SUM_TABLE["coordinate_files.tar.gz"], 'coordinate_files.tar.gz md5 checksum error'

    
    print_banner("Md5 checksum passed {}!".format('\033[32;1m'+ 'succesfully' + '\033[0m'))

    print_banner("Extracting started......")

    try:
        extract_files(file_name = os.path.join(output_folder, "picture_files.tar.gz"), output_folder = output_folder, remove_sourcefile = remove_sourcefile)
        extract_files(file_name = os.path.join(output_folder, "coordinate_files.tar.gz"), output_folder = output_folder, remove_sourcefile = remove_sourcefile)

    except Exception as e:
    	print(e)
    	return

	

    print_banner("quickdraw_414k is extracted {}!".format('\033[32;1m'+ 'succesfully' + '\033[0m'))

    print(torchsketch())



