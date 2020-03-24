import gdown
import os
import time
from torchsketch.utils.general_utils.print_banner import *
from torchsketch.utils.general_utils.torchsketch import *
from torchsketch.utils.general_utils.extract_files import *
from torchsketch.utils.general_utils.md5sum import *
from torchsketch.utils.general_utils.md5sum_table import *





def download_sketchy(output_folder = "./", remove_sourcefile = True):

    print(torchsketch())
    time.sleep(2)

    print_banner("Downloading started......")

    try:
        gdown.download("https://drive.google.com/uc?id=0B7ISyeE8QtDdbUpYWV8tcFJlY2M", os.path.join(output_folder, "sketches-06-04.7z"), quiet=False)
        gdown.download("https://drive.google.com/uc?id=0B7ISyeE8QtDdTjE1MG9Gcy1kSkE", os.path.join(output_folder, "rendered_256x256.7z"), quiet=False)
        gdown.download("https://drive.google.com/uc?id=0B7ISyeE8QtDdaFhqeTZiNVBYZjA", os.path.join(output_folder, "info-06-04.7z"), quiet=False)
        
    except Exception as e:
        print(e)
        return
	
    print("\n")
    print_banner("Sketchy is downloaded {}!".format('\033[32;1m'+ 'succesfully' + '\033[0m'))

    assert md5sum(os.path.join(output_folder, "sketches-06-04.7z")) == MD5SUM_TABLE["sketches-06-04.7z"], 'sketches-06-04.7z md5 checksum error'
    assert md5sum(os.path.join(output_folder, "rendered_256x256.7z")) == MD5SUM_TABLE["rendered_256x256.7z"], 'rendered_256x256.7z md5 checksum error'
    assert md5sum(os.path.join(output_folder, "info-06-04.7z")) == MD5SUM_TABLE["info-06-04.7z"], 'info-06-04.7z md5 checksum error'

    
    print_banner("Md5 checksum passed {}!".format('\033[32;1m'+ 'succesfully' + '\033[0m'))

    print_banner("Extracting started......")

    try:
        extract_files(file_name = os.path.join(output_folder, "sketches-06-04.7z"), output_folder = output_folder, remove_sourcefile = remove_sourcefile)
        extract_files(file_name = os.path.join(output_folder, "rendered_256x256.7z"), output_folder = output_folder, remove_sourcefile = remove_sourcefile)
        extract_files(file_name = os.path.join(output_folder, "info-06-04.7z"), output_folder = output_folder, remove_sourcefile = remove_sourcefile)

    except Exception as e:
    	print(e)
    	return

	

    print_banner("Sketchy is extracted {}!".format('\033[32;1m'+ 'succesfully' + '\033[0m'))

    print(torchsketch())



