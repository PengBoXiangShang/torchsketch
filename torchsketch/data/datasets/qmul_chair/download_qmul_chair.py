import gdown
import os
import time
from torchsketch.utils.general_utils.print_banner import *
from torchsketch.utils.general_utils.torchsketch import *
from torchsketch.utils.general_utils.extract_files import *
from torchsketch.utils.general_utils.md5sum import *
from torchsketch.utils.general_utils.md5sum_table import *





def download_qmul_chair(output_folder = "./", remove_sourcefile = True):

    print(torchsketch())
    time.sleep(2)

    print_banner("Downloading started......")

    try:
        gdown.download("https://drive.google.com/uc?id=1xlBhTJQwtssi8oJHbYbqVer8ZmyEG3wq", os.path.join(output_folder, "chairs.zip"), quiet=False)
        
    except Exception as e:
        print(e)
        return
	
    print("\n")
    print_banner("QMUL chair is downloaded {}!".format('\033[32;1m'+ 'succesfully' + '\033[0m'))

    assert md5sum(os.path.join(output_folder, "chairs.zip")) == MD5SUM_TABLE["chairs.zip"], 'chairs.zip md5 checksum error'

    
    print_banner("Md5 checksum passed {}!".format('\033[32;1m'+ 'succesfully' + '\033[0m'))

    print_banner("Extracting started......")

    try:
        extract_files(file_name = os.path.join(output_folder, "chairs.zip"), output_folder = output_folder, remove_sourcefile = remove_sourcefile)

    except Exception as e:
    	print(e)
    	return

	

    print_banner("QMUL chair is extracted {}!".format('\033[32;1m'+ 'succesfully' + '\033[0m'))

    print(torchsketch())



