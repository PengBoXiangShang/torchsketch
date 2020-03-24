import os
from pyunpack import Archive



def extract_files(file_name, output_folder, remove_sourcefile = True):

    try:
        Archive(file_name).extractall(output_folder)
    except Exception as e:
        print(e)

    print("{} has been extracted succesfully!".format(file_name))

    if remove_sourcefile == False:
        return

    try:
        os.remove(file_name)
    except Exception as e:
        print(e)

    print("Source file has been removed succesfully!")
    
