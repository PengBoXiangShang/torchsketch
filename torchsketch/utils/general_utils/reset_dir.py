import os
import shutil

def reset_dir(folder_path):

	if not os.path.exists(folder_path):
		os.mkdir(folder_path)
	else:
		shutil.rmtree(folder_path)
		os.mkdir(folder_path)