from xml.dom import minidom
from torchsketch.utils.general_utils.torchsketch import *

def count_strokes_4_svg(svg_url, verbose = True):

	doc = minidom.parse(svg_url)
	path_strings = [path.getAttribute('d') for path in doc.getElementsByTagName('path')]
	doc.unlink()

	assert type(path_strings) == list

	if verbose == True:
		print("{} has {} storkes.".format(svg_url, len(path_strings)))
		print(torchsketch())

	return len(path_strings)