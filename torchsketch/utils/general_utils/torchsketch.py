from pyfiglet import Figlet
from pyfiglet import print_figlet
import time

def torchsketch():
    custom_fig = Figlet(font='big')
    torchsketch_banner = custom_fig.renderText('TorchSketch')
    
    return "\n" + torchsketch_banner



def torchsketch_color():
    colors="38;157;127:"
    print_figlet("TorchSketch", font="big", colors=colors)
    time.sleep(3)
