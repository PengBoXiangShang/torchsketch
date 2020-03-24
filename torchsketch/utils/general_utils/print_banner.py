
def print_banner(content):
	
    screen_width = 60
    text_width = len(content.replace("\x1b[32;1m", "").replace("\x1b[0m", ""))
    box_width = text_width + 6
    left_margin = (screen_width - box_width)//2
    inner_margin = (box_width-text_width-2)//2
    print
    print (' ' * left_margin + '+' + '-' * (box_width-2)    + '+')
    print (' ' * left_margin + '|' + ' ' * (box_width-2)    + '|')
    print (' ' * left_margin + '|' + ' ' * inner_margin     + content + inner_margin * ' '+'|')
    print (' ' * left_margin + '|' + ' ' * (box_width-2)    + '|')
    print (' ' * left_margin + '+' + '-' * (box_width-2)    + '+')
    print 