import numpy as np
import pandas as pd

colorDic = {
    'red':                  '#FF0000',
    'orgred':            '#FF3300',
    'orange':         '#ff6600',
    'orgyellow':                 '#ff9900',
    'yellow':           '#ffff00',
    'yelgreen':                '#99ff00',
    'green':                '#00ff00',
    'bluegreen':               '#00ffff',
    'blue':                '#0000ff',
    'bluepurple':                '#6600ff',
    'purple':                '#ff00ff',
    'purplered':                '#ff0066',
}

colors = pd.Series(colorDic)
colorNum = len(colors)

def randColors(n):
    """
    get n colors randomly
    """
    rand = np.random.randint(0,12,n)
    return np.array(colors[rand])

def fixedColors(n):
    """
    get n fixed colors
    n should less than 12
    """
    if n > 12:
        print "you select too many colors"
        return None
    sel = range(0,12,12/n)
    return np.array(colors[sel])

if __name__ == '__main__':
    print fixedColors(12)
        
