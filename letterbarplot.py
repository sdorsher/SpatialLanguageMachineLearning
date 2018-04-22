import numpy as np
from io import StringIO
import re
file = open("airports.dat.txt","r")
for line in file:
    re.split(',',line);
    print line
