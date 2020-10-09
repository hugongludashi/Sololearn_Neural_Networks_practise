a = input("Please enter name of the pacakge or packages you want to instlall. Please sperate each package name with ',':") 

import os

l = a.strip().split(',')

for i in l:
    os.system('python -m pip install ' +i)
