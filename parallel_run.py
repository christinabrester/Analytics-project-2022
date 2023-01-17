# To run MLP_nowcasting.py in parallel

from sys import executable
from subprocess import Popen, CREATE_NEW_CONSOLE

phases = ['L1', 'L2', 'L3']
outputs = ['TDU', 'ITD', 'Q1act']


for output in outputs:
    for phase in phases:
        Popen([executable, 'MLP_nowcasting.py', phase, output], creationflags=CREATE_NEW_CONSOLE)
