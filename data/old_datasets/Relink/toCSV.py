from __future__ import print_function

import csv
import re
from os import walk
from pdb import set_trace
from scipy.io.arff import loadarff


def do():
    print('Doing')
    for (a, b, c) in walk('./'):
        pass

    for file in c:
        if 'arff' in file:
            # set_trace()
            print(a + '/' + file)
            arff = loadarff(a + '/' + file)
            tags = [aa for aa in arff[1]]
            header = ['$' + h for h in tags[:-1]]
            header.append('$>Defects')
            name = []
            with open(a + '/' + re.sub('.arff|[ ]', '', file) + '.csv', 'w+') as csvfile:
                writer = csv.writer(csvfile, delimiter=',', quotechar='|')
                writer.writerow(header)
                for body in arff[0].tolist():
                    body=[b for b in body]
                    # set_trace()
                    if body[-1]=='TRUE':
                        writer.writerow(body[:-1]+[1])
                    else:
                        writer.writerow(body[:-1]+[0])

if __name__ == '__main__':
    do()
