import re
import csv
import editdistance
import csv
import itertools


f1 = open("./amazonDesc.csv")
f2 = open("./kickProdDesc.csv")

csv_f1 = csv.reader(f1, delimiter= '`')
csv_f2 = csv.reader(f2, delimiter= '`')



for row1, row2 in itertools.izip(csv_f1, csv_f2):
    print row1[0], row2[0]