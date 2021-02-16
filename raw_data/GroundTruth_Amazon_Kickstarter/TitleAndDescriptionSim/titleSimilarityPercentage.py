import re
import csv
import jellyfish
import pandas as pd
import sys

reload(sys)
sys.setdefaultencoding('utf8')


with open('./kickTitleTest.csv','rb') as f:
    # reader = csv.reader(f, delimiter='`', )
    titles = pd.read_csv(f, delimiter='`')
    print type(titles['KickstaterTitle'])
    print type(titles['AmazonTitle'])
    # listofTitles = list(reader)

    # print listofTitles

    count = 0
    for index_kick, kick_title in  titles['KickstaterTitle'].iteritems():
        max = 0
        loc = 0
        for index_amz, amz_title in titles['AmazonTitle'].iteritems():
            score = jellyfish.jaro_distance(unicode(kick_title), unicode(amz_title))
            if score > max:
                print (unicode(kick_title), unicode(amz_title))
                max = score
                loc = index_amz
        print max, loc
        if index_kick == loc:
            count = count + 1
    print count
