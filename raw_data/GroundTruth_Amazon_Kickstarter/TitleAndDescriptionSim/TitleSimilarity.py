import re
import csv
import editdistance



nameFiles = []
target = open('./titleSimScore.csv', 'wb')
csvwriter = csv.writer(target, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
csvwriter.writerow(['Id','TitleSimScore'])


with open('./titleSimilarityIndiegogo.csv','rb') as f:
    reader = csv.reader(f, delimiter=',', )
    reader.next()
    for row in reader:
        id = row[0]

        # Kickstarter title cleaning
        kickTitle = row[1].lower()
        kickTitle = re.sub('[^A-Za-z0-9]+', ' ', kickTitle)

        # Amazon title cleaning
        amazonTitle = row[2].lower()
        amazonTitle = re.sub('[^A-Za-z0-9]+', ' ', amazonTitle)

        simScore = editdistance.eval(kickTitle, amazonTitle)
        print simScore,' , ', kickTitle,' , ', amazonTitle
        csvwriter.writerow([id,simScore])
