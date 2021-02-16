import os
from bs4 import BeautifulSoup
import csv
import re
from nltk.tokenize import RegexpTokenizer


tokenizer = RegexpTokenizer(r'\w+')
directory = "./kickstarter/comments/"
nameFiles = []
target = open('./kickComments.csv', 'wb')
csvwriter = csv.writer(target, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
csvwriter.writerow(['Id','countCreator','countSuperBacker','commentsText'])
outputFolder = "./kickstarter/cleanComments/"


# Cleaning Data
def clean(line):
    text = line.encode('utf-8').strip()
    # text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    # text = tokenizer.tokenize(text)
    # text = [w for w in text if not w in stops]
    # text = [snowball_stemmer.stem(word) for word in text]
    # text = [word for word in text if len(word) > 3]
    # text = ' '.join(word for word in text)
    # print text
    return text


def get_text(comment):
    result = ''
    for text in comment:
        result += ' '+text.get_text().strip()
    return result.replace('\n',' ')


for root, dirs, files in os.walk(directory):
    for name in files:
        currentFile = os.path.join(root, name)
        print currentFile
        nameFiles.append(currentFile)
        soup = BeautifulSoup(open(currentFile))
        commentsText = []
        countCreator = 0
        countSuperBacker = 0

        # Id
        id = name.split(".")[0]

        # open file to write
        targetFile = open(os.path.join(outputFolder, id+'.txt'), 'w')

        # All Comments
        if soup.find("ol", {"class": "comments"}):
            commentsDesc = soup.find("ol", {"class": "comments"})
            comments = commentsDesc.findAll("li")
            for comment in comments:
                text = comment.findAll("p")
                text = get_text(text)
                commentsText.append(text.strip())
                targetFile.write(clean(text.strip()))
                targetFile.write('\n')
                if comment.find("span", {"class": "repeat-creator-badge relative"}):
                    countCreator += 1
                elif comment.find("span", {"class": "creator-badge relative"}):
                    countCreator += 1
                if comment.find("div", {"class":"superbacker-badge tipsy_s"}):
                    countSuperBacker += 1
        print countCreator, countSuperBacker, len(commentsText)
        targetFile.close()
        #, commentsText
        csvwriter.writerow([id, countCreator, countSuperBacker, len(commentsText)])
