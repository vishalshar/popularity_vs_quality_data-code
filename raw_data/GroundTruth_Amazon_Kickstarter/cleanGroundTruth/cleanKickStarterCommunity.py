import os
from bs4 import BeautifulSoup
import csv


directory = "./kickstarter/Community/"
nameFiles = []
# target = open('./kickCreator.csv', 'wb')
# target = open('./kickCommunity.csv', 'wb')
# csvwriter = csv.writer(target, delimiter='`',quotechar='|', quoting=csv.QUOTE_MINIMAL)
# csvwriter.writerow(['Id','New Backers','Returning Backers'])

output = "./community/"

for root, dirs, files in os.walk(directory):
    for name in files:
        currentFile = os.path.join(root, name)
        print currentFile
        nameFiles.append(currentFile)
        soup = BeautifulSoup(open(currentFile))

        target = open('./community/%s.csv'%id, 'wb')
        csvwriter_comm = csv.writer(target, delimiter='`',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter_comm.writerow(['left_primary', 'left_secondary' ,'right_text'])
        print target

        # Id
        id = name.split(".")[0]

        # New Backers
        if soup.find("div", {"class": "new-backers"}):
            backers = soup.find("div", {"class": "new-backers"})
            new_backers = backers.find("div", {"class": "count"})
            new_backers = new_backers.get_text().strip()
            print new_backers
            
        # Return Backers
        if soup.find("div", {"class": "existing-backers"}):j
            backers = soup.find("div", {"class": "existing-backers"})
            return_backers = backers.find("div", {"class": "count"})
            return_backers = return_backers.get_text().strip()
            print s


        if soup.find("div", {"class": "community-section__locations_cities"}):
            text = soup.findAll("div", {"class": "location-list__item js-location-item"})
            for t in text:
                left_secondary_text = ''
                left_primary_text = ''
                if t.find("div",{"class":"primary-text js-location-primary-text"}):
                    left_primary_text = t.find("div",{"class":"primary-text js-location-primary-text"}).get_text().strip()
                if t.find("div",{"class":"secondary-text js-location-secondary-text"}):
                    left_secondary_text = t.find("div",{"class":"secondary-text js-location-secondary-text"}).get_text().strip()
                right_text = t.find("div",{"class":"right"}).get_text().strip()
                print left_primary_text, left_secondary_text, right_text
                csvwriter_comm.writerow([left_primary_text.encode('utf-8').strip(), left_secondary_text.encode('utf-8').strip(), right_text.encode('utf-8').strip()])
                # csvwriter_comm.quit()
        # csvwriter.writerow([id, new_backers, return_backers])

# div class verified pb2 border-bottom f5
# print countTwitter
