import os
from bs4 import BeautifulSoup
import csv


directory = "./kickstarter/creator/"
nameFiles = []
# target = open('./kickCreator.csv', 'wb')
target = open('./kickCreatorTwitter.csv', 'wb')
csvwriter = csv.writer(target, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
# csvwriter.writerow(['Id','backedProjects','createdProjects','backerLocation','websiteLinked', 'collaborator',
#                     'accountVerified', 'facebookConnected', 'facebookFriends'])
csvwriter.writerow(['Id','twitterUserName'])


countTwitter = 0
for root, dirs, files in os.walk(directory):
    for name in files:
        currentFile = os.path.join(root, name)
        print currentFile
        nameFiles.append(currentFile)
        soup = BeautifulSoup(open(currentFile))

        # Id
        id = name.split(".")[0]

        # Creator Name
        # if soup.find("h1", {"class": "f2 normal mb2"}):
        #     name = soup.findAll("h1", {"class": "f2 normal mb2"})
        #     print name[0].get_text().strip()

        # Creator Location
        # if soup.find("p", {"class": "f5 bold mb0"}):
        #     location = soup.findAll("p", {"class": "f5 bold mb0"})
        #     location = location[0].get_text().strip()
        #     location = location.split(',')[1].strip()
        #     print location

        # Creator Bio
        # if soup.find("div", {"class": "readability"}):
        #     bio = soup.findAll("div", {"class": "readability"})
        #     print bio[0].get_text().strip()

        # # of projects created and backed
        # if soup.find("a", {"class": "green-dark bold remote_modal_dialog"}):
        #     projectsCreated = soup.findAll("a", {"class": "green-dark bold remote_modal_dialog"})
        #     backed = ''
        #     created = ''
        #     if len(projectsCreated) == 2:
        #         created = projectsCreated[0].get_text().strip().split(" ")[0].strip()
        #         backed = projectsCreated[1].get_text().strip().split(" ")[0].strip()
        #     if len(projectsCreated) == 1:
        #         created = 1
        #         backed = projectsCreated[0].get_text().strip().split(" ")[0].strip()
        #     print backed, created

        # # of websites linked
        if soup.find("ul", {"class": "links list f5 bold"}):
            websites = soup.find("ul", {"class": "links list f5 bold"})
            countWeb = websites.findAll('li')
            noOfWeb = len(countWeb)
            writeInCSV = True
            for web in countWeb:
                website = web.find('a', href=True)
                if "twitter" in website['href']:
                    countTwitter = countTwitter + 1
                    print website['href'], id
                    csvwriter.writerow([id, website['href']])
                    writeInCSV = False
            if writeInCSV:
                csvwriter.writerow([id, ''])
        else:
            csvwriter.writerow([id, ''])
        # print countTwitter
            # print noOfWeb, countWeb

        # Collaborators count
        # if soup.find("div", {"class": "pt3 pt7-sm mobile-hide row"}):
        #     collaborators = soup.find("div", {"class": "pt3 pt7-sm mobile-hide row"})
        #     collaboratorsCount = collaborators.findAll("div", {"class": "flag col col-4 mb3"})
        #     collaboratorsCount = len(collaboratorsCount)
        #     print collaboratorsCount

        # account verified
        # if soup.find("div", {"class": "verified pb2 border-bottom f5"}):
        #     verified = 1
        #     print verified
        # else:
        #     verified = 0

        # Facebook Connected
        # if soup.find("div", {"class": "facebook py2 border-bottom f5"}):
        #     facebook = soup.find("div", {"class": "facebook py2 border-bottom f5"})
        #     facebookCon = 1
        #     facebookFriends = facebook.get_text().strip().split()[0].strip()
        #     facebookFriends = facebookFriends.replace(",","")
        #     if facebookFriends == 'Not':
        #         facebookCon = 0
        #         facebookFriends = 0
        #     print facebookCon, facebookFriends
        # csvwriter.writerow([id, backed, created,location,countWeb,collaboratorsCount,verified,facebookCon, facebookFriends ])

# div class verified pb2 border-bottom f5
print countTwitter