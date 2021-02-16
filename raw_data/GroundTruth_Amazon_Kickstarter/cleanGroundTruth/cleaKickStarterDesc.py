import os
from bs4 import BeautifulSoup
import csv
import re
import unicodedata
import sys
from textstat.textstat import textstat

reload(sys)
sys.setdefaultencoding("utf-8")

directory = "./kickstarter/desc/"
nameFiles = []
# target = open('./kickDesc.csv', 'wb')
# target = open('./kickTitle.csv', 'wb')
target = open('./kickProdDesc.csv', 'wb')
csvwriter = csv.writer(target, delimiter='`',quotechar='|', quoting=csv.QUOTE_MINIMAL)
# csvwriter.writerow(['Id','NumOfBackers','MoneyRaised','AmountPledged','FAQcount','UpdateCount', 'CommentsCount',
#                     'ProjectCAtegory', 'ProjectLocation','ofImages','ofVideos','ofDescText','backersLeastRewards', 'backersMaxRewards'])
# csvwriter.writerow(['Id', 'prodDescLength', 'prodDesc_ColemanLiau', 'noOfRewards', 'pledgeDescLength', 'pledge_ColemanLiau'])
csvwriter.writerow(['Id', 'KickstaterDesc'])


for root, dirs, files in os.walk(directory):
    for name in files:
        currentFile = os.path.join(root, name)
        print currentFile
        nameFiles.append(currentFile)
        soup = BeautifulSoup(open(currentFile))

        # Id
        id = name.split(".")[0]

        # Project Name
        if soup.find("div", {"class": "NS_project_profile__title"}):
            projectTitle = soup.find("div", {"class": "NS_project_profile__title"})
            print projectTitle.get_text().strip()

        # Project short desc
        # if soup.find("div", {"class": "NS_project_profiles__blurb"}):
        #     projectDesc = soup.find("div", {"class": "NS_project_profiles__blurb"})
        #     print projectDesc.get_text().strip()

        # Project creator
        # if soup.find("div", {"class": "creator-name"}):
        #     projectCreator = soup.find("div", {"class": "creator-name"})
        #     name = projectCreator.findAll("div")
        #     print name[0].get_text().strip()

        # Project stats
        # if soup.find("div", {"class": "NS_campaigns__spotlight_stats"}):
        #     projectStat = soup.find("div", {"class": "NS_campaigns__spotlight_stats"})
        #     backers = projectStat.find('b')
        #     money = projectStat.find('span')
        #     backers = backers.get_text().strip()
        #     backers = backers.replace(',','')
        #     backers = backers.split()[0].strip()
        #     backers = backers.strip()
        #     money = money.get_text().strip()
        #     money = money.replace(',', '')
        #     money = money.replace('$', '')
        #     money = re.sub("\D", "", money)
        #     money = money.strip()
        #     print backers, money

        # Project # FAQ # Comments # updates
        # if soup.find("div", {"class": "project-nav__links"}):
        #     faqCount = 0
        #     updatesCount = 0
        #     commentsCount = 0
        #     navData = soup.find("div", {"class": "project-nav__links"})
        #     faq = navData.find("a", {"class": "js-load-project-content js-load-project-faqs mx3 project-nav__link--faqs tabbed-nav__link type-14"})
        #     faqCount = faq.find("span", {"class": "count"})
        #     updates = navData.find("a", {"class": "js-load-project-content js-load-project-updates mx3 project-nav__link--updates tabbed-nav__link type-14"})
        #     updatesCount = updates.find("span", {"class": "count"})
        #     comments = navData.find("a", {"class": "js-load-project-comments js-load-project-content mx3 project-nav__link--comments tabbed-nav__link type-14"})
        #     commentsCount = comments.find("span", {"class": "count"})
        #     if faqCount:
        #         faqCount = faqCount.get_text().strip()
        #         faqCount = faqCount.replace(',', '')
        #         faqCount = faqCount.strip()
        #     if updatesCount:
        #         updatesCount = updatesCount.get_text().strip()
        #         updatesCount = updatesCount.replace(',', '')
        #         updatesCount = updatesCount.strip()
        #     if commentsCount:
        #         commentsCount = commentsCount.get_text().strip()
        #         commentsCount = commentsCount.replace(',', '')
        #         commentsCount = commentsCount.strip()
        #     if faqCount == None:
        #         faqCount = 0
        #     print faqCount, updatesCount, commentsCount

        # Product pledge amount
        # if soup.find("div", {"class": "col-right col-4 py3 border-left"}):
        #     pledge = soup.find("div", {"class": "col-right col-4 py3 border-left"})
        #     pledgeAmount = pledge.findAll("span", {"class": "money"})
        #     pledgeAmount = pledgeAmount[1]
        #     pledgeAmount = pledgeAmount.get_text().strip()
        #     # pledgeAmount = pledgeAmount.replace(',', '')
        #     # pledgeAmount = pledgeAmount.replace('$', '')
        #     # pledgeAmount = pledgeAmount.replace('', '')
        #     # pledgeAmount = pledgeAmount.replace('', '')
        #     # pledgeAmount = pledgeAmount.replace('CA ', '')
        #     pledgeAmount = re.sub("\D", "",pledgeAmount)
        #     print pledgeAmount

        # Product Category and Location
        # if soup.find("div", {"class": "NS_projects__category_location ratio-16-9"}):
        #     category = soup.find("div", {"class": "NS_projects__category_location ratio-16-9"})
        #     prodCategory = category.findAll("a", {"class": "grey-dark mr3 nowrap type-12"})
        #     category = prodCategory[1].get_text().strip()
        #     category = category.replace(',', ' ')
        #     category = category.strip()
        #     location = prodCategory[0].get_text().strip()
        #     location = location.split(',')[1].strip()
        #     location = location.strip()
        #     print category, location

        # Product Description, # number of images and videos in product description
        # # div class ="col col-8 description-container"
        # if soup.find("div", {"class": "col col-8 description-container"}):
        #     prodDesc = soup.find("div", {"class": "col col-8 description-container"})
        #     images = prodDesc.findAll("img", {"class": "fit"})
        #     video = prodDesc.findAll("video")
        #     prodDecsText = prodDesc.find("div", {"class": "full-description js-full-description responsive-media formatted-lists"}).get_text().strip()
        #     prodDecsText = prodDecsText.replace('\n', ' ')
        #     prodDecsText = ' '.join(prodDecsText.strip().split())
        #     prodDecsText = re.sub(r'[^\x00-\x7F]+', ' ', prodDecsText)
        #     if len(prodDecsText.split()) > 0:
        #         prodDecsText = " ".join(prodDecsText.split())
        #         prodDesc_ColemanLiau = textstat.coleman_liau_index(prodDecsText)
        #     else:
        #         prodDesc_ColemanLiau = 8
        #     prodDescLength = len(prodDecsText.split())
        #     print len(images), len(video), prodDescLength, prodDesc_ColemanLiau


        # Pledge div : Product rewards description
        # div class="col col-4"
        # if soup.find("div", {"class": "col col-4"}):
        #     rewardsDesc = soup.find("div", {"class": "col col-4"})
        #     rewardsDescText = rewardsDesc.findAll("li", {"class": "hover-group pledge--inactive pledge-selectable-sidebar"})
        #     rewardsBackersStat = rewardsDesc.findAll("span", {"class" : "pledge__backer-count"})
        #     if rewardsBackersStat[0]:
        #         backersLeastRewards = rewardsBackersStat[0].get_text().encode('utf-8').strip()
        #         backersLeastRewards = backersLeastRewards.encode('ascii','ignore')
        #         backersLeastRewards = backersLeastRewards.split()[0].strip()
        #         backersLeastRewards = backersLeastRewards.replace(',', '')
        #         backersLeastRewards = backersLeastRewards.strip()
        #         # print backersLeastRewards
        #             # , rewardsBackersStat[-1].get_text().strip()
        #     if rewardsBackersStat[len(rewardsBackersStat)-1]:
        #         backerMaxRewards = rewardsBackersStat[-1].get_text().encode('utf-8').strip()
        #         backerMaxRewards = backerMaxRewards.encode('ascii', 'ignore')
        #         backerMaxRewards = backerMaxRewards.split()[0].strip()
        #         backerMaxRewards = backerMaxRewards.replace(',', '')
        #         backerMaxRewards = backerMaxRewards.strip()
        #         # print backerMaxRewards
        #     noOfRewards = len(rewardsBackersStat)
        #     print noOfRewards
        #     # if rewardsBackersStat[1]:
        #     #     print rewardsBackersStat[1].get_text().strip()
        #     # if rewardsBackersStat[-2]:
        #     #     print rewardsBackersStat[-2].get_text().strip()
        #     # print rewardsDesc.get_text().strip()


        # Rewards Description
        # if soup.find("div", {"class": "col col-4"}):
        #     rewardsDesc = soup.find("div", {"class": "col col-4"})
        #     rewardsText = rewardsDesc.findAll("li", {"class":"hover-group pledge--inactive pledge-selectable-sidebar"})
        #     description = []
        #     text = []
        #     for rewards in rewardsText:
        #         rewardsDescription = rewards.find("div", {"class": "pledge__reward-description pledge__reward-description--expanded"}).get_text().strip()
        #         temp = ' '.join(rewardsDescription.strip().split())
        #         temp = re.sub(r'[^\x00-\x7F]+', ' ', temp)
        #         text.append(temp)
        #         desc = rewardsDescription.split()
        #         description.extend(desc)
        #         pledgeDescText = len(description)
        #     text = ' '.join(text)
        #     pledge_FleschReadingEase = textstat.flesch_reading_ease(text)
        #     pledge_GunningFog = textstat.gunning_fog(text)
        #     pledge_SmogIndex = textstat.smog_index(text)
        #     pledge_ColemanLiau = textstat.coleman_liau_index(text)
        #     pledge_fleschKincaidGrade = textstat.flesch_kincaid_grade(text)
        #     print pledgeDescText,",", pledge_ColemanLiau
        # SMOG
        # csvwriter.writerow([id,backers,money,pledgeAmount,faqCount,
        #                     updatesCount,commentsCount, category, location,
        #                     len(images), len(video), len(prodDecsText.split()), backersLeastRewards,
        #                     backerMaxRewards])
        # csvwriter.writerow([id, prodDescLength, prodDesc_ColemanLiau, noOfRewards, pledgeDescText, pledge_ColemanLiau])
        # csvwriter.writerow([id, projectTitle.get_text().strip()])
        csvwriter.writerow([id, prodDecsText.strip()])