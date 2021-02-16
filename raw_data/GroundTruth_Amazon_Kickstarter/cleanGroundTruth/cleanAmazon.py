import os
import sys
from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import csv
import re

reload(sys)
sys.setdefaultencoding("utf-8")


directory = "./amazon/"
nameFiles = []
# target = open('./amazon.csv', 'wb')
# target = open('./amazonTitle.csv', 'wb')
# target = open('./amazonDesc.csv', 'wb')
# target = open('./amazonInfo.csv', 'wb')
target = open('./amazonNumRating.csv', 'wb')

csvwriter = csv.writer(target, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
# csvwriter.writerow(['Id','AmazonCategory','AmazonPrice','AmazonImagesVideo','AmazonProdDescLength','rating'])
# csvwriter.writerow(['Id','AmazonTitle'])
# csvwriter.writerow(['Id','AmazonDescription'])
csvwriter.writerow(['Id','Number of Rating'])


for root, dirs, files in os.walk(directory):
    for name in files:
        currentFile = os.path.join(root, name)
        print currentFile
        nameFiles.append(currentFile)
        soup = BeautifulSoup(open(currentFile))

        # Id
        id = name.split(".")[0]

        # Product category
        # if soup.find("div", {"id": "wayfinding-breadcrumbs_feature_div"}):
        #     productCategory = soup.find("div", {"id": "wayfinding-breadcrumbs_feature_div"})
        #     productCategory = productCategory.findAll('li')
        #     # productCategory = soup.findAll("span", {"class": "a-list-item"})
        #     productCategory = productCategory[0].get_text().strip()
        #     productCategory = re.sub('\W+',' ', productCategory )
        #     print productCategory
        # elif soup.find("div", {"class": "nav-search-facade"}):
        #     productCategory = soup.find("div", {"class": "nav-search-facade"})
        #     productCategory = productCategory.find("span", {"class": "nav-search-label"})
        #     productCategory = productCategory.get_text().strip()
        #     print productCategory

        # Product price
        # if soup.find("span", {"id": "priceblock_ourprice"}):
        #     productPrice = soup.findAll("span", {"id": "priceblock_ourprice"})
        #     productPrice = productPrice[0].get_text().strip().replace('$', '')
        #     productPrice = productPrice.replace(',','')
        #     print productPrice
        # elif soup.find("span", {"class": "oneclick dv-action-dark dv-action-purchase-button js-purchase-button dv-oneclick"}):
        #     productPrice = soup.findAll("span", {"class": "oneclick dv-action-dark dv-action-purchase-button js-purchase-button dv-oneclick"})
        #     productPrice = productPrice[0].get_text()
        #     productPrice = productPrice.split("$")[1].strip()
        #     print productPrice
        # else:
        #     if soup.find("div", {"class":"a-section a-spacing-none a-padding-none DigitalBuyButtonSection DigitalBuyButtonBuyBoxSection"}):
        #         productPrice = soup.find("div", {"class":"a-section a-spacing-none a-padding-none DigitalBuyButtonSection DigitalBuyButtonBuyBoxSection"})
        #         productPrice = productPrice.get_text()
        #         productPrice = productPrice.split("$")[1].strip()
        #         print productPrice

        # Product title
        # if soup.find("div", {"id": "title_feature_div"}):
        #     title = soup.find("div", {"id": "title_feature_div"})
        #     if title.find("span", {"id": "productTitle"}):
        #         title = title.find("span", {"id": "productTitle"})
        #         print title.get_text().strip()
        #     else:
        #         print title.get_text().strip()

        # Product rating
        # if soup.find('div', {'id': 'averageCustomerReviews'}):
        #     avgRating = soup.find('div', {'id': 'averageCustomerReviews'})
        #     if avgRating.find("span", {"class" : "a-icon-alt"}):
        #         rating = avgRating.find("span", {"class" : "a-icon-alt"})
        #         rating = rating.get_text().strip().split()[0]
        #         print rating

        # Product ship and sold
        # if soup.find("div", {"id": "merchant-info"}):
        #     productShip_Sold = soup.findAll("div", {"id": "merchant-info"})
        #     # print type(productShip_Sold[0].get_text().strip()+"")
        #     ship_sold = productShip_Sold[0].get_text().strip()
        #     ship_sold = " ".join(ship_sold.split())
        #     print ship_sold
        # elif soup.find("span", {"id": "merchant-info"}):
        #     productShip_Sold = soup.findAll("span", {"id": "merchant-info"})
        #     ship_sold = productShip_Sold[0].get_text().strip()
        #     ship_sold = " ".join(ship_sold.split())
        #     print ship_sold

        # Product short description
        # if soup.find("div", {"id": "featurebullets_feature_div"}):
        #     productFeatureDesc = soup.find("div", {"id": "featurebullets_feature_div"})
        #     shortDesc = productFeatureDesc.findAll("span", {"class": "a-list-item"})
        #     for desc in shortDesc:
        #         print desc.get_text().strip()

        # Product description
        # if soup.find("div", {"id": "productDescription"}):
        #     productDesc = soup.findAll("div", {"id": "productDescription"})
        #     desc = " ".join(productDesc[0].get_text().strip().split())
        #     print desc

        # Product Information
        # if soup.find("table", {"id": "productDetails_detailBullets_sections1"}):
        #     productInfo = soup.find("table", {"id": "productDetails_detailBullets_sections1"})
        #     records = []  # To store all product information records
        #     for row in productInfo.findAll('tr'):
        #         firstCol = row.find('th').get_text().strip()
        #         secondCol = row.find('td').get_text().strip()
        #         # print firstCol, secondCol
        #         if firstCol != 'Customer Reviews':
        #             record = '%s;%s' % (firstCol, secondCol)  # store the record with a ';' between prvy and druhy
        #             records.append(record)
        #     # print records
        #     # for record in records:
        #     #     print record
        # elif soup.find("div", {"id": "prodDetails"}):
        #     productInfo = soup.find("div", {"id": "prodDetails"})
        #     productDetails = productInfo.find('table')
        #     records = []
        #     for row in productDetails.findAll('tr'):
        #         col = row.findAll('td')
        #         firstCol = col[0].get_text().strip()
        #         secondCol = col[1].get_text().strip()
        #         # print firstCol, secondCol
        #         if firstCol != 'Customer Reviews':
        #             record = '%s;%s' % (firstCol,
        #                             secondCol)  # store the record with a ';' between prvy and druhy
        #             records.append(record)
        #     # for record in records:
        #     #     print record
        # elif soup.find("div", {"id": "detail-bullets"}):
        #     productInfo = soup.find("div", {"id": "detail-bullets"})
        #     productDetails = productInfo.find('table')
        #     records = []
        #     for row in productDetails.findAll('li'):
        #         row = row.get_text().strip()
        #         if row.split(':')[0] != 'Average Customer Review':
        #             if row.split(':')[0] != 'Amazon Best Sellers Rank':
        #                 records.append(row)
        #     # for record in records:
        #     #     print record
        # print len(records)-1
        # # of images or videos
        # if soup.find("div", {"id": "altImages"}):
        #     images = soup.find("div", {"id": "altImages"})
        #     countImgVid = len(images.find_all("li")) - 1
        #     print countImgVid

        # Product number of people rating
        if soup.find('div', {'id': 'averageCustomerReviews'}):
            avgRating = soup.find('div', {'id': 'averageCustomerReviews'})
            if avgRating.find("span", {"class" : "a-size-base"}):
                num_rating = avgRating.find("span", {"class" : "a-size-base"})
                num_rating = num_rating.get_text().strip().split(' ')[0].replace(',','')
                print num_rating

        csvwriter.writerow([id, num_rating])

