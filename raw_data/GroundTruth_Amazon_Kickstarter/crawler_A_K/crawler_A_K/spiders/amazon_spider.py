import scrapy
import os
import requests
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings


def getUrls():
    with open('./amazon', 'rb') as file :
        array = []
        for line in file:
            array.append(line)
    return array


class AmazonSpider(scrapy.Spider):
    name = "amazon"

    custom_settings = {
        'DOWNLOAD_DELAY': '10',
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    }


    def start_requests(self):
        urls = getUrls()
        for url in urls:
            file = url.split("###V###")[1].strip()
            url = url.split("###V###")[0].strip()
            yield scrapy.Request(url=url, callback=lambda r, file=file: self.parse(r, file), dont_filter=True)

    def parse(self, response, file):
        directory = './data/kickstarter/creator/'
        filename = '%s.html' % file
        with open(os.path.join(directory, filename), 'wb') as f:
            f.write(response.url.strip())
            f.write(response.body)


print "Starting Crawl"
## Start crawling process and Spider
process = CrawlerProcess()
process.crawl(AmazonSpider)
process.start()
process.stop()