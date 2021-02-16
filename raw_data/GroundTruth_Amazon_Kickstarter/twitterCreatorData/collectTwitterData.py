import json
from twitter import *
import csv


# Target CSV file to write
target = open('./KicktwitterData.csv', 'wb')
csvwriter = csv.writer(target, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
csvwriter.writerow(['Id','followers_count', 'friends_count', 'public_list_count', 'tweets_count', 'likes_count'])



# Set up twitter API access
ACCESS_TOKEN = '1231144110-xie9x4GCwHMDt5UtoVhwEwRDsSSlP3g7GQw2uVz'
ACCESS_SECRET = '6NUmYHwE7yaZJ60igu8A2DdcijLy2WkmLdD2AH6xuofeT'
CONSUMER_KEY = '4H57xRidOn1FERST1PxGSlmlr'
CONSUMER_SECRET = 'NCzEuTWTjkNwfUMMOwEadR1wZfZPXIcssG9vwP5etQywEJ36Cw'
twitter = Twitter(auth = OAuth(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET))



# Read CSV with Id and User Profile Name
with open('./kickCreatorTwitter.csv','rb') as f:
    reader = csv.reader(f, delimiter=',', )
    reader.next()
    for row in reader:
        id = row[0]
        # Ex. https://twitter.com/nondainc
        profile = row[1]
        if len(profile) > 10:
            print profile
            profileName = profile.split('/')[-1]
            try:
                results = twitter.users.lookup(screen_name=profileName)
                for user in results:
                    followers_count = user["followers_count"]
                    friends_count = user["friends_count"]
                    public_list_count = user["listed_count"]
                    tweets_count = user["statuses_count"]
                    likes_count = user["favourites_count"]
                    print followers_count, friends_count, public_list_count, tweets_count, likes_count
                    csvwriter.writerow([id, followers_count, friends_count, public_list_count, tweets_count, likes_count])
            except:
                # & user deleted there profile
                print "User does not exist"
        else:
            csvwriter.writerow([id, 0, 0, 0, 0, 0])
