import re

from poetry_scraper import PoetryScraper

reader = iter(PoetryScraper())

for i in range(10):
    for title, poem in next(reader):
        f = open("test_poems/{0}.txt".format(re.sub(r'[^A-Za-z0-9]', '', title)), 'w')
        f.write(title+'\n')
        f.write(poem)
        f.close()
