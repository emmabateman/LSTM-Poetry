#read the entire collection of poetry found on poetryfoundation.org

from bs4 import BeautifulSoup
import requests
import re

BROWSE_URL = "https://www.poetryfoundation.org/poems/browse?page={0}"
POEM_URL = "https://www.poetryfoundation.org/poetrymagazine/poems/*"
LINE_STYLE = "text-indent: -1em; padding-left: 1em;"
NUM_PAGES=2300

def get_poems(page):
    poems = []
    soup = BeautifulSoup(requests.get(BROWSE_URL.format(page)).text, features="lxml")
    links = soup.find_all(href=re.compile(POEM_URL))
    if len(links) == 0:
        more_poety = False
    else:
        for l in links:
            title = l.text
            soup = BeautifulSoup(requests.get(l["href"]).text, features="lxml")
            poem_text = '\n'.join([item.text for item in soup.find_all(style="text-indent: -1em; padding-left: 1em;")]).replace('\r', '')
            if len(poem_text) > 0:
                poems.append((title, poem_text))

    return poems

class PoetryScraper:
    def __iter__(self):
        self.page = 1
        return self
    def __next__(self):
        poems = get_poems(self.page)
        self.page += 1
        if self.page > NUM_PAGES:
            return None
        return poems
