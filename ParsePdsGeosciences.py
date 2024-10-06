import requests

from bs4 import BeautifulSoup
import pandas as pd
import time

import os

domain = "https://pds-geosciences.wustl.edu"
url = "/insight/urn-nasa-pds-insight_seis/data/xb/"

def getUrl(allHref):
    if allHref == []:
        return "Finish"
    allHrefTemp = []

    for aurl in allHref:
        responseTemp = requests.get(domain + aurl)
        soupTemp = BeautifulSoup(responseTemp.content, 'html.parser')
        hrefTemp = [ alink['href'] for alink in soupTemp.find_all('a', href=True) ]
        for elem in hrefTemp[1:len(hrefTemp)]:
            print(elem)
            if "." not in elem:
                allHrefTemp.append(elem)
                os.makedirs("." + elem)
            else:
                r = requests.get(domain+elem, allow_redirects=True)
                open('.'+elem, 'wb').write(r.content)
    getUrl(allHrefTemp)

urls = getUrl([url])
print(urls)