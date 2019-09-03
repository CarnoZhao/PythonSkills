import requests
from bs4 import BeautifulSoup
import pandas as pd
import LianJiaFunctions
import numpy as np
import re

def loadInformation():
    url = lambda page: 'https://bj.lianjia.com/zufang/pg' + str(page) + '/#contentList'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36'}
    columns = [
        'names',
        'district',
        'location',
        'xaxis',
        'yaxis',
        'roomspace',
        'orientation',
        'numrooms',
        'height',
        'floors',
        'owner',
        'postdate',
        'attrs',
        'price'
        ]
    return url, headers, columns

def getPos(href):
    url = 'https://bj.lianjia.com' + href
    result = requests.get(url, headers = headers)
    soup = BeautifulSoup(result.text, 'html.parser')
    raw = [x for x in soup.find_all('script') if 'longitude' in x.get_text()][0]
    pos = [x for x in raw.get_text().split('\n') if 'longitude' in x or 'latitude' in x]
    pos = [eval(re.findall(r'\d*\.\d*', x)[0]) for x in pos]
    return pos


def contentSpliter(content):
    title = content.find(class_ = 'content__list--item--title twoline')
    names = title.get_text().strip()
    href = title.find('a').attrs['href']
    pos = getPos(href)
    des = content.find(class_ = 'content__list--item--des').get_text().replace('/', '').split()
    brand = content.find(class_ = 'content__list--item--brand oneline').get_text().strip()
    time = content.find(class_ = 'content__list--item--time oneline').get_text().strip()
    attrs = content.find(class_ = 'content__list--item--bottom oneline')
    attrsTags = attrs.find_all('i')
    attrs = [tag.attrs['class'][0].split('--')[1] for tag in attrsTags]
    line = [names, pos, des, brand, time, attrs]
    return line

def pageSpliter(soup):
    lines = []
    contentList = soup.find_all(class_ = 'content__list--item--main')
    for content in contentList:
        line = contentSpliter(content)
        lines.append(line)
    return lines

def main():
    global headers
    url, headers, columns = loadInformation()
    data = []
    del columns
    for page in range(1, 101):
        print(page)
        result = requests.get(url(page), headers = headers)
        soup = BeautifulSoup(result.text, 'html.parser')
        lines = pageSpliter(soup)
        data.extend(lines)
    df = pd.DataFrame(data)
    df.to_csv('/home/tongxueqing/zhaox/codes/PythonSkills/LianJia/contents.csv', sep = '\t', index = False)
    print(len(df))

main()
