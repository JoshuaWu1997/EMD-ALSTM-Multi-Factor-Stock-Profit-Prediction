from bs4 import BeautifulSoup
import re
import numpy as np

with open("factor.html", "rb") as f:
    html = f.read().decode("utf8")
    f.close()

soup = BeautifulSoup(html, "html.parser").prettify()
soup = soup.split('</li>')
contents = []
factor_list = dict()

for content in soup[1:-1]:
    temp = re.split('\n|\"', content)
    title = re.split('\s|（|）|\(|\)', temp[9])
    if temp[2] == 'indent_1':
        label = title[0]
        factor_list[label] = [[], []]
    else:
        factor_list[label][0].append(title[2])
        factor_list[label][1].append(title[3])
