import requests

url = 'https://www.appeditmobile.com/2020/01/06/android-internet-usage-monitoring-who-consume-all-the-data-from-your-network/'

res = requests.get(url)

print(res.status_code)

html_page = res.content

from bs4 import BeautifulSoup

soup = BeautifulSoup(html_page, 'html.parser')

text = soup.find_all(text=True)

output = ''
blacklist = [
    '[document]',
    'noscript',
    'header',
    'html',
    'meta',
    'head', 
    'input',
    'script',
    'style',
    # there may be more elements you don't want, such as "style", etc.
]

# print(soup.prettify())

# paragraphs = soup.find_all('p')
# f = open('text_corpu','a+')

# for p in paragraphs:
#     print(p.get_text())
#     f.write(p.get_text())
#     f.write('\n')

# f.close()


for t in text:
    if t.parent.name not in blacklist:
        output += '{} '.format(t)

f = open('text_corpu','a+')
f.write(output)
f.close()


print(output)