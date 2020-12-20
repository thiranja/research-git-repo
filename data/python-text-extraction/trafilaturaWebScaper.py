import trafilatura
import time

# links = open('linksAppeditmobile.txt','r')
linkFiles = ['linksAppeditMobile.txt','linksXDAforum.txt','linksXDAWebsite.txt','linksMobileInternist.txt','linksBestForAndroid.txt','linksAndriodCentral.txt','linksDroidLife.txt','linksAndroidAdvices.txt','linksAndroidGuys.txt','linksAndroidHeadlines.txt']
corpusFiles = []

with open('linksAndriodCentral.txt', 'rb') as txtfile:
	links = txtfile.readlines()

print(links[0])

f = open('corpusAndriodCentral.txt','w+')

totalLinks = len(links)
extracted = 0
error = 0

for link in links:
    link = link.rstrip()
    downloaded = trafilatura.fetch_url(link)
    text = trafilatura.extract(downloaded)
    if (text == None):
        error = error +1
    else:
        extracted = extracted + 1
        f.write(text)
    print('Extracted ' + str(extracted) + ' Error ' + str(error) + ' Total ' + str(totalLinks))
    time.sleep(15)
    


f.close()
# downloaded = trafilatura.fetch_url('https://www.appeditmobile.com/2020/01/06/android-internet-usage-monitoring-who-consume-all-the-data-from-your-network/')
# text = trafilatura.extract(downloaded)

# print(text)