import requests
from bs4 import BeautifulSoup as bs

headers = {'accept':'*/*',
           'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36'}
base_url = 'https://www.gipernn.ru/prodazha-kvartir?sort=address&per-page=100&page=1'
full_links = []
def adv_parser(base_url, headers):

    urls = []
    session = requests.session()
    request = session.get(base_url, headers=headers, verify=False)
    if request.status_code == 200:
        soup = bs(request.content, 'lxml')

        pagination = soup.find('div', attrs={'class': 'count'}).text
        count_adv = int(pagination.split(" ")[1:][0])
        if count_adv % 100 == 0:
            count_page = count_adv // 100
        else:
            count_page = count_adv // 100 +1

        for i in range(1, count_page + 1):
            url = f'https://www.gipernn.ru/prodazha-kvartir?sort=address&per-page=100&page={i}'
            if url not in urls:
                urls.append(url)

        for url in urls:
            request = session.get(url, headers=headers, verify=False)
            soup = bs(request.content, 'lxml')

            blocks = soup.find_all('tr')
            for block in blocks:
                try:
                    if block.a == None:
                        pass
                    else:
                        link = block.find('a', attrs={'class': 'photo'})['href']
                        link = 'https://www.gipernn.ru' + link
                        #if link not in full_links:
                        #print(link)
                        full_links.append(link)
                except:
                    pass
        print(len(full_links))

    else:
        print("ERRRORRRRrrrrr")

    return full_links

adv_parser(base_url, headers)


