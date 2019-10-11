import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import csv

headers = {'accept':'*/*',
           'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36'}
base_url = 'https://www.gipernn.ru/dom/gorod-nizhniY-novgorod?per-page=100&page=1'

def hh_parse(base_url, headers):
    real_estate = []
    urls = []
    district = None
    addres = None
    material = None
    flors_number = None
    year_buid = None
    appartment_number = None
    mean_price_sqr_m = None

    session = requests.session()
    requset = session.get(base_url, headers=headers, verify=False)
    if requset.status_code == 200:
        soup = bs(requset.content, 'lxml')

        pagination = soup.find('div', attrs={'class': 'count'}).text
        count_ad = int(pagination.split(" ")[1:][0])
        if count_ad % 100 == 0:
            count_page = count_ad // 100
        else:
            count_page = count_ad // 100 + 1

        for i in range(1, count_page + 1):
            url = f'https://www.gipernn.ru/dom/gorod-nizhniY-novgorod?per-page=100&page={i}'
            if url not in urls:
                urls.append(url)


    for url in urls:
        requset = session.get(url, headers=headers, verify=False)
        soup = bs(requset.content, 'lxml')

        divs = soup.find_all('tr')
        for div in divs:
            try:
    # district - район
                if div.span == None:
                    pass
                else:
                    district = div.span.text
                    #print(district)
    # addres - адрес
                if div.td == None:
                    pass
                else:
                    if div.td.next.next.next.text == "":
                        pass
                    else:
                        addres = div.td.next.next.next.a.text
                        #print(addres)
    # material - материал стен
                if div.td == None:
                    pass
                else:
                    if div.td.next.next.next.text == "":
                        pass
                    else:
                        material = div.td.next.next.next.next.next.next.next.next.next.text
                        #print(material)
    # flors_number - этажность
                if div.td == None:
                    pass
                else:
                    if div.td.next.next.next.text == "":
                        pass
                    else:
                        flors_number = div.td.next.next.next.next.next.next.next.next.next.next.next.text
                        #print(flors_number)
    # year_buid - год постройки
                if div.td == None:
                    pass
                else:
                    if div.td.next.next.next.text == "":
                        pass
                    else:
                        year_buid = div.span.next.next.next.next.next.next.text
                        #print(year_buid)
    # appartment_number - количество квартир
                if div.td == None:
                    pass
                else:
                    if div.td.next.next.next.text == "":
                        pass
                    else:
                        appartment_number = div.span.next.next.next.next.next.next.next.next.text
                        #print(appartment_number)
    # mean_price_sqr_m - средняя цена за кв. метр
                if div.td == None:
                    pass
                else:
                    if div.td.next.next.next.text == "":
                        pass
                    else:
                        mean_price_sqr_m = div.span.next.next.next.next.next.next.next.next.next.next.next.next.text
                        mean_price_sqr_m = "".join(mean_price_sqr_m.split())
                        #print(mean_price_sqr_m)
                real_estate.append({
                    'district': district,
                    'addres': addres,
                    'material': material,
                    'flors_number': flors_number,
                    'year_buid': year_buid,
                    'appartment_number': appartment_number,
                    'mean_price_sqr_m': mean_price_sqr_m
                })
            except:
                pass
        print(len(real_estate))

    else:
        print('ERRRRoR or Done' + str(requset.status_code))
    return real_estate


def files_writer(real_estate):
    with open('parsed_estate.csv', 'w') as file:
        a_pen = csv.writer(file)
        a_pen.writerow(('Район', 'Адрес', 'Материал стен', 'Этажность', 'Год постройки', 'Всего квартир', 'Средняя цена за кв. метр'))
        for estate in real_estate:
            a_pen.writerow((estate['district'], estate['addres'], estate['material'], estate['flors_number'], estate['year_buid'],
                            estate['appartment_number'], estate['mean_price_sqr_m']))

real_estate = hh_parse(base_url, headers)
files_writer(real_estate)

df = pd.DataFrame(list(real_estate))
df = df.drop_duplicates(subset=['addres'], keep='first')
print(df)
print(df.shape)
df.to_excel("output.xlsx")
