import requests
from bs4 import BeautifulSoup as bs
from decimal import Decimal
from gipernn_parser import full_links
import pandas as pd

headers = {'accept':'*/*',
           'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36'}
base_url = 'https://www.gipernn.ru/prodazha-kvartir/2-komnatnaya-ul-1-y-mikrorayon-shherbinki-d-6-id2571864'
adv_list = []
def per_adv_parser(base_url, headers):
    session = requests.session()
    for full_link in full_links:
        request = session.get(full_link, headers=headers, verify=False)
        if request.status_code == 200:
            soup = bs(request.content, 'lxml')
            try:

                price = soup.find('div', attrs={'class': 'price'}).text
                price = price.replace(' ', '')
                price = price.split("руб.")[:1][0]
                price = int("".join(price.split()))
                structure = soup.find_all('td')
                district = structure[1].text
                address = structure[2].text
                number_of_rooms = int(structure[3].text.split(" ")[:1][0])
                square = structure[4].text
                total_square = Decimal(square.split("/")[:1][0].replace(',', '.'))
                living_square = Decimal(square.split("/")[:2][1].replace(',', '.'))
                kitchen_square = Decimal(square.split("/")[:3][2].replace(',', '.'))
                current_floor = structure[5].text.split("/")[:1][0]
                number_of_floors = int(structure[5].text.split("/")[:2][1])
                build_year = structure[6].text
                material = structure[7].text
                ceiling_height = structure[8].text.split(" ")[:1][0].replace(',', '.')
                adv_list.append({
                    'price': price,
                    'district': district,
                    'address': address,
                    'number_of_rooms': number_of_rooms,
                    'total_square': total_square,
                    'living_square': living_square,
                    'kitchen_square': kitchen_square,
                    'current_floor': current_floor,
                    'number_of_floors': number_of_floors,
                    'build_year': build_year,
                    'material': material,
                    'ceiling_height': ceiling_height,
                })
                print(len(adv_list))
            except:
                pass


        else:
            print('ERRRrrorrrrr')
    return adv_list

per_adv_parser(base_url, headers)


df = pd.DataFrame(list(adv_list))
print(df)
print(df.shape)
df.to_excel("all_adv.xlsx")