import requests, re
text = requests.get('https://data.worldpop.org/GIS/AgeSex_structures/Global_2000_2020/2020/IND/').text
print(re.findall(r'href="([^"]+)"', text)[:30])
