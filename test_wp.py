import requests, re
text = requests.get('https://data.worldpop.org/GIS/AgeSex_structures/Global_2015_2030/R2025A/2025/BGD/v1/100m/constrained/').text
print([m for m in re.findall(r'href="([^"]+)"', text) if '_t_' in m])
