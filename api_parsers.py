import json
import requests
import time
import re

from bs4 import BeautifulSoup


def parse_single_anime(node):
    res = {**node['node'], **node['list_status']}
    try:
        res.pop('main_picture')
    except Exception:
        print(res)

    return res


def _get_anime_list(url, token):
    response = requests.get(
        url,
        headers={'Authorization': F'Bearer {token}'}
    )

    return json.loads(response.text)


def _get_user_anime_list_url(username, token):
    url = F'https://api.myanimelist.net/v2/users/{username}/animelist?fields=list_status&limit=1000'

    return url


# @lru_cache(maxsize=None)
def get_user_anime_list(user_name, token):
    url = _get_user_anime_list_url(user_name, token)
    res = _get_anime_list(url, token)

    if 'error' in res:
        raise Exception(F"[{res['error']}] encountered for {user_name}")

    parsed = []
    for anime in res['data']:
        parsed.append(parse_single_anime(anime))

    while 'next' in res['paging']:
        time.sleep(.5)
        res = _get_anime_list(res['paging']['next'], token)

        for anime in res['data']:
            parsed.append(parse_single_anime(anime))

    return parsed


def get_random_list_of_users():
    response = requests.get('https://myanimelist.net/users.php')
    soup = BeautifulSoup(response.text, 'html.parser')

    # soup.findAll("div", attrs={"style": "margin-bottom"}, recursive=True)  # This doesn't work for some reason
    filtered_soup = soup.findAll("a", attrs={"href": re.compile(".*profile.*")})

    names = []
    for spoon in filtered_soup:
        content = str(spoon.contents[0])
        if content[0] == '<':
            continue
        names.append(spoon.contents[0])

    return names
