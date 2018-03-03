import requests
import os
import numpy as np
from urllib import quote_plus

base_url = 'http://poetrydb.org/'
save_dir_base = 'data/poetryDB/'
save_dir_txt = os.path.join(save_dir_base, 'txt')
save_dir_arr = os.path.join(save_dir_base, 'arr')

all_titles_url = os.path.join(base_url, 'title')
r = requests.get(all_titles_url)
titles = r.json()['titles']

num_titles = len(titles)
for i, title in enumerate(titles):
    print ('%d / %d titles downloaded - ' % (i + 1, num_titles)) + title
    try:
        url_title = quote_plus(title)
    except KeyError:
        url_title = title

    lines_url = os.path.join(base_url, 'title', url_title, 'lines')
    r = requests.get(lines_url)

    try:
        lines = r.json()[0]['lines']
        lines_np = np.array(lines)
        text = '\n'.join(lines).encode('utf8')

        np.save(os.path.join(save_dir_arr, title), lines_np)
        with open(os.path.join(save_dir_txt, title + '.txt'), 'w') as f:
            f.write(text)
    except KeyError as e:
        print 'KeyError: Could not download ' + title
        print e
    except ValueError as e:
        print 'ValueError: Could not download ' + title
        print e
