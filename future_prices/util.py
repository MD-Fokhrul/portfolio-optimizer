import os
import re


filename_pattern = re.compile('model_\d+.pth')
digit_pattern = re.compile('\d+')


def find_latest_model_name(dir_path):
    global filename_pattern, digit_pattern

    filenames = os.listdir(dir_path)
    found = []
    for filename in filenames:
        is_match = filename_pattern.search(filename)
        if is_match:
            found.append(int(digit_pattern.search(filename).group(0)))

    return '{}/model_{}.pth'.format(dir_path, max(found))







