import matplotlib.pyplot as plt
from PIL import Image
import os
import random


def plate_image_concatenation (label_sequence= 'BEVIL'):
    data_directory = 'C:/Users/Wonjoon_LAB/PycharmProjects/AIGS538_Carplate_letter_sequence_geneneration/CNN letter Dataset'
    generated_image_size=(256,64)

    image_path_list = []
    for character in label_sequence:
        image_directory = f'{data_directory}/{character}'
        files = os.listdir(image_directory)
        image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

        selected_image_file = random.choice(image_files)
        selected_image_path = os.path.join(image_directory, selected_image_file)
        image_path_list.append(selected_image_path)

        images = [Image.open(image) for image in image_path_list]

    widths, heights = zip(*(i.size for i in images))

    # 총 너비와 가장 높은 높이를 계산
    total_width = sum(widths)
    max_height = max(heights)

    # 새로운 이미지 생성 (전체 너비와 가장 높은 높이 사용)
    new_im = Image.new('RGB', (total_width, max_height))

    # 각 이미지를 새 이미지에 붙여넣기
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    new_im = new_im.resize(generated_image_size)

    return new_im



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    plate_image_concatenation()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
