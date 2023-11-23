import matplotlib.pyplot as plt
from PIL import Image
import os
import random

data_directory = 'C:/Users/Wonjoon_LAB/PycharmProjects/AIGS538_Carplate_letter_sequence_geneneration/CNN letter Dataset'
Generated_dataset_directory='./CNN_generated_dataset'

def plate_image_concatenation (label_sequence= 'ALIS'):
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
def PLATE_dataset_generation ():

    plate_letters = random_label()
    plate_image = plate_image_concatenation(plate_letters)

    print(plate_letters)
    plate_image.show()
    return 0
def random_label (label_length=(4,10)):

    Length = random.randrange(label_length[0], label_length[1]+1)

    all_items = os.listdir(data_directory)
    characters = [item for item in all_items if os.path.isdir(os.path.join(data_directory, item))]

    letter_sequence = ''.join(random.choice(characters) for _ in range(Length))

    return letter_sequence
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    PLATE_dataset_generation()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
