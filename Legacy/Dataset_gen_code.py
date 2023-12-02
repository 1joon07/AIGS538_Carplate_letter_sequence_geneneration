import matplotlib.pyplot as plt
from PIL import Image
import os
import random
import pandas as pd

## Plate label length setting
## (x,y) -> length variation from x to y
Label_length = (4,11)

## Generated image resolution
generated_image_size=(256,64)

## Number of images to be generated & saved
Iteration = 400

## Location of source character images
data_directory = './CNN_letter_dataset'

## Location of the generated dataset
## a .csv file -> a list of labels
## .png files -> generated images
Generated_dataset_directory='./CNN_generated_dataset_val'

def plate_image_concatenation (label_sequence= 'ALIS'):

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

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    new_im = new_im.resize(generated_image_size)

    return new_im

def PLATE_dataset_generation (LABEL_Length):

    plate_letters = random_label(label_length=LABEL_Length)
    plate_image = plate_image_concatenation(plate_letters)

    print(plate_letters)

    return plate_letters, plate_image

def random_label (label_length=(4,10)):

    Length = random.randrange(label_length[0], label_length[1]+1)

    all_items = os.listdir(data_directory)
    characters = [item for item in all_items if os.path.isdir(os.path.join(data_directory, item))]

    letter_sequence = ''.join(random.choice(characters) for _ in range(Length))

    return letter_sequence

if __name__ == '__main__':
    Filename_list = []
    Label_list = []

    ## File deletion (generated dataset directory)
    for filename in os.listdir(Generated_dataset_directory):
        file_path = os.path.join(Generated_dataset_directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    ## Plate image generation
    for i in range(Iteration):
        PlateImage_Filename = f'Plate_{i}.png'
        tmp_label, tmp_plate_image = PLATE_dataset_generation(LABEL_Length=Label_length)
        tmp_plate_image.save(f'{Generated_dataset_directory}/{PlateImage_Filename}', 'png')

        Filename_list.append(PlateImage_Filename)
        Label_list.append(tmp_label)

    df = pd.DataFrame(Filename_list, columns=['Filename'])
    df['Label'] = Label_list
    df.to_csv(f'{Generated_dataset_directory}/Labels.csv', index=False)
