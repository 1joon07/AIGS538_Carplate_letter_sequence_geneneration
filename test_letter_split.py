import os
import Letter_split as splitter
import pandas as pd
from termcolor import colored
dataset_test = './CNN_generated_dataset2'


def obtain_csv_and_image_path(directory):
    csv_files = []
    image_files = []

    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            csv_files.append(os.path.join(directory,filename))
        elif filename.endswith(('.png', '.jpg')):
            image_files.append(os.path.join(directory,filename))

    return csv_files[0], image_files

def find_label_for_image(csv_path, image_path):
    image_filename = os.path.basename(image_path)
    df = pd.read_csv(csv_path)
    label_row = df[df['Filename'] == image_filename]
    if not label_row.empty:
        return label_row['Label'].values[0]
    else:
        return None

def check_letter_split(image_files, csv_path):
    total_plates = 0
    total_characters = 0
    correctly_segmented_plates = 0
    correctly_detected_characters = 0
    for image_path in image_files:
        label = find_label_for_image(csv_path, image_path)
        if label is None:
            continue
        num_characters_in_label = len(label)
        segmented_letters = splitter.pipeline_split_letter_wpath(image_path)
        num_segmented_letters = len(segmented_letters)
        total_plates += 1
        total_characters += num_characters_in_label
        if num_segmented_letters == num_characters_in_label:
            correctly_segmented_plates += 1
            correctly_detected_characters += num_segmented_letters
            print(colored(f'Plate correctly segmented {image_path}' ,'green'))
        else:
            min_detected = min(num_segmented_letters, num_characters_in_label)
            correctly_detected_characters += min_detected
            print(colored(f'Plate badly segmented {image_path}' ,'red'))
    plate_accuracy = correctly_segmented_plates/total_plates
    character_segmented = correctly_detected_characters/total_characters
    print('Test Character segmentation')
    print(f'Plate Accuracy {plate_accuracy}\n Character segmented {character_segmented}\n details :\n Total Plate :{total_plates}\n Total Characters {total_characters}\n Correctly Segmented Plates :{correctly_segmented_plates}\n Number of Characters Segmented : {correctly_detected_characters}\n')
    return {
        "total_plates": total_plates,
        "total_characters": total_characters,
        "correctly_segmented_plates": correctly_segmented_plates,
        "correctly_detected_characters": correctly_detected_characters
    }


if __name__ == '__main__':
    csv, images_files = obtain_csv_and_image_path(dataset_test)
    check_letter_split(images_files, csv)