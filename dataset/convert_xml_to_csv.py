# based on https://github.com/datitran/raccoon_dataset/blob/master/xml_to_csv.py

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        print(xml_file)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            bdnBox = member.find('bndbox')

            value = (root.find('filename').text,
                     int(root.find('size')[0].text)//10,
                     int(root.find('size')[1].text)//10,
                     member.find('name').text,
                     int(bdnBox.find('xmin').text)//10,
                     int(bdnBox.find('ymin').text)//10,
                     int(bdnBox.find('xmax').text)//10,
                     int(bdnBox.find('ymax').text)//10
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    for folder in ['train', 'test']:
        image_path = os.path.join(os.getcwd(), ('annotations/' + folder))
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv(('images/'+folder+'_labels.csv'), index=None)
    print('Successfully converted xml to csv.')


main()
