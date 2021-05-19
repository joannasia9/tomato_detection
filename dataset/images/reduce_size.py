from PIL import Image
import PIL
import os
import glob

def reduce(path):
    for file in glob.glob(path + '/*.png'):
        image = Image.open(file)
        hsize = int(float(image.size[1]) * 0.1)
        wsize = int(float(image.size[0]) * 0.1)
        image = image.resize((wsize, hsize), Image.ANTIALIAS)
        image.save(file)

def main():
    for folder in ['train', 'test']:
        image_path = os.path.join(os.getcwd(), (folder))
        reduced_file = reduce(image_path)
        
main()
