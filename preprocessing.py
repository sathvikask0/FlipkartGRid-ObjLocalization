import os
import csv
from PIL import Image

with open('/home/karna/Desktop/fg/test.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    i = 0 
    for row in csv_reader:
        for filename in os.listdir('/home/karna/Desktop/fg/images'):
            if( row[0] == filename ):
                im = Image.open(os.path.join('/home/karna/Desktop/fg/images',filename))
                img = im.resize((224, 224))
                img.save("{}.png".format(i))
                i = i + 1
                print(i)


       

