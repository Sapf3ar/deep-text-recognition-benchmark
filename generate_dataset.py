import glob
from sklearn.model_selection import train_test_split
from create_lmdb_dataset import createDataset
p = 'vin_crops/*jpg'
ims = glob.glob(p)
train_ims, test_ims = train_test_split(ims, test_size=0.1)

with open('train_ann.txt', 'x') as file:
    for im in train_ims:
      vin = im.split('.')[-2]
      string = '\t'.join([im, vin]) + '\n'
      file.writelines(string)


with open('test_ann.txt', 'x') as file:
   for im in test_ims:
      vin = im.split('.')[-2]

      string = '\t'.join([im, vin]) + '\n'
      file.writelines(string)
createDataset('.', 'train_ann.txt', 'lmdb_ocr_vin/train')

createDataset('.', 'test_ann.txt', 'lmdb_ocr_vin/test')
