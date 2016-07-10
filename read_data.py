import pandas as pd
from skimage import io
from PIL import Image
import numpy as np
import os
import re
import matplotlib.pyplot as plt

rootdir0="/home/mina/data_science_game/roof_images/"

count=0
dic={}
STANDARD_SIZE = (200, 200)
#print(os.path.dirname(os.path.abspath(__file__)))
#df_shape = pd.DataFrame(columns=('x', 'y', 'rgb'))

for root, dirnames, filenames in os.walk( rootdir0 ):
    for filename in filenames:

	# By using skimage.io.imread, we load the image into numpy.ndarray format.
	img = Image.open(os.path.join(root,filename))
	img = img.resize(STANDARD_SIZE)
	img = np.array(img)
#	myfile=open(os.path.join(root,filename))
#	mg=io.imread(myfile)
	#print(img.shape) # From the shape we can see that each image is a RGB color image, however the size is different from one another.	
	#df_shape.loc[count] = img.size

	count=count+1
	# We keep only the id for each picture, so that we remove '.jpg' in the filename.
	filename=re.sub(r'.jpg','',filename)
	
	dic[filename]=img
#	data.append(dic)
	if count%5000==0:
		print(len(dic))
print(count)
#del df_shape['rgb']
#plt.figure()
#df_shape.plot(kind='scatter', x='x', y='y')
#plt.savefig('myfig.png')
#plt.figure()
#df_shape.boxplot()
#plt.savefig('box.png')
#df_shape.boxplot(by='y')
#plt.savefig('ybox.png')

ordered=sorted(dic.items())
img_data = pd.DataFrame(ordered)
img_data.columns = ['Id', 'data']
# Convert the Id type into string to ensure that it will match the Id of label
img_data['Id']=img_data['Id'].astype(str)
#print(img_data)

label=pd.read_csv("/home/mina/data_science_game/id_train.csv")
# Convert the Id type into string to ensure that it will match the Id of img_data
label['Id']=label['Id'].astype(str)
#print(label)

# By merging the two datasets, we obtain the train set. There are 8000 observations with the label of roof type. Since we have 42759 images, but only 8000 labels. It's possible to try semi-supervised learning with the unlabeled images.

train=pd.merge(img_data, label, on='Id', how='inner')
#print(train)

train.to_csv("train.csv", sep=';', index=False)
