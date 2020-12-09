import urllib.request
import os
import os.path
import zipfile

if not os.path.exists(os.path.join('data')):
    os.makedirs('data')

if os.path.exists(os.path.join('data', 'ut-zap50k-images')):
    pass
else:
    urllib.request.urlretrieve("http://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-images.zip", filename="data/ut-zap50k-imgs.zip")
    zip_ref = zipfile.ZipFile("data/ut-zap50k-imgs.zip", 'r')
    zip_ref.extractall("data")
    zip_ref.close()
    os.remove("data/ut-zap50k-imgs.zip")

if os.path.exists(os.path.join('data', 'tripletlists')):
    pass
else:    
    urllib.request.urlretrieve("https://s3.amazonaws.com/andreas-datasets/csn_zappos_triplets.zip", filename="data/triplets.zip")
    zip_ref = zipfile.ZipFile("data/triplets.zip", 'r')
    zip_ref.extractall("data")
    zip_ref.close()
    os.remove("data/triplets.zip")