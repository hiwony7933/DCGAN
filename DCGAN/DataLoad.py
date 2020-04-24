from PIL import Image
import os, glob, sys, numpy as np
from sklearn.model_selection import train_test_split

class DataLoad() : 
    def __init__(self) : 
        path = 'cat'
        image_w = 64
        image_h = 64
        image_path_names = []
        person_names = set()

        X = []
        y = []

        for file_name in glob.glob(path + '/[a-zA-Z]*[1-9]*.jpg') : 
            image_path_names.append(file_name)
            person_names.add(image_path_names[-1].split('/')[-2].split('_')[0])

        for i, f in enumerate(image_path_names) :
            try :
                img = Image.open(f)
                img = img.convert('RGB')
                img = img.resize((image_w, image_h))
                data = np.asarray(img)
                X.append(data)
                y.append(person_names) 
                if i % 100 == 0 :
                    print(f)
            except : 
                print(file_name + str(i) + '에러')

        X = np.array(X)
        Y = np.array(y)
        print(X.shape)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
        xy = (X_train, X_test, Y_train, Y_test)
        np.save("binary_image_data.npy", xy)

