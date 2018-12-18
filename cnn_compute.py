

import argparse
parser = argparse.ArgumentParser(description="Train CNN model")
parser.add_argument("-c","--classes",choices=['airplane','bus', 'car', 'dog', 'person', 'train', 'all'],required=True, help="File with URL for download")
parser.add_argument("-m", "--model", choices=['lenet5','densenet', 'squeezenet', 'lenet5_64'], required=True, help="Select the Models from the list")
parser.add_argument("-d","--dir", default="data", help="Path of training images or image")
parser.add_argument("-p","--pridict",help="Pridict the given image", action="store_true")
parser.add_argument("-e","--evaluate",help="evaluate the object image", action="store_true")
args = parser.parse_args()

import numpy as np
import os
from models import lenet5,lenet5_64 ,squeezenet, densenet
from glob import glob 
from random import shuffle
from jfile import jfile
from cv2 import imread, resize 
print(args.evaluate)
no_c = 7 if args.classes == 'all' else 2
type_mod = eval(args.model)(no_c)
height, width = type_mod.height, type_mod.width
_class = ['not', args.classes] if args.classes != 'all' else ['not','airplane','bus','car','dog','person','train']

if args.pridict != True and args.evaluate != True:
    model = type_mod.build()
    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])
    dic = {'train':[], 'val':[]}
    arr = {'train':{'x':[], 'y':[]}, 'val':{'x':[],'y':[]}}
    for f in arr:
        p = args.dir + '/' + f + '/'
        for c in _class:
            dic[f].extend(list( w for w in glob(p+c+'*'))) 
        shuffle(dic[f])
        print(len(dic[f]))
        for e in dic[f]:
            tmp = imread(e)
            arr[f]['x'].append(resize(tmp,(type_mod.height, type_mod.width)))
            arr[f]['y'].append(_class.index(e.split('/')[-1].split('_')[0]))
            print("Total number of images imported -> ", len(arr[f]['y']), ' of ', len(dic[f]),end='\r')
    print(len(arr['train']['x']))


    from keras.preprocessing.image import ImageDataGenerator
    from keras.utils import to_categorical
    #ims(arr['train']['x'][1], _class[arr['train']['y'][1]])
    train_gen = ImageDataGenerator(rescale = 1/255,shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
    val_gen = ImageDataGenerator(rescale=1/255)
    train = train_gen.flow(np.asarray(arr['train']['x']), y = to_categorical(arr['train']['y'], num_classes = len(_class)), batch_size=32) 
    val = val_gen.flow(np.asarray(arr['val']['x']), y = to_categorical(arr['val']['y'], num_classes = len(_class)), batch_size=32)
    name = args.model+'_'+args.classes 
    from keras.callbacks import CSVLogger
    csvlogger = CSVLogger('log/'+ name +'.log')
    history = model.fit_generator(train , epochs = 25, steps_per_epoch=len(train),validation_data = val,validation_steps = len(val),  use_multiprocessing=True, callbacks=[csvlogger])
    model.save('models/'+ name + '.h5')
    output = jfile('output')
    output.data.update({name:{'class':_class, 'result':history.history}})
    output.save()
    import matplotlib.pyplot as plt
    x = range(1,26)
    plt.plot(x, history.history['acc'], '-g', label='Train Acc')
    plt.plot(x, history.history['loss'], ':g', label='Train loss')
    plt.plot(x, history.history['val_acc'], '-r', label='Val Acc')
    plt.plot(x, history.history['val_loss'], ':r', label='Val loss')
    plt.title("Training data")
    plt.xlabel("no of epochs")
    plt.ylabel("values")
    plt.savefig('fig/'+ name +'.jpg')
elif args.evaluate == True:
    from keras.preprocessing.image import ImageDataGenerator
    from keras.utils import to_categorical
    from keras.models import load_model 
    name = args.model+'_'+args.classes 
    model = load_model("models/" + name + ".h5")
    ev_class = _class[1:]
    val_gen = ImageDataGenerator(rescale=1/255)
    p = args.dir +"/val/"
    dic_not = list( w for w in glob(p+"not*"))
    for a in ev_class:
        arr = {"x":[], "y":[]}
        dic = list( w for w in glob(p+ a+"*")) + dic_not
        for e in dic:
            tmp = imread(e)
            arr['x'].append(resize(tmp,(type_mod.height, type_mod.width)))
            arr['y'].append(_class.index(e.split('/')[-1].split('_')[0]))
            print("Total number of images imported -> ", len(arr['y']), ' of ', len(dic),end='\r')
        val = val_gen.flow(np.asarray(arr['x']), y = to_categorical(arr['y'], num_classes = len(_class)), batch_size=32)
        res = model.evaluate_generator(val , steps=len(val), use_multiprocessing=True, verbose=0)
        print("Results for class ", a ," test is Loss : " ,res[0], " and Accuracy : " , res[1])

elif args.pridict == True:
    from keras.models import load_model 
    img = imread(args.dir)
    im = resize(img, (height, width))
    i = np.expand_dims(im, axis=0)
    i = i/ 255
    name = args.model+'_'+args.classes 
    model = load_model("models/" + name + ".h5")
    res = model.predict(i)
    for i in range(len(_class)):
        print(_class[i], " : ", round(res[0][i] * 100, 2), '%' )

        


