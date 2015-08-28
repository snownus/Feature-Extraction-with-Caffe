import numpy as np
import os

directory = 'The Oxford-IIIT Pet Dataset'
data = os.listdir(directory)
classes = [' '.join(data[i].rsplit('_')[0:-1]) for i in range(len(data))]

dic= {} # count number of images per class
for d in classes:
    if d in dic:
        dic[d]= dic[d]+1
    else:
        dic[d]= 1

name_label = {} # define class number
i = 0
for d in classes:
    if d not in name_label:
        name_label[d]= i
        i = i+1

for d in dic: # print n of images, class number, class name
    print '%d %2d %s' % (dic[d], name_label[d], d)

N1 = 150 # n of images for training per class
#N2 = 50 # n of images for training validation per class
train = []
test = []
for label in range(i):
    filenames = [[name,label] for i,name in enumerate(data) \
                            if name_label[classes[i]]==label]
    np.random.shuffle(filenames) # shuffle list filenames
    train.extend(filenames[0:N1])
    test.extend(filenames[N1:])

# write filename and class number to text
np.save('train_filenames.npy', np.array(train).transpose()[0])
np.save('train_labels.npy', np.array(train).transpose()[1].astype(np.int32))
np.save('test_filenames.npy', np.array(test).transpose()[0])
np.save('test_labels.npy', np.array(test).transpose()[1].astype(np.int32))
#np.savetxt('train.txt', np.array(train), fmt="%s")
#np.savetxt('test.txt', np.array(test), fmt="%s")

# write class names in ascending order
categories = [[name_label[d], d] for d in dic]
categories.sort()
words = np.array(categories)
words = np.transpose(words)
np.save('class_names.npy', np.array(words[1]))
#np.savetxt('words.txt', words[1], fmt="%s")
