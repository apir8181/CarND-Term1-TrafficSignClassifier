
# Self-Driving Car Engineer Nanodegree

## Deep Learning

## Project: Build a Traffic Sign Recognition Classifier

In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 

> **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 

In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.

The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.


>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

---
## Step 0: Load The Data


```python
# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'traffic-signs-data/train.p'
validation_file= 'traffic-signs-data/valid.p'
testing_file = 'traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
```

---

## Step 1: Dataset Summary & Exploration

The pickled data is a dictionary with 4 key/value pairs:

- `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
- `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
- `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
- `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**

Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 

### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas


```python
### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
import numpy as np

n_train = X_train.shape[0]

# TODO: Number of validation examples
n_validation = X_valid.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_train.shape[1:]

# TODO: How many unique classes/labels there are in the dataset.
n_classes = np.unique(np.concatenate([y_train, y_valid, y_test], axis=0)).shape[0]

with open('./signnames.csv', 'r') as in_file:
    lines = in_file.readlines()[1:]
    classes_name = list( map(lambda x: x.strip().split(',')[1][:20], lines) )

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
```

    Number of training examples = 34799
    Number of testing examples = 12630
    Image data shape = (32, 32, 3)
    Number of classes = 43
    

### Include an exploratory visualization of the dataset

Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc. 

The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.

**NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections. It can be interesting to look at the distribution of classes in the training, validation and test set. Is the distribution the same? Are there more examples of some classes than others?


```python
### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
%matplotlib inline
```

For each class, visualize a image from the dataset. It seems those images are taken under differnet light condition.


```python
import math

def draw_images(X, y, draw_per_class=1):
    draw_remain_class = [draw_per_class] * n_classes
    draw_remain_total = draw_per_class * n_classes
    iter_idx = 0
    plt.figure(figsize=(12, 8), dpi=100)
    while iter_idx < y.shape[0] and draw_remain_total > 0:
        label = y[iter_idx]
        if draw_remain_class[label] > 0:
            plt.subplot(math.ceil(draw_per_class * n_classes / 10), 10, label * draw_per_class + draw_remain_class[label])
            plt.title(classes_name[label], fontsize=6)
            plt.imshow(X[iter_idx])
            plt.axis('off')
            draw_remain_class[label] -= 1
            draw_remain_total -= 1
        iter_idx += 1
    plt.show()
    
draw_images(X_train, y_train)
```


![png](output_10_0.png)


In each dataset count the number of appearence of different classes. Their distributions are very close to each other.


```python
import pandas as pd

histogram = lambda y: pd.DataFrame({'y': y}).groupby('y').size()
train_hist, valid_hist, test_hist = histogram(y_train), histogram(y_valid), histogram(y_test)

ind = np.arange(train_hist.shape[0])
rect_1 = train_hist / y_train.shape[0]
rect_2 = valid_hist / y_valid.shape[0]
rect_3 = test_hist / y_test.shape[0]

fig, ax = plt.subplots(figsize=(10, 6))
width = 0.2
rect_1 = ax.bar(ind - width, train_hist / y_train.shape[0], width, color='r', label='y_train')
rect_2 = ax.bar(ind, valid_hist / y_valid.shape[0], width, color='g', label='y_valid')
rect_3 = ax.bar(ind + width, test_hist / y_test.shape[0], width, color='b', label='y_test')

ax.set_ylabel('Proption')
ax.set_title('Proption of classes')
ax.set_xticks(ind)
ax.legend()

plt.show()
```


![png](output_12_0.png)


Visualize training images in each classes.

----

## Step 2: Design and Test a Model Architecture

Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play! 

With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission. 

There are various aspects to consider when thinking about this problem:

- Neural network architecture (is the network over or underfitting?)
- Play around preprocessing techniques (normalization, rgb to grayscale, etc)
- Number of examples per label (some have more than others).
- Generate fake data.

Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

### Pre-process the Data Set (normalization, grayscale, etc.)

Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data and can be used in this project. 

Other pre-processing steps are optional. You can try different techniques to see if it improves performance. 

Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.


```python
### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.

# normalize images
X_mean = np.mean(X_train, axis=(0, 1, 2))
print(X_mean)
```

    [86.69812205 79.49594061 81.83870445]
    

### Model Architecture


```python
### Define your architecture here.
### Feel free to use as many code cells as needed.

### Shallow resnet like model:
### - 3 convolution blocks. Each block contains 2 convolution layers and 1 max pooling layer.
### - 2 fully connected layers with dropout enabled.
from keras.backend import zeros_like
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, BatchNormalization, MaxPooling2D, Add, Flatten, Dense, Dropout, Concatenate, Lambda
from keras.optimizers import SGD, Adam
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy

def pad_like(A, B):
    n_channel = B.shape.as_list()[-1]
    pad_channel = n_channel - A.shape.as_list()[-1]
    C = zeros_like(B)
    C = C[:, :, :, :pad_channel]
    return Concatenate()([A, C])

def resnet_block(Z, n_channel, name):
    conv_1 = Conv2D(filters=n_channel, kernel_size=3, padding='same', activation='relu', name=name + '_conv1')(Z)
    conv_2 = Conv2D(filters=n_channel, kernel_size=3, padding='same', activation='relu', name=name + '_conv2')(conv_1)
    
    pad_Z = Lambda(pad_like, arguments={'B': conv_2}, name=name + '_in_pad')(Z)
    combine = Add(name=name + '_add')([conv_2, pad_Z])
    combine = BatchNormalization(name=name + '_bn')(combine)
    pool = MaxPooling2D(name=name + '_maxpool')(combine)
    return pool


X = Input(shape=(32, 32, 3))
X_bn = BatchNormalization(name='X_bn')(X)
block_1 = resnet_block(X_bn, 16, 'block1')
block_2 = resnet_block(block_1, 24, 'block2')
block_3 = resnet_block(block_2, 32, 'block3')
flatten = Flatten(name='Z')(block_3)
fc1 = Dense(128, activation='relu', name='fc1_dense')(flatten)
fc1 = BatchNormalization(name='fc1_bn')(fc1)
fc1 = Dropout(0.5, name='fc1_dropoput')(fc1)
fc2 = Dense(128, activation='relu', name='fc2_dense')(fc1)
fc2 = BatchNormalization(name='fc2_bn')(fc2)
fc2 = Dropout(0.5, name='fc2_dropout')(fc2)
softmax = Dense(n_classes, activation='softmax', name='softmax')(fc2)
model = Model(inputs=X, outputs=softmax)
model.compile(Adam(), loss='categorical_crossentropy',
              metrics=['categorical_accuracy', 'top_k_categorical_accuracy'])
model.summary()
```

    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_3 (InputLayer)            (None, 32, 32, 3)    0                                            
    __________________________________________________________________________________________________
    X_bn (BatchNormalization)       (None, 32, 32, 3)    12          input_3[0][0]                    
    __________________________________________________________________________________________________
    block1_conv1 (Conv2D)           (None, 32, 32, 16)   448         X_bn[0][0]                       
    __________________________________________________________________________________________________
    block1_conv2 (Conv2D)           (None, 32, 32, 16)   2320        block1_conv1[0][0]               
    __________________________________________________________________________________________________
    block1_in_pad (Lambda)          (None, 32, 32, 16)   0           X_bn[0][0]                       
    __________________________________________________________________________________________________
    block1_add (Add)                (None, 32, 32, 16)   0           block1_conv2[0][0]               
                                                                     block1_in_pad[0][0]              
    __________________________________________________________________________________________________
    block1_bn (BatchNormalization)  (None, 32, 32, 16)   64          block1_add[0][0]                 
    __________________________________________________________________________________________________
    block1_maxpool (MaxPooling2D)   (None, 16, 16, 16)   0           block1_bn[0][0]                  
    __________________________________________________________________________________________________
    block2_conv1 (Conv2D)           (None, 16, 16, 24)   3480        block1_maxpool[0][0]             
    __________________________________________________________________________________________________
    block2_conv2 (Conv2D)           (None, 16, 16, 24)   5208        block2_conv1[0][0]               
    __________________________________________________________________________________________________
    block2_in_pad (Lambda)          (None, 16, 16, 24)   0           block1_maxpool[0][0]             
    __________________________________________________________________________________________________
    block2_add (Add)                (None, 16, 16, 24)   0           block2_conv2[0][0]               
                                                                     block2_in_pad[0][0]              
    __________________________________________________________________________________________________
    block2_bn (BatchNormalization)  (None, 16, 16, 24)   96          block2_add[0][0]                 
    __________________________________________________________________________________________________
    block2_maxpool (MaxPooling2D)   (None, 8, 8, 24)     0           block2_bn[0][0]                  
    __________________________________________________________________________________________________
    block3_conv1 (Conv2D)           (None, 8, 8, 32)     6944        block2_maxpool[0][0]             
    __________________________________________________________________________________________________
    block3_conv2 (Conv2D)           (None, 8, 8, 32)     9248        block3_conv1[0][0]               
    __________________________________________________________________________________________________
    block3_in_pad (Lambda)          (None, 8, 8, 32)     0           block2_maxpool[0][0]             
    __________________________________________________________________________________________________
    block3_add (Add)                (None, 8, 8, 32)     0           block3_conv2[0][0]               
                                                                     block3_in_pad[0][0]              
    __________________________________________________________________________________________________
    block3_bn (BatchNormalization)  (None, 8, 8, 32)     128         block3_add[0][0]                 
    __________________________________________________________________________________________________
    block3_maxpool (MaxPooling2D)   (None, 4, 4, 32)     0           block3_bn[0][0]                  
    __________________________________________________________________________________________________
    Z (Flatten)                     (None, 512)          0           block3_maxpool[0][0]             
    __________________________________________________________________________________________________
    fc1_dense (Dense)               (None, 128)          65664       Z[0][0]                          
    __________________________________________________________________________________________________
    fc1_bn (BatchNormalization)     (None, 128)          512         fc1_dense[0][0]                  
    __________________________________________________________________________________________________
    fc1_dropoput (Dropout)          (None, 128)          0           fc1_bn[0][0]                     
    __________________________________________________________________________________________________
    fc2_dense (Dense)               (None, 128)          16512       fc1_dropoput[0][0]               
    __________________________________________________________________________________________________
    fc2_bn (BatchNormalization)     (None, 128)          512         fc2_dense[0][0]                  
    __________________________________________________________________________________________________
    fc2_dropout (Dropout)           (None, 128)          0           fc2_bn[0][0]                     
    __________________________________________________________________________________________________
    softmax (Dense)                 (None, 43)           5547        fc2_dropout[0][0]                
    ==================================================================================================
    Total params: 116,695
    Trainable params: 116,033
    Non-trainable params: 662
    __________________________________________________________________________________________________
    

### Train, Validate and Test the Model

A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.


```python
### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
from keras.utils.np_utils import to_categorical
y_train_oh = to_categorical(y_train)
y_valid_oh = to_categorical(y_valid)
y_test_oh = to_categorical(y_test)
```


```python
# train model
model.fit(x=X_train, y=y_train_oh, epochs=10, validation_data=(X_valid, y_valid_oh))
```

    Train on 34799 samples, validate on 4410 samples
    Epoch 1/10
    34799/34799 [==============================] - 178s 5ms/step - loss: 2.2501 - categorical_accuracy: 0.3891 - top_k_categorical_accuracy: 0.7006 - val_loss: 1.0716 - val_categorical_accuracy: 0.6619 - val_top_k_categorical_accuracy: 0.9259
    Epoch 2/10
    34799/34799 [==============================] - 176s 5ms/step - loss: 0.5627 - categorical_accuracy: 0.8305 - top_k_categorical_accuracy: 0.9787 - val_loss: 0.2745 - val_categorical_accuracy: 0.9193 - val_top_k_categorical_accuracy: 0.9896
    Epoch 3/10
    34799/34799 [==============================] - 173s 5ms/step - loss: 0.2233 - categorical_accuracy: 0.9328 - top_k_categorical_accuracy: 0.9954 - val_loss: 0.1542 - val_categorical_accuracy: 0.9458 - val_top_k_categorical_accuracy: 0.9948
    Epoch 4/10
    34799/34799 [==============================] - 179s 5ms/step - loss: 0.1345 - categorical_accuracy: 0.9601 - top_k_categorical_accuracy: 0.9975 - val_loss: 0.1539 - val_categorical_accuracy: 0.9474 - val_top_k_categorical_accuracy: 0.9968
    Epoch 5/10
    34799/34799 [==============================] - 161s 5ms/step - loss: 0.1081 - categorical_accuracy: 0.9677 - top_k_categorical_accuracy: 0.9981 - val_loss: 0.1336 - val_categorical_accuracy: 0.9587 - val_top_k_categorical_accuracy: 0.9973
    Epoch 6/10
    34799/34799 [==============================] - 159s 5ms/step - loss: 0.0800 - categorical_accuracy: 0.9749 - top_k_categorical_accuracy: 0.9990 - val_loss: 0.1229 - val_categorical_accuracy: 0.9689 - val_top_k_categorical_accuracy: 0.9966
    Epoch 7/10
    34799/34799 [==============================] - 161s 5ms/step - loss: 0.0733 - categorical_accuracy: 0.9774 - top_k_categorical_accuracy: 0.9989 - val_loss: 0.0953 - val_categorical_accuracy: 0.9721 - val_top_k_categorical_accuracy: 0.9984
    Epoch 8/10
    34799/34799 [==============================] - 159s 5ms/step - loss: 0.0598 - categorical_accuracy: 0.9818 - top_k_categorical_accuracy: 0.9991 - val_loss: 0.1088 - val_categorical_accuracy: 0.9673 - val_top_k_categorical_accuracy: 0.9955
    Epoch 9/10
    34799/34799 [==============================] - 171s 5ms/step - loss: 0.0545 - categorical_accuracy: 0.9834 - top_k_categorical_accuracy: 0.9994 - val_loss: 0.0844 - val_categorical_accuracy: 0.9764 - val_top_k_categorical_accuracy: 0.9975
    Epoch 10/10
    34799/34799 [==============================] - 168s 5ms/step - loss: 0.0483 - categorical_accuracy: 0.9849 - top_k_categorical_accuracy: 0.9991 - val_loss: 0.0627 - val_categorical_accuracy: 0.9805 - val_top_k_categorical_accuracy: 0.9975
    




    <keras.callbacks.History at 0x5026a048>




```python
# testset accuracy
metrics = model.evaluate(X_test, y_test_oh)
print('loss %f, accuracy %f, accuracy_5 %f' % (metrics[0], metrics[1], metrics[2]))
```

    12630/12630 [==============================] - 26s 2ms/step
    loss 0.130155, accuracy 0.965479, accuracy_5 0.995883
    


```python
# confusion matrix
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

y_test_pred = model.predict(X_test)
cm = confusion_matrix(y_test, np.argmax(y_test_pred, axis=1))
plt.figure(figsize=(20, 20))
plot_confusion_matrix(cm, classes=classes_name, title='Confusion matrix, without normalization')
```

    Confusion matrix, without normalization
    [[ 56   0   0 ...   0   0   0]
     [  0 711   7 ...   0   0   0]
     [  0   3 741 ...   2   0   0]
     ...
     [  0   0   0 ...  89   0   0]
     [  0   0   0 ...   0  58   0]
     [  0   0   0 ...   5   0  84]]
    


![png](output_25_1.png)


---

## Step 3: Test a Model on New Images

To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.

You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

### Load and Output the Images


```python
### Load the images and plot them here.
### Feel free to use as many code cells as needed.

SAMPLE_SIZE = 5
X, y = X_test[:SAMPLE_SIZE], y_test[:SAMPLE_SIZE]
name = [classes_name[label] for label in y]

plt.figure(figsize=(10, 4), dpi=100)
for i in range(SAMPLE_SIZE):
    plt.subplot(1, SAMPLE_SIZE, i + 1)
    plt.imshow(X[i])
    plt.title(name[i], fontsize=5)
    plt.axis('off')
plt.show()
```


![png](output_28_0.png)


### Predict the Sign Type for Each Image


```python
### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.
y_pred_multi = model.predict(X)
y_pred = np.argmax(y_pred_multi, axis=1)
plt.figure(figsize=(10, 4), dpi=100)
for i in range(SAMPLE_SIZE):
    plt.subplot(1, SAMPLE_SIZE, i + 1)
    plt.imshow(X[i])
    plt.title('gt:%d p:%d' % (y[i], y_pred[i]), fontsize=5)
    plt.axis('off')
plt.show()
```


![png](output_30_0.png)


### Analyze Performance


```python
### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.

### 100% correct, haha.
```

### Output Top 5 Softmax Probabilities For Each Image Found on the Web

For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 

The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.

`tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.

Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tf.nn.top_k` is used to choose the three classes with the highest probability:

```
# (5, 6) array
a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
         0.12789202],
       [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
         0.15899337],
       [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
         0.23892179],
       [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
         0.16505091],
       [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
         0.09155967]])
```

Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:

```
TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
       [ 0.28086119,  0.27569815,  0.18063401],
       [ 0.26076848,  0.23892179,  0.23664738],
       [ 0.29198961,  0.26234032,  0.16505091],
       [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
       [0, 1, 4],
       [0, 5, 1],
       [1, 3, 5],
       [1, 4, 3]], dtype=int32))
```

Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.


```python
### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.
print("prediction top 5:")
print(np.argsort(-y_pred_multi, axis=1)[:, :SAMPLE_SIZE])

print("prediction groundtruth")
print(y_test[:SAMPLE_SIZE])
```

    prediction top 5:
    [[16 11 32  5  9]
     [ 1  2  0 24  4]
     [38 40 36 34 39]
     [33 36 40 11 34]
     [11 27 30 33  0]]
    prediction groundtruth
    [16  1 38 33 11]
    

### Project Writeup

Once you have completed the code implementation, document your results in a project writeup using this [template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) as a guide. The writeup can be in a markdown or pdf file. 

> **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

---

## Step 4 (Optional): Visualize the Neural Network's State with Test Images

 This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.

 Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the [LeNet lab's](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.

For an example of what feature map outputs look like, check out NVIDIA's results in their paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.

<figure>
 <img src="visualize_cnn.png" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Your output should look something like this (above)</p> 
 </figcaption>
</figure>
 <p></p> 



```python
### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(conv_features, activation_min=-1, activation_max=-1 ,plt_num=1):
    featuremaps = conv_features.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(conv_features[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(conv_features[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(conv_features[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(conv_features[0,:,:, featuremap], interpolation="nearest", cmap="gray")
```


```python
from keras import backend as K

intermediate_result = K.function(inputs=[model.layers[0].input, K.learning_phase()],
    outputs=[
     model.get_layer("block1_conv1").output,
     model.get_layer("block1_conv2").output,
     model.get_layer("block2_conv1").output,
     model.get_layer("block2_conv2").output,
     model.get_layer("block3_conv1").output,
     model.get_layer("block3_conv2").output])
```


```python
block1_conv1 = intermediate_result([X, 0])[0]
outputFeatureMap(block1_conv1)
```


![png](output_41_0.png)



```python
block1_conv2 = intermediate_result([X, 0])[1]
outputFeatureMap(block1_conv2)
```


![png](output_42_0.png)



```python
block2_conv1 = intermediate_result([X, 0])[2]
outputFeatureMap(block2_conv1)
```


![png](output_43_0.png)



```python
block2_conv2 = intermediate_result([X, 0])[3]
outputFeatureMap(block2_conv2)
```


![png](output_44_0.png)



```python
block3_conv1 = intermediate_result([X, 0])[4]
outputFeatureMap(block3_conv1)
```


![png](output_45_0.png)



```python
block3_conv2 = intermediate_result([X, 0])[5]
outputFeatureMap(block3_conv2)
```


![png](output_46_0.png)

