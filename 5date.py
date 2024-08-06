#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install tensorflow


# In[2]:


import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical, plot_model
import matplotlib.pyplot as plt
import numpy as nnp
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from warnings import filterwarnings
filterwarnings('ignore')


# In[3]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Training set size:", x_train.shape, y_train.shape)
print("Test set sixe:", x_test.shape, y_test.shape)


# In[4]:


import numpy as np
num_labels = len(np.unique(y_train))
num_labels


# In[5]:


plt.figure(figsize=(5,5))
plt.imshow(x_train[560], cmap='gray')


# In[6]:


plt.figure(figsize=(5,5))
for i in range(0, 10):
    ax = plt.subplot(5, 5, i+1)
    plt.imshow(x_train[i], cmap = 'gray')
    plt.axis('off')


# In[7]:


def visualize_img(data, num =10):
    plt.figure(figsize=(5,5))
    for i in range(0, num):
        ax = plt.subplot(5,5, i+1)
        plt.imshow(data[i], cmap='gray')
        plt.axis('off')


# In[8]:


visualize_img(x_train, 20)


# In[9]:


def pixel_visualize(img):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap = 'gray')
    width, height = img.shape
    
    threshold = img.max()/ 2.5
    
    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(img[x][y], 2)), xy = (y,x),
                       color = 'white' if img[x][y] < threshold else 'black')


# In[10]:


pixel_visualize(x_train[2])


# In[11]:


y_train[0:5]


# In[12]:


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# In[14]:


y_train[0:5]


# In[15]:


image_size = x_train.shape[1]
image_size


# In[16]:


print(f"x_train size: {x_train.shape}\n\nx_test size: {x_test.shape}")


# In[18]:


x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28 ,1)


# In[20]:


print(f"x_train size: {x_train.shape}\n\nx_test size: {x_test.shape}")


# In[21]:


x_train = x_train.astype('float32')/ 255
x_test = x_test.astype('float32')/ 255


# In[33]:


model=tf.keras.Sequential([
    Flatten(input_shape =(28, 28,1)), 
    Dense(units = 128, activation ="relu", name = "layer1"),
    Dense(units = num_labels, activation = "softmax", name = "output_layer")])
model.compile(loss = "categorical_crossentropy",
             optimizer = "adam",
             metrics = [tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), "accuracy"])


# In[34]:


model.summary()


# In[35]:


model.fit(x_train, y_train, epochs = 8, batch_size =128,
          validation_data=(x_test, y_test))


# In[36]:


k=model.fit(x_train, y_train, epochs = 8, batch_size =128,
          validation_data=(x_test, y_test))


# In[40]:


plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
plt.plot(k.history["accuracy"], color = "b",
        label = "Training Accuracy")
plt.plot(k.history["val_accuracy"], color = "r",
        label="validation Accuracy")
plt.legend(loc= "lower right")
plt.xlabel("Epoch", fontsize = 16)
plt.ylabel("Accuracy", fontsize = 16)
plt.ylim([min(plt.ylim()),1])
plt.title("Training and Test Accuracy Graph", fontsize = 16)
plt.subplot(1,2,2)
plt.plot(k.history["loss"], color = "b", label = "Training loss")
plt.plot(k.history["val_loss"], color = "r",
        label = "validation loss")
plt.legend(loc ="upper right")
plt.xlabel("Epoch", fontsize =16)
plt.ylabel("Loss", fontsize = 16)
plt.ylim([0, max(plt.ylim())])
plt.title("Training and Test loss Graph", fontsize = 16)


# In[41]:


loss, precision, recall, acc= model.evaluate(x_test, y_test, verbose=False)
print(f"Test accuracy: {round(acc*100, 2)}")
print(f"Test loss: {round(loss*100, 2)}")
print(f"Test precision: {round(precision*100, 2)}")
print(f"Test recall: {round(recall*100, 2)}")


# In[42]:


y_pred = model.predict(x_test)


# In[43]:


y_pred_classes = np.argmax(y_pred, axis = 1)


# In[45]:


if len(y_test.shape)> 1 and y_test.shape[1] !=1:
    y_test = np.argmax(y_test, axis =1)


# In[49]:


cm = confusion_matrix(y_test,y_pred_classes)
plt.figure(figsize=(10,7))
sns.heatmap(cm,annot =True, fmt= 'd', cmap="Blues")
plt.xlabel('predicted')
plt.ylabel('True')
plt.title('confusion matrix')
plt.show()


# In[51]:


model.save('mnist_model.h5')


# In[52]:


import random
random = random.randint(0, x_test.shape[0])
print(y_test[random])
test_image = x_test[random]
print(y_test[random])


# In[53]:


plt.figure(figsize=(5,5))
plt.imshow(test_image.reshape(28,28), cmap="gray")


# In[54]:


test_data = x_test[random].reshape(1 , 28, 28, 1)


# In[55]:


probability = model.predict(test_data)


# In[56]:


predicted_classes =np.argmax(probability)


# In[ ]:


print(f"predicted class: {predicted_classes}\nprobability
     value of pre)

