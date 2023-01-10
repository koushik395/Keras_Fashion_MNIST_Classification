# Keras_Fashion_MNIST_Classification

### *About DataSet*
The very popular Fashion MNIST dataset to classify clothing,footwear and other related items.So,lets dive in.<br>
###### What actually is Fashion MNIST ? <br>
Fashion-MNIST is a dataset of Zalando’s article images consisting of a training set of 60,000 examples and a test set of 10,000 examples.It’s great for writing “hello world” tutorials for deep learning.This dataset contains 28 * 28px of clothing-greyscale images. The library contains 10 types of clothing items:

<ul>
<li>T-shirt/top</li>
<li>Trouser</li>
<li>Pullover</li>
<li>Dress</li>
<li>Coat</li>
<li>Sandal</li>
<li>Shirt</li>
<li>Sneaker</li>
<li>Bag</li>
<li>Ankle Boot</li>
</ul>

![Alt text](https://miro.medium.com/max/1100/1*ymrqRtMnRIy4IM4IexLr9g.webp "")

So, given an input image, these would be our possible outputs. In total, the Fashion MNIST dataset contains 70,000 images which are undoubtedly plenty for us to work with. Out of the 70,000 images, we will use 60,000 of them to train the neural network with the other 10,000 being used to test the neural network. Also, remember that each image is a 28px x 28px image meaning that there are 784 pixels. And so, the job would simply be to take the 784 pixels as input and then output one of the 10 different items of clothing the image represents.<br>

Let’s take a quick look at how our neural network would look like:<br>
<br>
What happens here is that the neural network can’t work with a 2-D image (the 28 x 28 image) and can only work with a 1-D image (array). So, what we’d do here is compress the image into a 1-D array by multiplying the length by the height. That’s where the 784 comes from (28 x 28). This is known as a method called *flattening*. So, our input layer would be 784 neurons in this case. Our hidden layer, in this case, would be of 128 neurons and then from there, show 10 outputs which were the possible types of clothing.

![Alt text](https://miro.medium.com/max/1400/1*S6t_smvyXvXnDAO7UkL4MA.webp "")
### *Data visualization*
Now we will see some of the sample images from the fashion MNIST dataset. For this, we will use the library matplotlib to show our np array data in the form of plots of images.

```

for i in range(1, 10):
    # Create a 3x3 grid and place the
    # image in ith position of grid
    plt.subplot(3, 3, i)
    # Insert ith image with the color map 'grap'
    plt.imshow(trainX[i], cmap=plt.get_cmap('gray'))
 
# Display the entire plot
plt.show()

```
![Alt text](https://media.geeksforgeeks.org/wp-content/uploads/20220408134114/sampleimage.png "")

### *Model Training*
```
#build
model = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),                           
  tf.keras.layers.Dense(128, activation='relu'), 
  tf.keras.layers.Dense(64, activation='relu'),                              
  tf.keras.layers.Dense(10),                             
  ])
  
  ```
 As you can see, we’re using the TensorFlow library to build our neural network. Let’s go a bit more in-depth into what we’re doing:
  
<ul>
<li>
<b>input</b> tf.keras.layers.Flatten - This layer transforms a 2-d array (matrix) into a 1-D array of 784 (28 x 28). Think of this layer as lining up the images from a square to one, long line. This layer doesn't learn anything; it simply reshapes the data.
</li>
<li>
<b>hidden</b> tf.keras.layers.Dense- A densely connected layer of 128 neurons. Each neuron (otherwise known as a node) takes input from all 784 nodes in the previous layer, weighting that input according to hidden parameters which will be learned during training, and outputs a value to the next layer.
</li>
<li>
<b>hidden</b> tf.keras.layers.Dense- A densely connected layer of 64 neurons. Each neuron (otherwise known as a node) takes input from all 128 nodes in the previous layer, weighting that input according to hidden parameters which will be learned during training, and outputs a single value to the next layer.
</li>
<li>
<b>output</b> tf.keras.layers.Dense This is a 10 node softmax layer with each node representing a class of clothing. As in the previous layer, each node takes input from the 64 nodes in the layer before it, weights that input according to learned parameters, and then outputs a value in the form of [0, 1] which of course, represents the probability of of the image belonging to that class. The sum of all 10 nodes is 1.
</li>
</ul>
 
Before we finish the model, we need to compile it:<br>
```
#compile
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```

<ul>
<li>Loss function — An algorithm for measuring how far the model’s outputs are from the desired output. The goal of training is this measure’s loss.</li>
<li>Optimizer — An algorithm for adjusting the inner parameters of the model to minimize loss.</li>
<li>Metrics — Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified:</li>
</ul>

### *Training the Model:*
```
model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_split=0.1)

```
Training is performed by calling the model.fit method:<br>
<ul>
<li>Feed the training data to the model using train_dataset</li>
<li>The model learns to associate images and labels.</li>
<li>The epochs=5 parameter limits training to 10 full iterations of the training dataset, so a total of 10 * 60000 = 600000 examples.</li>
<li>The batch_size=64 tells model.fit to use batches of 62 images and labels when updating the model variables.</li>
</ul>
And now, voila! Our model is undergoing training! <br>

As you can see, the loss and accuracy metrics are clearly displayed. After running the 10th epoch, we can see that we have a loss of ~24% with our model being 90% accurate.

*---------------------EPOCHS-----------*

### *Model Visualization*
Keras plot model is defined in tensorflow, a machine learning framework provided by Google. Basically, it is an open source that was used in the Tensorflow framework in conjunction with Python to implement the deep learning algorithm. It contains the optimization technique that was used to perform the complicated and mathematical operations. It is more scalable and will support multiple platforms.<br>

```
# FashionMNIST
tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

```
###### parameters:
<ul>
<li><b>model</b>	     :    A Keras model instance</li>
<li><b>show_shapes</b>	 :    whether to display shape information.</li>
<li><b>rankdir</b>	     : rankdir argument passed to PyDot, a string specifying the format of the plot: 'TB' creates a vertical plot; 'LR' creates a horizontal plot.</li>
<ul>

<br>
But wait…. it’s still not over yet. What if your dream classification machine made a mistake? What if 10% of all your clothes were classified incorrectly? What if I were to tell you that it’s possible for your machine to be able to classify your clothes with a 95+% accuracy? Welcome to the world of *Convolutions and Maxpooling.*
