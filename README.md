# Neural-Network---Tensorflow
Classification Tutorial Using TensorFlow Keras and Deep learning


Introduction

Machine Learning algorithm is a function that can tune variables in order to map our input to the output. Training of these data is done for thousands or even millions of iterations over the input and output data. In general, a ML problem has an input and an output but the algorithm to find these outputs is what must be learnt. The Neural networks are in between these inputs and outputs

Neural Network can be defined as a stack of layers where each layer has a predefined Math and internal variables. Each layer is made up of units also called as Neurons.

In Neural Networks the math and internal variables are applied, and the resulting output is produced. In order to produce the results a neural network must be trained repeatedly to map the inputs to the outputs. In general, while training the neural networks it involves the tuning of the internal variables in the layers until the network provides the output for the new inputs.

Fashion MNIST is a dataset of Zalando's article images, the dataset contains a set of 60000 train examples and 10000 test examples. Each of the images in this dataset is 28X28 greyscale image and it is associated with a label from 10 classes. This dataset can be used as a drop-in replacement for MNIST. The class labels are:

Label	Description
0	T-shirt/top
1	Trouser
2	Pullover
3	Dress
4	Coat
5	Sandal
6	Shirt
7	Sneaker
8	Bag
9	Ankle boot

Input Images (28*28 =784 pixels) - This is a flattened layer with images as a one long string each of 784 pixels.

Each of the 784 pixels in the input layer is connected to each of the 128 Neurons in the dense layer. Dense layer will adjust the weights and bias in the training phase.

The output layer consists of 10 Neurons corresponding to each of 10 classes. Each of the neuron will give a probability score for all the classes and the final output will be the output neuron with the highest probability.

To make this more robust, Intoduce CNN's - Convolutional Neural Networks.

Conclusion:

The input shape of the data plays a critical role in selection of layers and their depth. Selecting oversized/ undersized layers may lead to overfitting/ underfitting, this can learnt by experience, working on number of datasets regularly.
The model accuracies depend on the number of convolutional layer selected with proper number of filters, selecting the number of dense layers with the activation function is important for a building a better model.
Cross-Validation plays a critical role in deciding the model performance, various models should be tried before applying to entire train and test dataset.
References:

https://www.tensorflow.org/tutorials - For Basic syntax.

https://classroom.udacity.com/courses/ud187/lessons/1771027d-8685-496f-8891-d7786efb71e1/concepts/8b8c3d93-4117-4134-b678-77d54634b656 - For understanding concepts.

Dr. Timothy Havens - https://mtu.instructure.com/courses/1304186/modules - For understanding concepts.

https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-fashion-mnist-clothing-classification/ - For basic structure or Idea of the Tutorial.
