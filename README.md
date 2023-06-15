# Mnist-Machine-Learning-Using-Pytorch
Using Pytorch, this repo will learnhow to classify different numbers based off of the Mnist Database

Goal: Create a Neural Network that can accurately classify the numbers 0-9 as defined by the MNIST database

1. The first part of this project is creating the neural network that is structured to understand the information contained in a 28X28 grey scale image. My implementation of the neural network is contained in the mnistLearning.py file. The neural network uses convolutions of the original image, followed by max pooling, which is then provided to the neural network. The neural network contains 3 layers starting. The neurons associated with each layer goes as follows 484(1st)-->242(2nd)-->121(3rd)-->10(output) *the high number of neurons in each layer is due to the batch size which is 16*.

2. The next step is training and saving the associated weights and biases that are acheieved in the gradient descent when providing training data. The training occurs in mnistTrain.py file. This will train the neural network on 3750 images from the train dataset of the MNIST database. Then the weights and biases are saved to the mnistSave.pt file.

3. Finally the save data is tested for accuracy on 10,000 images using the test dataset in the MNIST database. The testing occurs in the mnistTesting.py file and achieved an accuracy of 96%.
