import mnistLearning
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt


batch_size = 16

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])

mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform = transform)

testloader = torch.utils.data.DataLoader(mnist_testset, batch_size= batch_size, shuffle=False, num_workers=0)


model = mnistLearning.cnnMnist()
model.load_state_dict(torch.load('mnistSave.pt'))

correct = 0
total = 0
numbersWrong = [0] * 10
imageList = []


print(len(testloader))

with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = model(images)
        # the class with the highest likelyhood is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for i in range(len(labels)):
            if (labels[i].item()!=outputs[i].argmax()):
                numbersWrong[labels[i].item()] += 1
                imageList += images[i]

        


print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

print(numbersWrong)

print(len(imageList))

fig, ax = plt.subplots()

def close_event(event):
    plt.close()

fig.canvas.mpl_connect('close_event', close_event)

for image in imageList:
    ax.imshow(image)
    plt.draw()
    plt.waitforbuttonpress()  # Small pause to allow the window to be responsive
    if not plt.fignum_exists(fig.number):
        break
    ax.cla()

plt.close()