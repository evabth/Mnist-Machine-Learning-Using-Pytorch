import mnistLearning
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn


batch_size = 16

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform= transform)


trainloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True, num_workers=0)

#mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

#testloader = torch.utils.data.DataLoader(mnist_testset, batch_size= batch_size, shuffle=False, num_workers=0)



model = mnistLearning.cnnMnist()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

i = 0

for data in trainloader:
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if ((i-1) % 999 ==0):
                print("1000 images done")

        i += 1


print("Finished")

torch.save(model.state_dict(), 'mnistSave.pt')