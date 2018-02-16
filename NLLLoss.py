import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

def simple_gradient():
    # print the gradient of 2x^2 + 5x
    x = Variable(torch.ones(2, 2) * 2, requires_grad=True)
    z = 2 * (x * x) + 5 * x
    # run the backpropagation
    z.backward(torch.ones(2, 2))
    print(x.grad)

#Enumerator
def create_nn(batch_size=200, learning_rate=0.01, epochs=10, log_interval=10):

    #data that we are using to train the neural network from MNIST data set
    train_loader=torch.utils.data.DataLoader(datasets.MNIST('../data',train=True,download=True,
                                                           transform=transforms.Compose([transforms.ToTensor(),
                                                                                          transforms.Normalize((0.1307,),(0,3081,))
                                                                                         ])),batch_size=batch_size,shuffle=True)

    #Shuffle the data set MNIST and use it to test the neural network
    test_loader=torch.utils.data.DataLoader(datasets.MNIST('../data',train=False,download=True,
                                                          transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(
                                                          (0.1307,),(0.3081,))])),batch_size=batch_size,shuffle=True)



    #Architechture for neural network
    class Net(nn.Module):
        def __init__(self):
            super(Net,self).__init__()
            #setting up the architechture for neural network
            self.fc1=nn.Linear(784,200)
            self.fc2=nn.Linear(200,200)
            self.fc3=nn.Linear(200,10)

        def forward(self,x):

            #connecates through the neural network layers
            #relu= rectified linear unit: f(x) = log (1+exp x),
            #activation function focusing on the positive part of the arguement
            x1=F.relu(self.fc1(x))
            x2=F.relu(self.fc2(x1))
            y_predict=F.relu(self.fc3(x2))
            #Because I am optimizing negative log likelihood, I need softmax for logarithmic probabilties
            return F.log_softmax(y_predict)

    net=Net()
    #checking the structure of the neural network
    print("For our purposes, the neural network is a 3 layered neural net ",net)


    #Setting up criteria and loss function
    learning_rate=0.01
    #Training the network:
    optimizer=optim.SGD(net.parameters(), lr=learning_rate,momentum=0.9)
    #Loss Function
    criterion=nn.NLLLoss()

    #training loop

    for epoch in range (epochs):
        for batch_idx,(data,target)in enumerate(train_loader):
            #wrap both the data and target from train_loader with Variable wrapper for auto_grad
            data,target=Variable(data),Variable(target)
            #resize data from batch_size,1,28,28 to (Batch_size,784)
            data=data.view(-1,784)
            #set grads to zero
            optimizer.zero_grad()

            #Forward propogation
            y_predict=net(data)
            #Checking Negative Log Loss
            loss=criterion(y_predict,target)
            #Backward propogation
            loss.backward()
            optimizer.step()

            #Spits out every 10 of a batch of 200
            if batch_idx % log_interval==0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data[0]))

    #run a test loop
    test_loss = 0
    #correct counter
    correct = 0
    for data, target in test_loader:
        #wrap the values from test_loader in Variable Wrapper
        data, target = Variable(data, volatile=True), Variable(target)
        #connecate to 1 by 784
        data = data.view(-1, 784)
        y_predict = net(data)
        # sum up batch loss
        test_loss += criterion(y_predict,target).data[0]
        #return the C class with max log probability
        prediction = y_predict.data.max(1)[1]
        #only sum if target data is equal to y_predict
        correct += prediction.eq(target.data).sum()

    test_loss /= len(test_loader.dataset)
    #how many we got correct/length of data set
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    # Save the Model
    torch.save(net.state_dict(), 'Mnist.pkl')


if __name__ == "__main__":
    run_opt = 2
    if run_opt == 1:
        simple_gradient()
    elif run_opt == 2:
        create_nn()


