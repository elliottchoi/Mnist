import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import transforms
import torchvision.datasets as dsets


#Hyper Parameters
#pixel width of a photo is 28*28
input_size=784
#number of possible y outputs {0,1,2....8,9}
C_class=10
#number of back and forward passes
num_epochs=3
#learning rate
learning_rate=0.0001
batch_size=100
#hiddenLayer
hiddenLayer1=500
hiddenLayer2=400
hiddenLayer3=300
hiddenLayer4=200
hiddenLayer5=100
hiddenLayer6=50

'''
Data Set Construction
'''
#MNST Dataset
train_dataset=dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset=dsets.MNIST(root='.data',
                            train=False,
                            transform=transforms.ToTensor(),
                            download=True)

print('The length of training set is %d and the length of testing set is %d'%(len(train_dataset),len(test_dataset)))

#data that we are using to train the neural network from MNIST data set
train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

#Shuffle the data set MNIST and use it to test the neural network
test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)




'''
Neural Network
'''
#Architechture for neural network
class Net(nn.Module):
    def __init__(self,input_size,hiddenLayer1,hiddenLayer2,hiddenLayer3,hiddenLayer4,hiddenLayer5,hiddenLayer6,C_class):
        super(Net,self).__init__()
        #setting up the architechture for neural network
        self.fc1=nn.Linear(input_size,hiddenLayer1)
        #take max of function and 0
        self.relu=nn.ReLU()
        #Layers:
        self.fc2 = nn.Linear(hiddenLayer1, hiddenLayer2)
        self.fc3 = nn.Linear(hiddenLayer2, hiddenLayer3)
        self.fc4 = nn.Linear(hiddenLayer3, hiddenLayer4)
        self.fc5 = nn.Linear(hiddenLayer4, hiddenLayer5)
        self.fc6 = nn.Linear(hiddenLayer5, hiddenLayer6)
        self.fc7 = nn.Linear(hiddenLayer6, C_class)

    def forward(self,x):

        #connecates through the neural network layers
        #relu= rectified linear unit: f(x) = log (1+exp x),
        #activation function focusing on the positive part of the arguement
        out=self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.relu(self.fc4(out))
        out = self.relu(self.fc5(out))
        out = self.relu(self.fc6(out))
        y_predict = self.relu(self.fc7(out))
        #Because I am optimizing negative log likelihood, I need softmax for logarithmic probabilties
        return y_predict

net=Net(input_size,hiddenLayer1,hiddenLayer2,hiddenLayer3,hiddenLayer4,hiddenLayer5,hiddenLayer6,C_class)
#checking the structure of the neural network
print("For our purposes, the neural network is a 3 layered neural net ",net)

'''
Training Loop
'''
#Setting up criteria and loss function
#Loss Function, use of CrossEntropy means we didnt have to use soft max, this is numerically more stable
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(net.parameters(), lr=learning_rate)


#training loop

for epoch in range (num_epochs):
    aggregate_epoch = 0
    placeholder_i=0
    for i, (images, labels)in enumerate(train_loader):
        #wrap both the data and target from train_loader with Variable wrapper for auto_grad
        images,labels=Variable(images),Variable(labels)
        #resize data from batch_size,1,28,28 to (Batch_size,784)
        images=images.view(-1,28*28)
        #set grads to zero
        optimizer.zero_grad()

        #Forward propogation
        y_predict=net(images)
        #Checking Negative Log Loss
        loss=criterion(y_predict,labels)
        #Backward propogation
        loss.backward()
        optimizer.step()

        #Spits out every 10 of a batch of 200
        if (i+1)%batch_size==0:
            print('Epoch [%d/%d], Interation [%d/%d], Loss: %.6f'
                  % (epoch + 1, num_epochs, (i + 1)*len(images), len(train_dataset) , loss.data[0]))
            aggregate_epoch+=loss.data[0]
        placeholder_i=i
    print('Mean Loss for Epoch: %.6f'%(aggregate_epoch/(placeholder_i/batch_size)))

'''
Testing Loop
'''

#run a test loop
total=0
#correct counter
correct = 0
for images, labels in test_loader:
    #wrap the values from test_loader in Variable Wrapper
    images = Variable(images)
    #connecate to 1 by 784
    images = images.view(-1, 28*28)
    y_predict = net(images)
    # sum up batch loss
    #return the C class with max log probability
    _,predicted=torch.max(y_predict.data,1)
    total+=labels.size(0)
    #only sum if target data is equal to y_predict
    correct += (predicted==labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# Save the Model
torch.save(net.state_dict(), 'Mnist.pkl')




