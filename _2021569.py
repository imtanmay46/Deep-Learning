import torch
import torchaudio
import torchvision
import torch.nn as nn
from Pipeline import *
from torch.utils.data import *
import numpy as num
import random as rand
import math
from sklearn.preprocessing import *

def helper(x):
    if(x==0):
        return False
    return True

"""
Write Code for Downloading Image and Audio Dataset Here
"""
# Image Downloader
image_dataset_downloader = torchvision.datasets.CIFAR10(
    root='./data', train=helper(0), download=helper(0), transform=torchvision.transforms.ToTensor()
)

# Audio Downloader
audio_dataset_downloader = torchaudio.datasets.SPEECHCOMMANDS(
    root='./data', download=helper(0), subset='testing'
)

image_idx = list(range(len(image_dataset_downloader)))
num.random.shuffle(image_idx)

image_split = int(num.floor(0.5*(len(image_dataset_downloader))))
image_val_idx, image_test_idx = image_idx[image_split:], image_idx[:image_split]

image_val_set = Subset(image_dataset_downloader, image_val_idx)
image_test_set = Subset(image_dataset_downloader, image_test_idx)

class ImageDataset(Dataset):
    def __init__(self, split:str="train") -> None:
        super().__init__()
        if split not in ["train", "test", "val"]:
            raise Exception("Data split must be in [train, test, val]")
        
        self.datasplit = split
        """
        Write your code here
        """
        if self.datasplit == 'train':
            gen_data = torchvision.datasets.CIFAR10(root='./data', train=helper(1), download=helper(0), transform=torchvision.transforms.ToTensor())
            image_train_idx = rand.sample(range(len(gen_data)), len(gen_data))
            self.data = Subset(gen_data, image_train_idx)
        elif self.datasplit == 'test':
            self.data = image_test_set
        else:
            self.data = image_val_set
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        x, y = self.data[i]
        return x, y


class AudioDataset(Dataset):
    def __init__(self, split: str = "train", transform=None):
        super().__init__()
        if split not in ["train", "test", "val"]:
            raise Exception("Data split must be in [train, test, val]")
        
        self.datasplit = split
        """
        Write your code here
        """
        self.data = torchaudio.datasets.SPEECHCOMMANDS(root='./data', url='speech_commands_v0.02', download=helper(1))
        self.encoder = LabelEncoder()
        self.T = transform
        self.classes = ["house", "zero", "one", "tree", "three", "eight", "learn", "six", "nine", "right", "two",
                       "five", "left", "backward", "follow", "off", "marvin", "sheila",
                       "forward", "down", "visual", "bird", "four", "no", "stop",
                       "wow", "bed", "cat", "go", "yes", "dog", "happy", "on", "seven", "up"]
        self.encoder.fit(self.classes)
        size_of_training_set = int(len(self.data) * 0.8)
        size_of_testing_set = int((len(self.data) - size_of_training_set) * (0.1 / (1 - 0.8)))
        size_of_validation_set = (len(self.data) - size_of_training_set) - size_of_testing_set

        training_set, performance_set = random_split(self.data, [size_of_training_set, (len(self.data) - size_of_training_set)])
        testing_set, validation_set = random_split(performance_set, [size_of_testing_set, size_of_validation_set])

        if split == "train":
            self.data = training_set
        elif split == "test":
            self.data = testing_set
        else:
            self.data = validation_set        
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        val = 16000
        x, _, y, *_ = self.data[i]

        y = self.encoder.transform([y])[0]
        audio_shape = x.shape[1]

        if audio_shape < val:
            x = torch.cat((x, torch.zeros(1, val - audio_shape)), dim=1)
        
        if self.T:
            x = self.T(x)

        return x, y

'''
___________________________________________________________________________________________________________________________________________________________________
___________________________________________________________________________________________________________________________________________________________________
'''

class ResBlock(nn.Module):
    def __init__(self, cin, cout, flag=0):
        super(ResBlock, self).__init__()
        if not helper(flag):
            self.conv1 = nn.Conv2d(cin, cout, kernel_size=3, padding=1)
            self.batch_norm1 = nn.BatchNorm2d(cout)
            self.conv2 = nn.Conv2d(cout, cout, kernel_size=3, padding=1)
            self.batch_norm2 = nn.BatchNorm2d(cout)
        else:
            self.conv1 = nn.Conv1d(cin, cout, kernel_size=3, padding=1)
            self.batch_norm1 = nn.BatchNorm1d(cout)
            self.conv2 = nn.Conv1d(cout, cout, kernel_size=3, padding=1)
            self.batch_norm2 = nn.BatchNorm1d(cout)
            
        self.ReLU = nn.ReLU()

    def forward(self, x):
        res = x
        z = self.conv1(x)
        z = self.batch_norm1(z)
        z = self.ReLU(z)
        z = self.conv2(z)
        z = self.batch_norm2(z)
        z += res
        z = self.ReLU(z)
        return z


class Resnet_Q1(nn.Module):
    def __init__(self):
        super(Resnet_Q1, self).__init__()
        """
        Write your code here
        """
        self.one_d_labels = 35
        self.two_d_labels = 10
        self.out = 1
        self.ReLU = nn.ReLU()

        self.conv1d = nn.Conv1d(1, self.out, kernel_size=3, padding=1)
        self.batch_norm1d = nn.BatchNorm1d(self.out)
        self.resBlock1d = nn.Sequential(*[ResBlock(self.out, self.out, 1) for _ in range(0, 18)])
        self.FC1d = nn.Linear(self.out*16000, self.one_d_labels)

        self.conv2d = nn.Conv2d(3, self.out, kernel_size=3, padding=1)
        self.batch_norm2d = nn.BatchNorm2d(self.out)
        self.resBlock2d = nn.Sequential(*[ResBlock(self.out, self.out, 0) for _ in range(0, 18)])
        self.FC2d = nn.Linear(self.out*32*32, self.two_d_labels)

    def forward(self, x):
        if len(x.shape) == 3:
            if x.shape[1] == 1:
                z = self.conv1d(x)
                z = self.batch_norm1d(z)
                z = self.ReLU(z)
                z = self.resBlock1d(z)
                z = z.view(z.size(0), -1)
                z = self.FC1d(z)
            else:
                z = self.conv2d(x)
                z = self.batch_norm2d(z)
                z = self.ReLU(z)
                z = self.resBlock2d(z)
                z = z.view(z.size(0), -1)
                z = self.FC2d(z)
        else:
            z = self.conv2d(x)
            z = self.batch_norm2d(z)
            z = self.ReLU(z)
            z = self.resBlock2d(z)
            z = z.view(z.size(0), -1)
            z = self.FC2d(z)
        return z




'''
___________________________________________________________________________________________________________________________________________________________________
___________________________________________________________________________________________________________________________________________________________________
'''


class VGG_Q2(nn.Module):
    def __init__(self):
        super(VGG_Q2, self).__init__()
        """
        Write your code here
        """
        self.flag = 0
        self.features = 0
        self.model = nn.Sequential()

        c = [int(math.ceil(8 * (0.65 ** _))) for _ in range(0, 5)]
        k = [int(math.ceil(3 * (1.25 ** _))) for _ in range(0, 5)]

        self.features_2d = nn.Sequential(
            self.convBlock2d(3, c[0], k[0], 2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.convBlock2d(c[0], c[1], k[1], 2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.convBlock2d(c[1], c[2], k[2], 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.convBlock2d(c[2], c[3], k[3], 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.convBlock2d(c[3], c[4], k[4], 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.features_1d = nn.Sequential(
            self.convBlock1d(1, c[0], k[0], 2),
            nn.MaxPool1d(kernel_size=2, stride=2),
            self.convBlock1d(c[0], c[1], k[1], 2),
            nn.MaxPool1d(kernel_size=2, stride=2),
            self.convBlock1d(c[1], c[2], k[2], 3),
            nn.MaxPool1d(kernel_size=2, stride=2),
            self.convBlock1d(c[2], c[3], k[3], 3),
            nn.MaxPool1d(kernel_size=2, stride=2),
            self.convBlock1d(c[3], c[4], k[4], 3),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )


    def layer_one_d(self, cin, cout, kernel_size):
        return [nn.Conv1d(cin, cout, kernel_size=kernel_size, padding=kernel_size//2), nn.BatchNorm1d(cout), nn.ReLU(inplace=helper(1))]

    def layer_two_d(self, cin, cout, kernel_size):
        return [nn.Conv2d(cin, cout, kernel_size=kernel_size, padding=kernel_size//2), nn.BatchNorm2d(cout), nn.ReLU(inplace=helper(1))]

    def convBlock1d(self, cin, cout, kernel_size, convolutions):
        layers = self.layer_one_d(cin, cout, kernel_size)

        i=1
        while(i<convolutions):
            layers.extend(self.layer_one_d(cout, cout, kernel_size))
            i+=1

        return nn.Sequential(*layers)
    
    def convBlock2d(self, cin, cout, kernel_size, convolutions):
        layers = self.layer_two_d(cin, cout, kernel_size)
        
        i=1
        while(i<convolutions):
            layers.extend(self.layer_two_d(cout, cout, kernel_size))
            i+=1

        return nn.Sequential(*layers)

    def forward(self, x):
        self.one_d_labels = 35
        self.two_d_labels = 10

        if len(x.shape) == 3:
            if x.shape[1] == 1:
                x = self.features_1d(x)
                if not helper(self.flag):
                    with torch.no_grad():
                        self.features = 1004
                        self.model = nn.Sequential(
                            nn.Linear(self.features, self.one_d_labels),
                            nn.ReLU(helper(1)),)
                        self.flag = 1
            else:
                x = self.features_2d(x)
                if not helper(self.flag):
                    with torch.no_grad():
                        shape1 = x.shape[1]
                        shape2 = x.shape[2]
                        shape3 = x.shape[3]
                        self.features = shape1 * shape2 * shape3

                        self.model = nn.Sequential(
                            nn.Linear(self.features, 4096),
                            nn.ReLU(helper(1)),
                            nn.Dropout(),
                            nn.Linear(4096, 4096),
                            nn.ReLU(helper(1)),
                            nn.Dropout(),
                            nn.Linear(4096, self.two_d_labels),
                        )
                        self.flag = 1
        else: 
            x = self.features_2d(x)
            if not helper(self.flag):
                with torch.no_grad():
                    shape1 = x.shape[1]
                    shape2 = x.shape[2]
                    shape3 = x.shape[3]
                    self.features = shape1 * shape2 * shape3

                    self.model = nn.Sequential(
                        nn.Linear(self.features, 4096),
                        nn.ReLU(helper(1)),
                        nn.Dropout(),
                        nn.Linear(4096, 4096),
                        nn.ReLU(helper(1)),
                        nn.Dropout(),
                        nn.Linear(4096, self.two_d_labels),
                    )
                    self.flag = 1

        x = torch.flatten(x, 1)
        x = self.model(x)
        return x



'''
___________________________________________________________________________________________________________________________________________________________________
___________________________________________________________________________________________________________________________________________________________________
'''

class InceptionBlock(nn.Module):
    def __init__(self, cin, cout, flag = 0):
        self.flag = flag
        super(InceptionBlock, self).__init__()

        self.one_d_branch1 = nn.Sequential(
            nn.Conv1d(cin, cout, kernel_size=1),
            nn.BatchNorm1d(cout),
            nn.ReLU(inplace=helper(1))
        )

        self.one_d_branch2 = nn.Sequential(
            nn.Conv1d(cin, cout, kernel_size=3, padding=1),
            nn.BatchNorm1d(cout),
            nn.ReLU(inplace=helper(1)),
            nn.Conv1d(cout, cout, kernel_size=5,padding=2),
            nn.BatchNorm1d(cout),
            nn.ReLU(inplace=helper(1))
        )

        self.one_d_branch3 = nn.Sequential(
            nn.Conv1d(cin, cout, kernel_size=3,padding=1),
            nn.BatchNorm1d(cout),
            nn.ReLU(inplace=helper(1)),
            nn.Conv1d(cout, cout, kernel_size=5,padding=2),
            nn.BatchNorm1d(cout),
            nn.ReLU(inplace=helper(1))
        )

        self.one_d_branch4 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

        self.two_d_branch1 = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=helper(1))
        )

        self.two_d_branch2 = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=helper(1)),
            nn.Conv2d(cout, cout, kernel_size=5,padding=2),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=helper(1))
        )

        self.two_d_branch3 = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=3,padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=helper(1)),
            nn.Conv2d(cout, cout, kernel_size=5,padding=2),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=helper(1))
        )

        self.two_d_branch4 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        if not helper(self.flag):
            branch1 = self.two_d_branch1(x)
            branch2 = self.two_d_branch2(x)
            branch3 = self.two_d_branch3(x)
            branch4 = self.two_d_branch4(x)
        else:
            branch1 = self.one_d_branch1(x)
            branch2 = self.one_d_branch2(x)
            branch3 = self.one_d_branch3(x)
            branch4 = self.one_d_branch4(x)

        return torch.cat([branch1, branch2, branch3, branch4], 1)


        
class Inception_Q3(nn.Module):
    def __init__(self):
        super(Inception_Q3, self).__init__()
        """
        Write your code here
        """
        self.one_d_labels = 35
        self.two_d_labels = 10

        self.one_d_block = nn.Sequential(InceptionBlock(1, 2, 1),
                                         InceptionBlock(7, 2, 1),
                                         InceptionBlock(13, 2, 1),
                                         InceptionBlock(19, 2, 1))

        self.two_d_block = nn.Sequential(InceptionBlock(3, 8, 0),
                                         InceptionBlock(27, 12, 0),
                                         InceptionBlock(63, 8, 0),
                                         InceptionBlock(87, 87, 0))

        self.FC_1d = nn.Linear(400000, self.one_d_labels)
        self.FC_2d = nn.Linear(356352, self.two_d_labels)

    def forward(self, x):
        if len(x.shape) == 3:
            if x.shape[1] == 1:
                x = self.one_d_block(x)
                x = x.view(x.size(0),-1)
                x = self.FC_1d(x)
            else:
                x = self.two_d_block(x)
                x = x.view(x.size(0),-1)
                x = self.FC_2d(x)
        else:
            x = self.two_d_block(x)
            x = x.view(x.size(0),-1)
            x = self.FC_2d(x)

        return x
        

'''
___________________________________________________________________________________________________________________________________________________________________
___________________________________________________________________________________________________________________________________________________________________
'''

class CustomNetwork_Q4(nn.Module):
    def __init__(self, cin=3):
        super(CustomNetwork_Q4, self).__init__()
        """
        Write your code here
        """
        self.one_d_labels = 35
        self.two_d_labels = 10
        
        self.img_resBlock1 = ResBlock(cin,3,0)
        self.img_resBlock2 = ResBlock(3,3,0)
        self.img_resBlock3 = ResBlock(33,33, 0)
        self.img_resBlock4 = ResBlock(78,78, 0)
        self.img_resBlock5 = ResBlock(138,138, 0)

        self.audio_resBlock1 = ResBlock(1,1,1)
        self.audio_resBlock2 = ResBlock(1,1,1)
        self.audio_resBlock3 = ResBlock(7,7, 1)
        self.audio_resBlock4 = ResBlock(10,10, 1)
        self.audio_resBlock5 = ResBlock(13,13, 1)
        
        self.img_inceptionBlock1 = InceptionBlock(3,5,0)
        self.img_inceptionBlock2 = InceptionBlock(18,5,0)
        self.img_inceptionBlock3 = InceptionBlock(33,15,0)
        self.img_inceptionBlock4 = InceptionBlock(78,20,0)
        self.img_inceptionBlock5 = InceptionBlock(138,32,0)

        self.audio_inceptionBlock1 = InceptionBlock(1,1,1)
        self.audio_inceptionBlock2 = InceptionBlock(4,1,1)
        self.audio_inceptionBlock3 = InceptionBlock(7,1,1)
        self.audio_inceptionBlock4 = InceptionBlock(10,1,1)
        self.audio_inceptionBlock5 = InceptionBlock(13,1,1)

        self.img_FC = nn.Linear(239616,self.two_d_labels)
        self.audio_FC = nn.Linear(256000,self.one_d_labels)

        self.model = nn.Sequential(nn.Linear(239616, self.two_d_labels))

    def forward(self, x):
        if len(x.shape) == 3:
            if x.shape[1] == 1:
                x = self.audio_resBlock1(x)
                x = self.audio_resBlock2(x)

                x = self.audio_inceptionBlock1(x)
                x = self.audio_inceptionBlock2(x)

                x = self.audio_resBlock3(x)
                x = self.audio_inceptionBlock3(x)

                x = self.audio_resBlock4(x)
                x = self.audio_inceptionBlock4(x)

                x = self.audio_resBlock5(x)
                x = self.audio_inceptionBlock5(x)
                x = x.view(x.size(0),-1)
                x = self.audio_FC(x)
            else:
                x = self.img_resBlock1(x)
                x = self.img_resBlock2(x)

                x = self.img_inceptionBlock1(x)
                x = self.img_inceptionBlock2(x)

                x = self.img_resBlock3(x)
                x = self.img_inceptionBlock3(x)

                x = self.img_resBlock4(x)
                x = self.img_inceptionBlock4(x)

                x = self.img_resBlock5(x)
                x = self.img_inceptionBlock5(x)
                x = x.view(x.size(0),-1)
                x = self.img_FC(x)
        else:
            x = self.img_resBlock1(x)
            x = self.img_resBlock2(x)

            x = self.img_inceptionBlock1(x)
            x = self.img_inceptionBlock2(x)

            x = self.img_resBlock3(x)
            x = self.img_inceptionBlock3(x)

            x = self.img_resBlock4(x)
            x = self.img_inceptionBlock4(x)

            x = self.img_resBlock5(x)
            x = self.img_inceptionBlock5(x)
            x = x.view(x.size(0),-1)
            x = self.img_FC(x)

        return x


'''
___________________________________________________________________________________________________________________________________________________________________
___________________________________________________________________________________________________________________________________________________________________
'''


def trainer(gpu="F",
            dataloader=None,
            network=None,
            criterion=None,
            optimizer=None):
    
    network = network.to(torch.device("cuda:0") if gpu == "T" else torch.device("cpu"))
    best_acc = 0
    
    # Assuming a global variable EPOCH
    for epoch in range(EPOCH):
        network.train()
        curr_train_loss = 0
        correct_train = 0
        tot_train = 0

        #For Training Set
        for x_train, y_train in dataloader:
            x_train, y_train = x_train.to(torch.device("cuda:0") if gpu == "T" else torch.device("cpu")), y_train.to(torch.device("cuda:0") if gpu == "T" else torch.device("cpu"))

            #Zeroing the gradient & doing prediction using our network
            optimizer.zero_grad()
            y_train_pred = network(x_train)
            train_error = criterion(y_train_pred, y_train) #Computing the error
            train_error.backward() #Back-propagation
            optimizer.step() #SGD in works (Updating weights & biases)

            #Computing the loss/error of the current training set, and classifying correctly trained samples amongst total trained
            curr_train_loss += train_error.item()
            pred = torch.max(y_train_pred.data, 1)[-1]
            tot_train += y_train.size(0)
            correct_train += (pred==y_train).sum().item()

        #Computing Training Loss & Training Accuracy for the current epoch
        loss1 = curr_train_loss/len(dataloader)
        training_accuracy = correct_train / tot_train
        
        # Print epoch loss and accuracy
        print("Training Epoch: {}, [Loss: {:.4f}, Accuracy: {:.2f}%]".format(
            epoch, loss1, training_accuracy * 100))

        # Save checkpoint if this epoch yielded the best accuracy so far
        if training_accuracy > best_acc:
            best_acc = training_accuracy
            torch.save(network.state_dict(), 'best_checkpoint.pth')

        if best_acc*100 >= 50:
            return 

def validator(gpu="F",
              dataloader=None,
              network=None,
              criterion=None,
              optimizer=None):
    
    # Load the saved network checkpoint
    network.load_state_dict(torch.load('best_checkpoint.pth'))
    network = network.to(torch.device("cuda:0") if gpu == "T" else torch.device("cpu"))

    network.eval()
    curr_val_loss = 0
    correct_val = 0
    tot_val = 0

    #Disabling the computation of gradients while in testing/validation mode
    with torch.no_grad():
        #Following similar ideology from training set into the validation one
        for x_val, y_val in dataloader:
            x_val, y_val = x_val.to(torch.device("cuda:0") if gpu == "T" else torch.device("cpu")), y_val.to(torch.device("cuda:0") if gpu == "T" else torch.device("cpu"))

            y_val_pred = network(x_val)
            val_error = criterion(y_val_pred, y_val)
            curr_val_loss += val_error.item()
            val_pred = torch.max(y_val_pred.data, 1)[-1]
            tot_val += y_val.size(0)
            correct_val += (val_pred == y_val).sum().item()

    #Computing Validation Loss & Validation Accuracy for the current epoch
    loss2 = curr_val_loss / len(dataloader)
    validation_accuracy = correct_val / tot_val

    # Print the validation loss and accuracy
    print("Validation: [Loss: {:.4f}, Accuracy: {:.2f}%]".format(loss2, validation_accuracy * 100))


def evaluator(gpu="F",
              dataloader=None,
              network=None,
              criterion=None,
              optimizer=None):

    if criterion==None:
        criterion = nn.CrossEntropyLoss()
    
    # Load the saved network checkpoint
    network.load_state_dict(torch.load('best_checkpoint.pth'))
    network = network.to(torch.device("cuda:0") if gpu == "T" else torch.device("cpu"))

    # Put the network in evaluation mode
    network.eval()

    curr_test_loss = 0
    correct_test = 0
    tot_test = 0

    with torch.no_grad():
        for x_test, y_test in dataloader:
            # x_test, y_test = x_test.to(torch.device("cuda:0") if gpu == "T" else torch.device("cpu")), y_test.to(torch.device("cuda:0") if gpu == "T" else torch.device("cpu"))

            y_test_pred = network(x_test)
            test_error = criterion(y_test_pred, y_test)
            curr_test_loss += test_error.item()
            test_pred = torch.max(y_test_pred.data, 1)[-1]
            tot_test += y_test.size(0)
            correct_test += (test_pred == y_test).sum().item()

    #Computing Testing Loss & Testing Accuracy for the current epoch
    loss3 = curr_test_loss / len(dataloader)
    testing_accuracy = correct_test / tot_test

    # Print the test loss and accuracy
    print("[Loss: {:.4f}, Accuracy: {:.2f}%]".format(loss3, testing_accuracy * 100))