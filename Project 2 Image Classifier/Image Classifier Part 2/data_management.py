import torch
from torchvision import datasets, transforms, models
from PIL import Image


def load_data(path):
    train_dir = path + '/train'
    valid_dir = path + '/valid'
    test_dir = path + '/test'
   
    train_transform = transforms.Compose([transforms.RandomRotation(50),
                                                  transforms.RandomResizedCrop(224),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406],
                                                                       [0.229, 0.224, 0.225])])

    valid_transform = transforms.Compose([transforms.Resize(255),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([transforms.Resize(255),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                                      [0.229, 0.224, 0.225])])

    # Loading the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform = train_transform)
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transform)
    test_data = datasets.ImageFolder(test_dir, transform = test_transform)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size = 64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 64)
    
    print("Finished loading and preprocessing data.")
    
    return train_data, trainloader, validloader, testloader

def process_image(image):
    ''' appyling transform in the input images 
    '''
    image = Image.open(image)
    
    image_transform = transforms.Compose([transforms.Resize(255),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                                      [0.229, 0.224, 0.225])])
    
    return image_transform(image)