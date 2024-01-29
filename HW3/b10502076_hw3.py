"""# Import Packages"""

_exp_name = "sample"

# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset, SubsetRandomSampler
from torchvision.datasets import DatasetFolder, VisionDataset
# This is for the progress bar.
from tqdm.auto import tqdm
import gc
import random
import torchvision.models as models
from sklearn.model_selection import KFold
from resnest.torch import resnest50 #需要 pip install git+https://github.com/zhanghang1989/ResNeSt

myseed = 7777  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

"""# Transforms
Torchvision provides lots of useful utilities for image preprocessing, data *wrapping* as well as data augmentation.

Please refer to PyTorch official website for details about different transforms.
"""

# Normally, We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.ToTensor(),
])

# However, it is also possible to use augmentation in the testing phase.
# You may use train_tfm to produce a variety of images and then test using ensemble methods

# transforms 參考來源 : https://ithelp.ithome.com.tw/articles/10276641

train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((240, 240)),
    # You may add some transforms here.
    transforms.RandomRotation(40),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=0.2),
    transforms.RandomResizedCrop(240),
    transforms.RandomHorizontalFlip(p=0.5),
    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
])


"""# Datasets
The data is labelled by the name, so we load images and label while calling '__getitem__'
"""

class FoodDataset(Dataset):

    def __init__(self,data,tfm,files = None):
        super(FoodDataset).__init__()
        self.files = data
        if files != None:
            self.files = files
            
        self.transform = tfm
  
    def __len__(self):
        return len(self.files)
  
    def __getitem__(self,idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)
        
        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1 # test has no label
            
        return im,label

"""# Model"""

"""# Configurations"""

# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize a model, and put it on the device specified.
# model = models.resnet50(weights = None, num_classes=11).to(device)

# The number of batch size.
batch_size = 32

# The number of training epochs.
n_epochs = 250

# If no improvement in 'patience' epochs, early stop.
# patience = 20

# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss() 

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-5)

"""# Dataloader"""

# Construct train and valid datasets.
# The argument "loader" tells how torchvision reads the data.
train = sorted([os.path.join("./train",x) for x in os.listdir("./train") if x.endswith(".jpg")])
valid = sorted([os.path.join("./valid",x) for x in os.listdir("./valid") if x.endswith(".jpg")])
data = train + valid

# cross validation 寫法參考 : https://medium.com/dataseries/k-fold-cross-validation-with-pytorch-and-sklearn-d094aa00105f

k=4
splits=KFold(n_splits=k,shuffle=True,random_state=76)

"""# Start Training"""

for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(data)))):
    
    print('Fold {}'.format(fold + 1))

    train_data = []
    valid_data = []

    for i in train_idx:
        train_data.append(data[i])
    for j in val_idx:  
        valid_data.append(data[j])

    train_set = FoodDataset(train_data, tfm=train_tfm)
    valid_set = FoodDataset(valid_data, tfm=test_tfm)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Initialize trackers, these are not parameters and should not be changed
    # stale = 0
    best_acc = 0
    model = resnest50(pretrained = False, num_classes=11).to(device)
    # model = models.wide_resnet101_2(weights=None, num_classes=11).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2, eta_min=0)

    for epoch in range(n_epochs):

        # ---------- Training ----------
        # Make sure the model is in train mode before training.
        model.train()

        # These are used to record information in training.
        train_loss = []
        train_accs = []

        for batch in tqdm(train_loader):

            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            #imgs = imgs.half()
            #print(imgs.shape,labels.shape)

            # Forward the data. (Make sure data and model are on the same device.)
            logits = model(imgs.to(device))

            # Calculate the cross-entropy loss.
            # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            loss = criterion(logits, labels.to(device))

            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()

            # Compute the gradients for parameters.
            loss.backward()

            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            # Update the parameters with computed gradients.
            optimizer.step()
            scheduler.step()

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_accs.append(acc)
            
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()

        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []

        # Iterate the validation set by batches.
        for batch in tqdm(valid_loader):

            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            #imgs = imgs.half()

            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(imgs.to(device))

            # We can still compute the loss (but not the gradient).
            loss = criterion(logits, labels.to(device))

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            valid_loss.append(loss.item())
            valid_accs.append(acc)
            #break

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        # Print the information.
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")


        # update logs
        if valid_acc > best_acc:
            with open(f"./{_exp_name + str(fold + 1)}_log.txt","a"):
                print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
        else:
            with open(f"./{_exp_name + str(fold + 1)}_log.txt","a"):
                print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")


        # save models
        if valid_acc > best_acc:
            print(f"Best model found at epoch {epoch + 1}, saving model")
            torch.save(model.state_dict(), f"{_exp_name + str(fold + 1)}_best.ckpt") # only save best to prevent output memory exceed error
            best_acc = valid_acc
            # stale = 0
        # else:
        #     stale += 1
        #     if stale > patience:
        #         print(f"No improvment {patience} consecutive epochs, early stopping")
        #         break
    del train_set
    del valid_set
    del train_loader
    del valid_loader
    gc.collect()


"""# Dataloader for test"""

# Test Time Augmentation 寫法參考 Cool 作業討論區

# Construct test datasets.
# The argument "loader" tells how torchvision reads the data.
test = sorted([os.path.join("./test",x) for x in os.listdir("./test") if x.endswith(".jpg")])
test_set1 = FoodDataset(test, tfm=train_tfm)
test_set2 = FoodDataset(test, tfm=test_tfm)
test_loader1 = DataLoader(test_set1, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
test_loader2 = DataLoader(test_set2, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

"""# Testing and generate prediction CSV"""

for number in range(k):
    model = resnest50(pretrained = False, num_classes=11).to(device)
    # model = models.wide_resnet101_2(weights=None, num_classes=11).to(device)
    model.load_state_dict(torch.load(f"{_exp_name + str(number+1)}_best.ckpt"))
    model.eval()

    pred = []
    test_list = []
    aug_test_list = []
    test_pred_ensemble = []

    with torch.no_grad():
        for data,_ in (tqdm(test_loader2)):
            test1 = model(data.to(device))
            test_list.append(test1)
        for data,_ in (tqdm(test_loader1)):
            test2 = model(data.to(device))
            aug_test_list.append(test2)
        for i in range(len(test_list)):
            test_pred_ensemble.append(0.8 * test_list[i] + 0.2 * aug_test_list[i])
        for i in range(len(test_pred_ensemble)):
            test_label = np.argmax(test_pred_ensemble[i].cpu().data.numpy(), axis=1)
            pred += test_label.squeeze().tolist()

    def pad4(i):
        return "0"*(4-len(str(i)))+str(i)
    df = pd.DataFrame()
    df["Id"] = [pad4(i) for i in range(len(test_set2))]
    df["Category"] = pred
    df.to_csv(str(number+1) + "submission.csv",index = False)

# voting for different csv (ensemble)
import csv
import pandas as pd
import numpy as np

csv_list = []
voting = []
prediction = []
length = k

for i in range(11):
    voting.append(0)

for i in range(length):
    file = open(str(i+1) + 'submission.csv')
    reader = csv.reader(file)
    data_list = list(reader) 
    csv_list.append(data_list)


for i in range(3000):
    for j in range(length):
        if (j == 0):
            voting[int(csv_list[j][i+1][1])] += 2
        else:
            voting[int(csv_list[j][i+1][1])] += 1
    maxid = 0
    for k in range(11):
        if voting[k] > voting[maxid]:
            maxid = k
    prediction.append(maxid)
    for l in range(11):
        voting[l] = 0

def pad4(i):
    return "0"*(4-len(str(i)))+str(i)
df = pd.DataFrame()
df["Id"] = [pad4(i) for i in range(len(prediction))]
df["Category"] = prediction
df.to_csv("submission.csv",index = False)

file.close()

"""# Q1. Augmentation Implementation
## Implement augmentation by finishing train_tfm in the code with image size of your choice. 
## Directly copy the following block and paste it on GradeScope after you finish the code
### Your train_tfm must be capable of producing 5+ different results when given an identical image multiple times.
### Your  train_tfm in the report can be different from train_tfm in your training code.

"""

train_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    # Resize the image into a fixed shape (height = width = 128)

    # You can add some transforms here.
    transforms.RandomRotation(40),
    # Rotate the image by 40 degree.

    transforms.RandomAffine(degrees = 0, translate=(0.2, 0.2), shear=0.2), 
    # Random affine transformation of the image keeping center invariant. 
    # degrees : Range of degrees to select from.
    # translate：tuple of maximum absolute fraction for horizontal and vertical translations. 
    # shear : Range of degrees to select from.

    transforms.RandomResizedCrop(128),
    # Crop a random portion of image and resize it to a given size (224).

    transforms.RandomHorizontalFlip(p=0.5), 
    # Horizontally flip the given image randomly with a given probability (0.5).

    transforms.Grayscale(0.2),
    # Randomly convert image to grayscale with a probability of 0.2 .
    
    transforms.ToTensor(),
])

"""# Q2. Visual Representations Implementation
## Visualize the learned visual representations of the CNN model on the validation set by implementing t-SNE (t-distributed Stochastic Neighbor Embedding) on the output of both top & mid layers (You need to submit 2 images). 

"""

import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.cm as cm
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the trained model
model = resnest50(pretrained = False, num_classes=11).to(device)
state_dict = torch.load(f"{_exp_name}1_best.ckpt")
model.load_state_dict(state_dict)
model.eval()

layers = dict(model.named_children())

# Load the vaildation set defined by TA
valid = sorted([os.path.join("./valid",x) for x in os.listdir("./valid") if x.endswith(".jpg")])
valid_set = FoodDataset(valid, tfm=test_tfm)
valid_loader = DataLoader(valid_set, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)


# Extract the representations for the specific layer of model
# index = ... # You should find out the index of layer which is defined as "top" or 'mid' layer of your model.
features = []
labels = []
for batch in tqdm(valid_loader):
    imgs, lbls = batch
    with torch.no_grad():
        imgs = imgs.to(device)
        for name, layer in layers.items():
            imgs = layer(imgs)
            if name == "layer4":  # 獲取指定的中間層 layer2 (middle) layer4 (top)
                break
        logits = imgs.view(imgs.size()[0], -1)
    labels.extend(lbls.cpu().numpy())
    logits = np.squeeze(logits.cpu().numpy())
    features.extend(logits)
    
features = np.array(features)
colors_per_class = cm.rainbow(np.linspace(0, 1, 11))

# Apply t-SNE to the features
features_tsne = TSNE(n_components=2, init='pca', random_state=42).fit_transform(features)

# Plot the t-SNE visualization
plt.figure(figsize=(10, 8))
for label in np.unique(labels):
    plt.scatter(features_tsne[labels == label, 0], features_tsne[labels == label, 1], label=label, s=5)
plt.legend()
plt.show()