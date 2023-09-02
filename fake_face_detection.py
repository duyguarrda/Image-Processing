
try:   
    import torch
    from torch.utils.data import Dataset, DataLoader
    import torch.nn as nn
    from torchvision import transforms
    import torch.optim as optim
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from sklearn.metrics import confusion_matrix
    import seaborn as sn
    import pandas as pd
    import os
    import cv2
except ImportError:
    print("Pelase Check the Library!")

real_path = "C:/Users/Duygu/Desktop/Duygu/realandfakeface/real_and_fake_face/training_real"
fake_path = "C:/Users/Duygu/Desktop/Duygu/realandfakeface/real_and_fake_face/training_fake"
test_path = "C:/Users/Duygu/Desktop/Duygu/Test_Data"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Classları tanımladık
classes = ["real","fake"]
# Görüntü verilerini ve etiketleri depolamak için listeler oluşturduk
data = []
labels = []

# Görüntülere uygulanacak dönüşümü tanımladık  
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])   # Normalize
])

# Gerçek görüntüleri yükledik,veri ve etiket listelerine ekledik
for img_file in os.listdir(real_path):
    img = Image.open(os.path.join(real_path, img_file))
    data.append(img)
    labels.append(1)

# Sahte resimler yükledik,veri ve etiket listelerine ekledik
for img_file in os.listdir(fake_path):
    img = Image.open(os.path.join(fake_path, img_file))
    data.append(img)
    labels.append(0)

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]

        if self.transform:
            x = self.transform(x)

        return x, torch.tensor(y)

# (%80 train, %20 test)
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.1, random_state=42)  
# Train_data'yı train ve doğrulama kümelerine daha fazla ayırın
train_data, valid_data, train_labels, valid_labels = train_test_split(train_data, train_labels, test_size=0.1, random_state=42) 
# train ve test veri kümelerinizi tanımladık
train_dataset = CustomDataset(train_data, train_labels, transform=transform)
test_dataset = CustomDataset(test_data, test_labels, transform=transform)
valid_dataset = CustomDataset(valid_data,valid_labels, transform=transform)

valid_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
train_dataLoader = DataLoader(train_dataset , batch_size= 64, shuffle=True)
test_dataLoader = DataLoader(test_dataset, batch_size=64, shuffle=False )

# sinir ağı
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.CNN_layers = nn.Sequential(
            # Convolutional Layer 1
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            # nn.Dropout(0.5),
            # Convolutional Layer 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            # Convolutional Layer 3
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            # Convolutional Layer 4
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.Linear_layers = nn.Sequential(
            # Linear Layer 1
            nn.Linear(in_features=32*6*6,out_features=128),
            nn.ReLU(),
            # Linear Layer 2
            nn.Linear(in_features=128, out_features=2)
        )
    
    def forward(self, x):
        x = self.CNN_layers(x)
        x = x.view(x.size(0),-1)
        x = self.Linear_layers(x)
        x = nn.functional.log_softmax(x,dim=1)
        return x

# Modeli oluşturup ve cihaza gönderdik
model = Net().to(device)

# Optimize Edici'yi öğrenme oranıyla tanımladık
optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay=0.0001)
#giriş logitleri ve hedef arasındaki çapraz entropi kaybını hesaplandı
criterion = nn.CrossEntropyLoss()
training_loss = []
training_accuracy = []
validation_loss = []
validation_accuracy = []
predict = []
target = []

# Eğitim
def train(model,epochs,optimizer,criterion):
    for epoch in range(epochs):
        # eğitim için modelin kullanımı
        model.train()
        train_loss = 0
        train_correct = 0
        n_samples = 0

        for index, (image,label) in enumerate(train_dataLoader):
            images = image.to(device)
            labels = label.to(device)

            
            optimizer.zero_grad()

           
            output = model(images)

            
            loss = criterion(output,labels)

         
            loss.backward()

          
            optimizer.step()

            
            train_loss += loss.item()

           
            _,predicted = torch.max(output.data,1)

           
            n_samples += labels.size(0)

            
            train_correct += (predicted == labels).sum().item()

            
            train_acc = train_correct/n_samples

        model.eval()
        valid_loss = 0
        n_samples = 0
        valid_correct = 0

        for image,label in valid_loader:
            images = image.to(device)
            labels = label.to(device)

            
            output = model(images)

            
            loss = criterion(output,labels)

            
            valid_loss += loss.item()

            _, predicted = torch.max(output.data,1)

           
            n_samples += labels.size(0)

            
            valid_correct += (predicted == labels).sum().item()

            
            valid_accuracy = valid_correct/n_samples

        
        validation_accuracy.append(valid_accuracy)
        validation_loss.append(valid_loss/len(valid_loader))
        training_accuracy.append(train_acc)
        training_loss.append(train_loss/len(train_dataLoader))

        print("Epoch: {}/{}  ".format(epoch+1, epochs),  
                "Training loss: {:.4f}  ".format(train_loss/len(train_dataLoader)),
                "Training Accuracy : {:.4f}".format(train_acc),
                "Validation loss: {:.4f}".format(valid_loss/len(valid_loader)),
                "Validation Accuracy : {:.4f}".format(valid_accuracy))
    
    print("Training Finished")
    return validation_accuracy,validation_loss,training_accuracy,training_loss

def torch_no_grad(model,criterion):
    model.eval()
    test_loss = 0
    n_samples = 0
    test_correct = 0
    with torch.no_grad():
        for image,label in test_dataLoader:
            images = image.to(device)
            labels = label.to(device)
            
            output = model(images)
            
            loss = criterion(output,labels)
            
            test_loss += loss.item()

            _,predicted = torch.max(output.data,1)
            
            n_samples += labels.size(0)
            
            test_correct += (predicted == labels).sum().item()
            
            test_accuracy = test_correct/n_samples
            labels = labels.data.cpu().numpy()
            predict.extend(predicted)
            target.extend(labels)
        
        print("Test Accuracy: {:.4f}".format(test_accuracy),
              ("Test Loss: {:.4f}".format(test_loss/len(test_dataLoader))))
        return predict,target
        
def plot(train_acc, train_loss,valid_acc,valid_loss):
   
    training_loss = np.array(train_loss)
    validation_loss = np.array(valid_loss)
    training_accuracy = np.array(train_acc)
    validation_accuracy = np.array(valid_acc)

    
    fig,axes = plt.subplots(nrows=1, ncols=2, figsize = (10,5))

    # Axes[0]
    axes[0].plot(validation_loss, label = "Validation Loss")
    axes[0].plot(training_loss, label = "Training Loss")
    axes[0].set_xlabel("Iterations")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Axes[1]
    axes[1].plot(validation_accuracy,label = "Validation Accuracy")
    axes[1].plot(training_accuracy, label = "Training Accuracy")
    axes[1].set_xlabel("Iterations")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

def save_model():
    path = "./model.pt"
    torch.save(model.state_dict(), path)

def confusion_matrixx(y_test, y_predict):
    cf_matrix = confusion_matrix(y_test, y_predict)

    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = classes,
                     columns = classes)
    
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.show()
    plt.savefig('C:/Users/Duygu/Desktop/Duygu/Test_Data/output.png')

def test_model(img_path,model):
    
    state = torch.load('C:/Users/Duygu/Desktop/Duygu/model.pt')
    model.load_state_dict(state)
    
    model.eval()
    for img_file in os.listdir(img_path):
        img = Image.open(os.path.join(img_path, img_file))

        
        img_ = transform(img)

        
        img_ = img_.unsqueeze(0)

        
        with torch.no_grad():
            outputs = model(img_)
            _, predicted = torch.max(outputs.data, 1)
            predicted_label = classes[predicted]
            
            
            plt.imshow(img)
            plt.title(f'Prediction: {predicted_label}')
            plt.axis('off')
            plt.show()

if __name__ == "__main__":
    # train(model,epochs = 15,optimizer = optimizer,criterion = criterion)
    # torch_no_grad(model, criterion)      
    # plot(training_accuracy,training_loss,validation_accuracy,validation_loss)
    # confusion_matrixx(predict,target)
    # save_model()
    test_model(test_path,model)