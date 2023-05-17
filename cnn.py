import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# CNN model definition
class Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super(Classifier, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 12, 3, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 20, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(20, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.classifier = nn.Linear(32 * 112 * 112, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 32 * 112 * 112)
        x = self.classifier(x)
        return x

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomRotation((-20, 20)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_data_loader(path):
    dataset = datasets.ImageFolder(root=path, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
    return data_loader

trainloader = get_data_loader('./train')
testloader = get_data_loader('./test')
valloader = get_data_loader('./val')

# Initialize the model, loss function, and optimizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Classifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
num_epochs = 10
losses = []

for epoch in range(num_epochs):
    running_loss = 0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        losses.append(loss.item())
        
    print(f"Epoch {epoch+1} - Training loss: {running_loss/len(trainloader)}")

# Testing
correct_count, all_count = 0, 0
for images, labels in testloader:
    images, labels = images.to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct_count += (predicted == labels).sum().item()
        all_count += labels.size(0)

print(f"Number Of Images Tested = {all_count}")
print(f"Model Accuracy = {correct_count/all_count}")
