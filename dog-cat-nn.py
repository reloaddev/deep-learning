if __name__ == "__main__":

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.utils.data as data
    import torchvision
    from torchvision import transforms, datasets

    ### Uncomment for Google Colab ###
    # from google.colab import drive
    # drive.mount('/content/drive', force_remount=True)

    # TRAIN_DATA_PATH = "/content/drive/MyDrive/dl-data/project1/data/train"
    # TEST_DATA_PATH = "/content/drive/MyDrive/dl-data/project1/data/test"
    # VALIDATION_DATA_PATH = "/content/drive/MyDrive/dl-data/project1/data/validation"
    ###

    ### Comment for Google Colab ###
    TRAIN_DATA_PATH = "data/train"
    TEST_DATA_PATH = "data/test"
    VALIDATION_DATA_PATH = "data/validation"
    ###

    BATCH_SIZE=50
    IMAGE_SIZE=(100,100)

    ### Create the data loaders ###
    ###############################

    # Define data transformation
    flip_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(1),
        transforms.ToTensor(),
    ])
    trivial_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.Grayscale(num_output_channels=1),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
    ])
    original_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    random_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomInvert(0.5),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    # Load the datasets
    train_dataset_flip = datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=flip_transform)
    train_dataset_triv = datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=trivial_transform)
    train_dataset_org = datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=original_transform)
    train_dataset_random1 = datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=random_transform)
    train_dataset_random2 = datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=random_transform)
    train_data = torch.utils.data.ConcatDataset(
        [train_dataset_flip, train_dataset_triv, train_dataset_org, train_dataset_random1, train_dataset_random2]
    )
    train_data_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=test_transform)
    test_data_loader  = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


    ### Constructing the network ###
    ################################
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = 0.001
    epochs = 10

    class CNNNetwork(nn.Module):
        def __init__(self):
            super(CNNNetwork, self).__init__()
            self.conv_stack = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.conv_stack2 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(128 * 3 * 3, 256),
                nn.ReLU(),
                nn.Linear(256, 10)
            )

        def forward(self, x):
            x = self.conv_stack(x)
            x = self.conv_stack2(x)
            x = self.conv_stack2(x)
            x = self.conv_stack2(x)
            x = x.view(x.size(0), -1)  # Flatten the tensor
            x = self.linear_relu_stack(x)
            return x

    model = CNNNetwork().to(device)


    ### Train and test the network ###
    ##################################

    def train_loop(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 50 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


    def test_loop(dataloader, model, loss_fn):
        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_data_loader, model, loss_fn, optimizer)
        test_loop(test_data_loader, model, loss_fn)
    print("Done!")