import torch
from torch import nn
from torch.utils.data import DataLoader

from NextWordModel import NextWordModel
from WordDataset import WordDataset


def train(model, dataloader, criterion, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs,labels in dataloader:
            #inputs = torch.cat([torch.tensor([[0.0]] * inputs).to(device), inputs], dim=1)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader):.4f}')


if __name__ == '__main__':
    # 检查GPU是否可用 若可用则使用GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define hyperparameters
    input_size = 1
    hidden_size = 128
    output_size = 26
    learning_rate = 0.001
    batch_size = 64
    epochs = 10

    # Load data
    index_path = 'data/words_index.txt'
    data_path = 'data/words_ch.txt'
    dataset = WordDataset(index_path,data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define model, loss function, optimizer
    model = NextWordModel(input_size, hidden_size, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train(model, dataloader, criterion, optimizer, device, epochs)
