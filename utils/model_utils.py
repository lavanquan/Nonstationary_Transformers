import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import wandb

# Define MLP 
class MLPModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_hidden_layers, output_dim, use_instance_norm=False, dropout_rate=0):
        super().__init__()
        # Define the first layer
        model = [nn.Linear(input_dim, hidden_dim)]
        if use_instance_norm:
            model += [nn.InstanceNorm1d(hidden_dim)]
        model += [nn.ReLU(), nn.Dropout(dropout_rate)]

        # implement n_hidden_layers
        for _ in range(n_hidden_layers):
            model += [nn.Linear(hidden_dim, hidden_dim)]
            if use_instance_norm:
                model += [nn.InstanceNorm1d(hidden_dim)]
            model += [nn.ReLU(), nn.Dropout(dropout_rate)]
        
        # Implement output:
        model += [nn.Linear(hidden_dim, output_dim)]
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)

# Define LSTM
class LstmModule(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first, out_features):
        super(LstmModule, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.fc1 = torch.nn.Linear(in_features=hidden_size, out_features=out_features)

    def forward(self,x):
        x, _status = self.lstm(x)
        x = x[:,-1,:]
        output = self.fc1(torch.relu(x))
        # output = self.fc1(x)
        return output

def model_train(model, criterion, optimizer, num_epochs, train_loader, wandb=False, l2_penalty=0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.train()
    print(f'Model is train on {device}')
    print('---------------------------')
    for epoch in tqdm(range(num_epochs), desc='training'):
        for i, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            targets = targets.reshape(-1, 1)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            for param in model.parameters():
                l2_reg = torch.norm(param, p=2)
            loss += l2_penalty*l2_reg
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    print('---------------------------')
    return model

def model_train_mul_gpus(model, criterion, optimizer, num_epochs, train_loader, wandb=False, l2_penalty=0):
    num_gpus = torch.cuda.device_count()
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=list(range(num_gpus)), dim=0)
    model.train()
    print(f'Model is train on list of devices: {list(range(num_gpus))}')
    print('---------------------------')
    for epoch in tqdm(range(num_epochs), desc='training'):
        for i, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            targets = targets.reshape(-1, 1)
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            for param in model.parameters():
                l2_reg = torch.norm(param, p=2)
            loss += l2_penalty*l2_reg
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    print('---------------------------')
    return model

def model_val(model, test_loader, criterion):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            targets = targets.reshape(-1, 1)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, targets).item()

    avg_test_loss = test_loss / len(test_loader)
    print(f'Average Test Loss: {avg_test_loss:.4f}')
    return avg_test_loss




def train_wandb(model, train_loader, test_loader, config):
    num_gpus = config.num_gpus
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=list(range(num_gpus)), dim=0)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    print(f'Model is trained on {num_gpus} gpus')
    print('---------------------------')
    for epoch in tqdm(range(config.num_epochs), desc='training'):
        # Train model
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            targets = targets.reshape(-1, 1)
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if config.l2_penalty !=0:
                for param in model.parameters():
                    l2_reg = torch.norm(param, p=2)
                loss += config.l2_penalty * l2_reg
            loss.backward()
            optimizer.step()

        # Eval model
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                targets = targets.reshape(-1, 1)
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                test_loss += criterion(outputs, targets).item()
        avg_test_loss = test_loss / len(test_loader)

        wandb.log({"train_loss": loss, "val_loss": avg_test_loss})

    return model

def prediction(model, X_test, num_ts=37, window_size=24):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  
    model = model.to(device)

    x_data_test = X_test.reshape(-1, num_ts, window_size - 1)
    # y_data_test = Y_test.reshape(-1, num_ts, 1)
    # x_data_test.shape, y_data_test.shape
    x_data_test = torch.tensor(x_data_test, dtype=torch.float32)
    # y_data_test = torch.tensor(y_data_test, dtype=torch.float32)  

    model.eval()
    x_data_test = x_data_test.to(device)
    # y_data_test = y_data_test.to(device)


    for i in range(x_data_test.shape[0]):
        outputs = model(x_data_test[i])
        if i == 0:
            predict = outputs
            continue
        predict = torch.cat((predict, outputs), dim=1)

    predict = predict.cpu().detach().numpy()
    return predict

def lstm_prediction(model, X_test, num_ts=37, window_size=24):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  
    model = model.to(device)

    x_data_test = X_test.reshape(-1, num_ts, window_size - 1, 1)
    # y_data_test = Y_test.reshape(-1, num_ts, 1)
    # x_data_test.shape, y_data_test.shape
    x_data_test = torch.tensor(x_data_test, dtype=torch.float32)
    # y_data_test = torch.tensor(y_data_test, dtype=torch.float32)  

    model.eval()
    x_data_test = x_data_test.to(device)
    # y_data_test = y_data_test.to(device)


    for i in range(x_data_test.shape[0]):
        outputs = model(x_data_test[i])
        if i == 0:
            predict = outputs
            continue
        predict = torch.cat((predict, outputs), dim=1)

    predict = predict.cpu().detach().numpy()
    return predict

def save_model(model_name, model, config, optimizer_name, criterion_name, val_loss, file_path):
    torch.save({
        'dataset': config.dataset,
        'preprocess_type': config.preprocess_type,
        'num_user': config.num_user,
        'model_name': model_name,
        'model': model.state_dict(),
        'batch_size': config.batch_size,
        'global_epochs': config.global_epochs,
        'local_epochs': config.local_epochs,
        'learning_rate': config.learning_rate,
        'user_ratio': config.user_ratio,
        'detrending_data': config.detrending_data,
        'total_time_series': config.total_time_series,
        'optimizer_name': optimizer_name,
        'criterion_name': criterion_name,
        'val_loss': val_loss,
    }, file_path)
    print(f"Model is saved to {file_path}")
