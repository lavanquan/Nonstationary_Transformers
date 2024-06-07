import copy
import torch 
from tqdm import tqdm
from abc import ABC, abstractmethod

class UserBase(ABC):
    def __init__(self, train_loader, model, user_id, local_epochs):
        # super().__init__()
        self.model = copy.deepcopy(model)
        self.train_loader = train_loader
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.id = user_id
        self.local_epochs = local_epochs
    
    @abstractmethod
    def user_train(self):
        # # Get model
        # model = self.model

        # # Use all available gpus for training
        # num_gpus = torch.cuda.device_count()
        # model.cuda()
        # model = torch.nn.DataParallel(model, device_ids=list(range(num_gpus)), dim=0)

        # # Define the loss function and optimizer
        # criterion = torch.nn.MSELoss()
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # # Train model
        # for epoch in range(self.local_epochs):
        #     model.train()
        #     for inputs, targets in self.train_loader:
        #         optimizer.zero_grad()
        #         targets = targets.reshape(-1, 1)
        #         inputs, targets = inputs.cuda(), targets.cuda()
        #         outputs = model(inputs)
        #         loss = criterion(outputs, targets)
        #         loss.backward()
        #         optimizer.step()
        # loss = loss.item()
        # # print(f'User {self.id} Train Loss: {loss:.3f}')
        # self.model = model
        # return loss
        pass
