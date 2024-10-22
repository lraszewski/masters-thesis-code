import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from helpers import get_batch_size, get_roberta
from training import train_model

DEVICE = 'cuda'
SAVE_PATH = 'reptile/'

# trains a model using reptile, checkpoints and logs epoch loss
def reptile(task_distribution, model, criterion, epochs, interp, inner_lr, inner_steps):

    # setup
    roberta = get_roberta()

    for epoch in range(epochs):

        train_loss = outer_loop(task_distribution, roberta, model, criterion, interp, inner_lr, inner_steps)
        print(f"---\ntrain loss: {train_loss}")

        # save a checkpoint of the model
        path = SAVE_PATH + 'model.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'train_loss': train_loss,
        }, path)

# outer loop for the reptile algorithm
def outer_loop(task_distribution, roberta, model, criterion, interp, inner_lr, inner_steps):

    # reptile only ever deals with one task
    tasks_dataloader = DataLoader(task_distribution, batch_size=1, shuffle=True, collate_fn=lambda x: x)

    epoch_loss = 0.0
    for batch in tqdm(tasks_dataloader):
        for task in batch:

            # train a clone
            clone = model.clone()
            params, loss = inner_loop(roberta, task, clone, criterion, inner_lr, inner_steps)
            epoch_loss += loss

            # update meta parameters
            for old, new in zip(model.parameters(), params):
                old.data += interp * (new.data - old.data)
    
    return epoch_loss

# inner loop fo the reptile algorithm, returns learned parameters        
def inner_loop(roberta, task, model, criterion, lr, steps):
    
    # unpack task
    _, support_set, _, query_set = task

    # reptile makes no distinction between support and query set
    dataset = ConcatDataset([support_set, query_set])

    # create necessary dataloaders
    batch_size = get_batch_size(len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # train the clone x update steps
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss = train_model(roberta, model, optimiser, dataloader, criterion, max_steps=steps)

    # return the model parameters and the average train loss
    return model.parameters(), loss.item()