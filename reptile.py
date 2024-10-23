import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from helpers import get_batch_size, get_roberta
from training import train_model

DEVICE = 'cuda'

# trains a model using reptile, checkpoints and logs epoch loss
def reptile(save_path, task_distribution, model, criterion, epochs, interp, inner_lr, inner_steps):

    # setup
    roberta = get_roberta()
    epoch_losses = []

    iterator = tqdm(range(epochs), desc="reptile")
    for epoch in iterator:

        epoch_loss = outer_loop(task_distribution, roberta, model, criterion, interp, inner_lr, inner_steps)
        epoch_losses.append(epoch_loss)

        # save a checkpoint of the model
        path = save_path + '/model.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'epoch_loss': epoch_loss,
        }, path)

        # update progress bar
        iterator.set_postfix({
            'First Epoch Loss': epoch_losses[0],
            'Last Epoch Loss': epoch_losses[-1],
            'Min Epoch Loss': min(epoch_losses)
        })

# outer loop for the reptile algorithm
def outer_loop(task_distribution, roberta, model, criterion, interp, inner_lr, inner_steps):

    # reptile only ever deals with one task
    tasks_dataloader = DataLoader(task_distribution, batch_size=1, shuffle=True, collate_fn=lambda x: x)

    epoch_loss = 0.0
    iterator = tqdm(tasks_dataloader, leave=False)
    for batch in iterator:
        for task in batch:

            # train a clone
            clone = model.clone()
            params, loss, fn = inner_loop(roberta, task, clone, criterion, inner_lr, inner_steps)
            epoch_loss += loss

            # update meta parameters
            for old, new in zip(model.parameters(), params):
                old.data += interp * (new.data - old.data)

            # update progress bar
            iterator.set_postfix({
                'Task': fn,
                'Loss': loss
            })
    
    return epoch_loss

# inner loop for the reptile algorithm, returns learned parameters        
def inner_loop(roberta, task, model, criterion, lr, steps):
    
    # unpack task
    fn, _, support_set, _, query_set = task

    # reptile makes no distinction between support and query set
    dataset = ConcatDataset([support_set, query_set])

    # create necessary dataloaders
    batch_size = get_batch_size(len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # train the clone x update steps
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss = train_model(roberta, model, optimiser, dataloader, criterion, max_steps=steps)

    # return the model parameters and the average train loss
    return model.parameters(), loss.item(), fn