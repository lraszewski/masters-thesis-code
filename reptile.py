import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, ConcatDataset

from tqdm import tqdm

from helpers import get_roberta, get_dataloader, get_latest_model
from training import train_model

DEVICE = 'cuda'

# trains a model using reptile, checkpoints and logs epoch loss
def reptile(save_path, task_distribution, model, criterion, epochs, outer_lr, inner_lr, inner_steps, serial=True):

    # setup
    roberta = get_roberta()
    epoch_losses = []

    # check for continued training
    latest = get_latest_model(save_path)
    epoch = 0
    if latest:
        save = torch.load(latest)
        model.load_state_dict(save['model_state_dict'])
        epoch = save['epoch'] + 1
        print(f"loaded model checkpoint at epoch {epoch-1}, beginning epoch {epoch}")

    iterator = tqdm(range(epoch, epochs), desc="reptile")
    for epoch in iterator:

        if serial:
            epoch_loss = outer_loop_serial(task_distribution, roberta, model, criterion, outer_lr, inner_lr, inner_steps, epoch, epochs)
        else:
            meta_optimiser = torch.optim.SGD(model.parameters(), lr=outer_lr)
            epoch_loss = outer_loop_sgd(task_distribution, roberta, model, meta_optimiser, criterion, inner_lr, inner_steps)
        epoch_losses.append(epoch_loss)

        # save a checkpoint of the model
        path = f'{save_path}/model_{epoch}.pt'
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
def outer_loop_serial(task_distribution, roberta, model, criterion, interp, inner_lr, inner_steps, epoch, epochs):

    # reptile only ever deals with one task
    tasks_dataloader = DataLoader(task_distribution, batch_size=1, shuffle=True, collate_fn=lambda x: x)
    
    # linear lr annealing
    meta_iters = len(task_distribution) * epochs
    curr_iter = len(task_distribution) * epoch
    frac_done = curr_iter / meta_iters
    curr_interp = frac_done * 0.0 + (1 - frac_done) * interp

    epoch_loss = 0.0
    iterator = tqdm(tasks_dataloader, leave=False)
    for batch in iterator:
        for task in batch:
            
            # linear annealing
            frac_done = curr_iter / meta_iters
            curr_interp = frac_done * 0.0 + (1 - frac_done) * interp
            curr_iter += 1

            # train a clone
            clone = model.clone()
            params, loss, fn = inner_loop(roberta, task, clone, criterion, inner_lr, inner_steps)
            epoch_loss += loss

            # update meta parameters
            for old, new in zip(model.parameters(), params):
                old.data += curr_interp * (new.data - old.data)

            # update progress bar
            iterator.set_postfix({
                'Task': fn,
                'Loss': loss
            })
    
    return epoch_loss

# update grads and use sgd instead of updating the parameters directly
def outer_loop_sgd(task_distribution, roberta, model, meta_optimiser, criterion, inner_lr, inner_steps):

    # reptile only ever deals with one task
    tasks_dataloader = DataLoader(task_distribution, batch_size=1, shuffle=True, collate_fn=lambda x: x)
    scheduler = torch.optim.lr_scheduler.LinearLR(meta_optimiser, start_factor=0.5, total_iters=4)

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
                if old.grad is None:
                    old.grad = torch.zeros_like(old)
                old.grad.data.add_(old.data - new.data)
            meta_optimiser.step()
            meta_optimiser.zero_grad()

            # update progress bar
            iterator.set_postfix({
                'Task': fn,
                'Loss': loss
            })
    
    return epoch_loss


# inner loop for the reptile algorithm, returns learned parameters        
def inner_loop(roberta, task, model, criterion, lr, steps):
    
    # unpack task
    fn = task['fn']
    support_set = task['support_set_triplet']
    query_set = task['test_set_triplet']

    # reptile makes no distinction between support and query set
    dataset = ConcatDataset([support_set, query_set])
    dataloader = get_dataloader(dataset, shuffle=True, drop_last=True)

    # train the clone x update steps, adam parameters set in reptile paper
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.0, 0.999))
    loss = train_model(roberta, model, optimiser, dataloader, criterion, max_steps=steps)

    # return the model parameters and the average train loss
    return model.parameters(), loss.item(), fn