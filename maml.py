import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader
from torch.autograd import Variable

from helpers import get_batch_size, get_roberta
from training import model_loop, train_model, validate_model

BATCH_SIZE = 5
DEVICE = 'cuda'
SAVE_PATH = 'models/'

# trains a model using maml, checkpoints and logs epoch loss
def maml(task_distribution, model, criterion, epochs, outer_lr, inner_lr):

    # setup
    roberta = get_roberta()
    optimiser = torch.optim.SGD(model.parameters(), lr=outer_lr)

    # train, log and save
    for epoch in range(epochs):

        train_loss = outer_loop(task_distribution, roberta, model, criterion, optimiser, inner_lr)

        print(f"---\ntrain loss: {train_loss}")
        # pick a random task
        # index = random.randint(0, len(task_distribution)-1)
        # task = task_distribution[index]

        # save a checkpoint of the model
        path = SAVE_PATH + 'model.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimiser.state_dict(),
            'train_loss': train_loss,
        }, path)


# outer loop for the maml algorithm, returns the total epoch loss
def outer_loop(task_distribution, roberta, model, criterion, optimiser, inner_lr):
    
    model.train()
    tasks_dataloader = DataLoader(task_distribution, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: x)
    
    epoch_loss = 0.0
    for batch in tasks_dataloader:
        optimiser.zero_grad()
        metaloss = 0
        for task in batch:

            # train a clone
            clone = model.clone()
            params, loss = inner_loop(roberta, task, clone, criterion, inner_lr)
            metaloss += loss
            
            # accumulate gradients (first order approximation)
            # for p, new_p in zip(model.parameters(), params):
            #     if p.grad is None:
            #         p.grad = Variable(torch.zeros(p.size())).to(DEVICE) # verify this line, possibly use below
            #         p.grad = torch.zeros_like(p).to(DEVICE)
            #     p.grad.data.add_(p.data - new_p.data)

            # # accumulate gradients
            # grads = torch.autograd.grad(loss, params, retain_graph=True)
            # for p, g in zip(model.parameters(), grads):
            #     if p.grad is None:
            #         p.grad = g.clone()  # Initialize the gradient for the original model
            #     else:
            #         p.grad += g

        # accumulate gradients
        grads = torch.autograd.grad(metaloss, params, retain_graph=False)
        for p, g in zip(model.parameters(), grads):
            p.grad = g

        # update params
        optimiser.step()
        epoch_loss += metaloss
        
    return epoch_loss

# inner loop for the maml algorithm, trains a clone and returns val loss
def inner_loop(roberta, task, model, criterion, lr, max_steps):
    
    # unpack task
    _, support_set, _, query_set = task

    # create necessary dataloaders
    support_batch_size = get_batch_size(len(support_set))
    query_batch_size = get_batch_size(len(query_set))
    support_dataloader = DataLoader(support_set, batch_size=support_batch_size, shuffle=True, drop_last=True)
    query_dataloader = DataLoader(query_set, batch_size=query_batch_size, shuffle=False, drop_last=False)

    # train and validate the clone over x gradient steps
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    train_model(roberta, model, optimiser, support_dataloader, criterion, max_steps=max_steps)
    val_loss = validate_model(roberta, model, query_dataloader, criterion, maml=True)

    # return the model parameters and the min val loss
    return model.parameters(), val_loss
