import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from helpers import get_roberta, get_dataloader, get_latest_model
from training import train_classifier, train_model

def pretrained(save_path, task_distribution, model, criterion, lr, epochs):

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

    iterator = tqdm(range(epoch, epochs), desc="pretrained")
    for epoch in iterator:

        epoch_loss = outer_loop(task_distribution, roberta, model, criterion, lr)
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

def outer_loop(task_distribution, roberta, model, criterion, lr):

    epoch_loss = 0.0
    tasks_dataloader = DataLoader(task_distribution, batch_size=1, shuffle=True, collate_fn=lambda x: x)
    iterator = tqdm(tasks_dataloader, leave=False)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    for batch in iterator:
        for task in batch:

            # unpack task
            fn = task['fn']
            support_set = task['support_set_triplet']
            query_set = task['test_set_triplet']
            pos_weight = task['pos_weight']

            # no difference between support and test set
            dataset = ConcatDataset([support_set, query_set])
            dataloader = get_dataloader(dataset, shuffle=True, drop_last=True)

            # train on the entire task
            # loss = train_classifier(roberta, None, model, optimiser, dataloader, criterion, pos_weight)
            loss = train_model(roberta, model, optimiser, dataloader, criterion, max_steps=1)
            loss = loss.item()

            epoch_loss += loss

            # update progress bar
            iterator.set_postfix({
                'Task': fn,
                'Loss': loss
            })

    return epoch_loss