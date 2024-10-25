import torch

from helpers import EarlyStopper

# function to train and validate an embedding model using triplet loss
def model_loop(roberta, model, optimiser, criterion, train_dataloader, val_dataloader, epochs, logging):
    train_losses = []
    val_losses = []
    early_stopper = EarlyStopper()

    # track best model
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):

        train_loss = train_model(roberta, model, optimiser, train_dataloader, criterion)
        val_loss = validate_model(roberta, model, val_dataloader, criterion)

        train_loss = train_loss.item()
        val_loss = val_loss.item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if logging:
            print(str(epoch) + ": ", train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

        if early_stopper.early_stop(val_loss):
            break

    # return model to best state
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return min(val_losses)

# function to perform one epoch of training on an embedding model
def train_model(roberta, model, optimiser, dataloader, criterion, max_steps=None):
    model.train()
    total_loss = 0.0

    for i, batch in enumerate(dataloader):

        # allow a maximum number of training steps for reptile
        if max_steps is not None and i >= max_steps:
            break

        optimiser.zero_grad()

        anchor_input_ids, anchor_attention_mask, positive_input_ids, positive_attention_mask, negative_input_ids, negative_attention_mask = batch
        
        with torch.no_grad():
            anchor_emb = roberta(anchor_input_ids, anchor_attention_mask).last_hidden_state
            positive_emb = roberta(positive_input_ids, positive_attention_mask).last_hidden_state
            negative_emb = roberta(negative_input_ids, negative_attention_mask).last_hidden_state

        anchor_emb = model(anchor_emb, anchor_attention_mask)
        positive_emb = model(positive_emb, positive_attention_mask)
        negative_emb = model(negative_emb, negative_attention_mask)

        loss = criterion(anchor_emb, positive_emb, negative_emb)
        loss.backward()
        optimiser.step()
        total_loss += loss
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

# function to perform one epoch of validation on an embedding model
# maml flag indicates whether gradients should be computed
def validate_model(roberta, model, dataloader, criterion, maml=False):

    model.eval()
    total_loss = 0.0
    context = torch.no_grad() if not maml else torch.enable_grad()

    for batch in dataloader:
       
        anchor_input_ids, anchor_attention_mask, positive_input_ids, positive_attention_mask, negative_input_ids, negative_attention_mask = batch
        
        with torch.no_grad():
            anchor_emb = roberta(anchor_input_ids, anchor_attention_mask).last_hidden_state
            positive_emb = roberta(positive_input_ids, positive_attention_mask).last_hidden_state
            negative_emb = roberta(negative_input_ids, negative_attention_mask).last_hidden_state

        with context:
            anchor_emb = model(anchor_emb, anchor_attention_mask)
            positive_emb = model(positive_emb, positive_attention_mask)
            negative_emb = model(negative_emb, negative_attention_mask)
            
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            loss.detach()
            total_loss += loss

    avg_loss = total_loss / len(dataloader)
    return avg_loss


# function to train and validate a classifier model using BCE loss
def classifier_loop(roberta, model, classifier, optimiser, criterion, train_dataloader, val_dataloader, pos_weight, epochs, logging):

    train_losses = []
    val_losses = []
    early_stopper = EarlyStopper()

    # track best classifier
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        
        train_loss = train_classifier(roberta, model, classifier, optimiser, train_dataloader, criterion, pos_weight)
        val_loss = validate_classifier(roberta, model, classifier, val_dataloader, criterion, pos_weight)

        train_loss = train_loss.item()
        val_loss = val_loss.item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if logging:
            print(str(epoch) + ": ", train_loss, val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = classifier.state_dict()

        if early_stopper.early_stop(val_loss):
            break
    
    # return classifier to best state
    if best_state is not None:
        classifier.load_state_dict(best_state)

    return min(val_losses)

# function to perform one epoch of training on a classifier model
def train_classifier(roberta, model, classifier, optimiser, dataloader, criterion, pos_weight):
    if model:
        model.eval()
    classifier.train()
    total_loss = 0.0

    for batch in dataloader:
        optimiser.zero_grad()
        
        input_ids, attention_mask, labels = batch
        with torch.no_grad():
            roberta_embeddings = roberta(input_ids, attention_mask).last_hidden_state
            if model is None:
                # use the roberta cls token if there is no model
                model_embeddings = roberta_embeddings[:,0,:]
            else:
                model_embeddings = model(roberta_embeddings, attention_mask)
        logits = classifier(model_embeddings)
        
        loss = criterion(logits.float(), labels.unsqueeze(1).float(), pos_weight=pos_weight)
        loss.backward()
        optimiser.step()
        total_loss += loss

    avg_loss = total_loss / len(dataloader)
    return avg_loss

# function to perform one epoch of validation on a classifier model     
def validate_classifier(roberta, model, classifier, dataloader, criterion, pos_weight):
    if model:
        model.eval()
    classifier.eval()
    total_loss = 0.0

    for batch in dataloader:
        input_ids, attention_mask, labels = batch
        with torch.no_grad():
            roberta_embeddings = roberta(input_ids, attention_mask).last_hidden_state
            if model is None:
                # use the roberta cls token if there is no model
                model_embeddings = roberta_embeddings[:,0,:]
            else:
                model_embeddings = model(roberta_embeddings, attention_mask)
            logits = classifier(model_embeddings)
            loss = criterion(logits.float(), labels.unsqueeze(1).float(), pos_weight=pos_weight)
        total_loss += loss

    avg_loss = total_loss / len(dataloader)
    return avg_loss


# function to test the performance of a trained model and classifier
def test(roberta, model, classifier, dataloader):
    if model:
        model.eval()
    classifier.eval()
    
    all_labels = []
    all_probs = []
    all_embeds = []
    for batch in dataloader:
        with torch.no_grad():
            input_ids, attention_mask, labels = batch
            roberta_embeddings = roberta(input_ids, attention_mask).last_hidden_state
            if model is None:
                # use the roberta cls token if there is no model
                model_embeddings = roberta_embeddings[:,0,:]
            else:
                model_embeddings = model(roberta_embeddings, attention_mask)
            logits = classifier(model_embeddings)
            probs = torch.sigmoid(logits).flatten()
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
            all_embeds.append(model_embeddings.cpu())
    
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)
    all_embeds = torch.cat(all_embeds)
    return all_labels, all_probs, all_embeds