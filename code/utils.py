# Basics + Viz
import pickle
import time
import datetime
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns

# Models
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# Metrics
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

# Suppress warnings
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    try:
        roc = roc_auc_score(pred_flat, labels_flat)
    except ValueError:
        roc = 0
    return f1_score(pred_flat, labels_flat), roc, np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def single_topic_train(device, optimizer, scheduler, model, epochs, train_dataloader, test_dataloader, seed_val=42, validate=True, get_f1s=False, verbose=False):
    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # Set the seed value all over the place to make this reproducible.
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    # Store the average loss after each epoch so we can plot them.
    training_losses = []
    testing_losses = []
    f1s = []
    
    for epoch_i in range(0, epochs):
        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.
        if verbose:
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')
        
        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode.
        model.train()
        
        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)

                # Report progress.
                if verbose:
                    print(f'  Batch {step:>5,}  of  {len(train_dataloader):>5,}.    Elapsed: {elapsed:}.')

            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()        

            outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)

            loss = outputs[0]
            total_train_loss += outputs[0].item()

            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()
        
        # Calculate the average loss over the training data.
        avg_train_loss = total_train_loss / len(train_dataloader)            
        training_losses.append(avg_train_loss)
        
        if verbose:
            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epoch took: {:}".format(format_time(time.time() - t0)))
        
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.
        if validate:
            if verbose:
                print("")
                print("Running Validation...")

            # Measure how long the testing takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_test_loss = 0

            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            model.eval()

            # Tracking variables 
            predictions , true_labels = [], []

            # Evaluate data for one epoch
            for batch in test_dataloader:

                # Add batch to GPU
                batch = tuple(t.to(device) for t in batch)

                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_labels = batch

                # Telling the model not to compute or store gradients, saving memory and
                # speeding up validation
                with torch.no_grad():        
                    outputs = model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask, 
                                    labels=b_labels)

                total_test_loss += outputs[0].item()

                logits = outputs[1]
                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Store predictions and true labels
                predictions.append(logits)
                true_labels.append(label_ids)

            # Combine the predictions for each batch into a single list of 0s and 1s.
            flat_predictions = [item for sublist in predictions for item in sublist]
            flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

            # Combine the correct labels for each batch into a single list.
            flat_true_labels = [item for sublist in true_labels for item in sublist]

            # Calculate f1
            f1 = f1_score(flat_true_labels, flat_predictions)
            f1s.append(f1)

            avg_test_loss = total_test_loss / len(test_dataloader)            
            testing_losses.append(avg_test_loss)

            # Report the final accuracy for this validation run.
            if verbose:
                ra = roc_auc_score(flat_true_labels, flat_predictions)
                acc = np.sum(flat_predictions == flat_true_labels) / len(flat_true_labels)
                print("  F1 Score: {0:.2f}".format(f1))
                print("  ROC_AUC: {0:.2f}".format(ra))
                print("  Accuracy: {0:.2f}".format(acc))
                print("  Validation took: {:}".format(format_time(time.time() - t0)))
                print("  Average validation loss: {0:.2f}".format(avg_test_loss))
    if not validate:
        return model, training_losses
    elif get_f1s:
        return model, training_losses, testing_losses, f1s
    else:
        return model, training_losses, testing_losses

def draw_test_train_curve(test_losses, train_losses, topic_name):
    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12,6)

    # Plot the learning curve.
    plt.plot(train_losses, 'b-o', label='Train')
    plt.plot(test_losses, 'r-o', label='Test')

    # Label the plot.
    plt.title(f"Train/Test loss - {topic_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()
    
def run_evaluation(model, test_dataloader, device, verbose=False):
    # Put model in evaluation mode
    model.eval()
    
    # Tracking variables 
    predictions , true_labels = [], []

    # Predict 
    for batch in test_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
          # Forward pass, calculate logit predictions
          outputs = model(b_input_ids, token_type_ids=None, 
                          attention_mask=b_input_mask)

        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)
        
    # Create results
    matthews_set = []

    # Evaluate each test batch using Matthew's correlation coefficient
    if verbose:
        print('Calculating Matthews Corr. Coef. for each batch...')

        # For each input batch...
        for i in range(len(true_labels)):
            # The predictions for this batch are a 2-column ndarray (one column for "0" 
            # and one column for "1"). Pick the label with the highest value and turn this
            # in to a list of 0s and 1s.
            pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
            # Calculate and store the coef for this batch.  
            matthews = matthews_corrcoef(true_labels[i], pred_labels_i)

            print("Predicted Label for Batch " + str(i) + " is " + str(pred_labels_i))
            print("True Label for Batch " + str(i) + " is " + str(true_labels[i])) 
            print("Matthew's correlation coefficient for Batch " + str(i) + " is " + str(matthews))
            matthews_set.append(matthews)
    
    # Combine the predictions for each batch into a single list of 0s and 1s.
    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    # Combine the correct labels for each batch into a single list.
    flat_true_labels = [item for sublist in true_labels for item in sublist]

    # Calculate the MCC
    mcc = matthews_corrcoef(flat_true_labels, flat_predictions)
    f1 = f1_score(flat_true_labels, flat_predictions)
    ra = roc_auc_score(flat_true_labels, flat_predictions)

    cm = confusion_matrix(flat_true_labels, flat_predictions)
    sns.heatmap(cm, annot=True)
    plt.show()

    print('MCC: %.3f' % mcc)
    print('ROC_AUC: %.3f' % ra)
    print('F1: %.3f' % f1)
    print(classification_report(flat_true_labels, flat_predictions))

def train_test_dataloader(*args, topic, batch_size):
    result = []
    for df in args:
        x = torch.tensor(df.bert.values.tolist())
        y = torch.tensor(df[topic].values.astype(int))
        masks = torch.tensor(df.attention.values.tolist())
    
        data = TensorDataset(x, masks, y)
        sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
        
        result.append(dataloader)
    return tuple(result)

def augmented_dataloader(train_df, test_df, topic, batch_size):
    # Test data should NOT have any augmented data in it.
    # Therefore the process is the same as the past.
    test_x = torch.tensor(test_df.bert.values.tolist())
    test_y = torch.tensor(test_df[topic].values.astype(int))
    test_masks = torch.tensor(test_df.attention.values.tolist())
    
    test_data = TensorDataset(test_x, test_masks, test_y)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    
    # Train data should have the augmented data added.
    # BUT we only want the positive cased augmented. No need in throwing in more negative cases.
    aug_index = train_df[topic] == 1
    
    train_x = train_df.bert.values.tolist() + train_df.bert_aug[aug_index].values.tolist()
    train_y = train_df[topic].values.astype(int).tolist() + train_df[topic][aug_index].values.astype(int).tolist()
    train_masks = train_df.attention.values.tolist() + train_df.attention[aug_index].values.tolist()
    
    train_x = torch.tensor(train_x)
    train_y = torch.tensor(train_y)
    train_masks = torch.tensor(train_masks)
    
    train_data = TensorDataset(train_x, train_masks, train_y)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    
    return train_dataloader, test_dataloader

def augmented_validationloader(train_df, test_df, validation_df, topic, batch_size):
    train_dataloader, test_dataloader = augmented_dataloader(train_df, test_df, topic, batch_size)
    
    # Test data should NOT have any augmented data in it.
    # Therefore the process is the same as the past.
    validation_x = torch.tensor(validation_df.bert.values.tolist())
    validation_y = torch.tensor(validation_df[topic].values.astype(int))
    validation_masks = torch.tensor(validation_df.attention.values.tolist())
    
    validation_data = TensorDataset(validation_x, validation_masks, validation_y)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
    
    return train_dataloader, test_dataloader, validation_dataloader