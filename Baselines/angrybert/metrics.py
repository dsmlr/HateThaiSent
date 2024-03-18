from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report
from tqdm import tqdm
import torch
import numpy as np


def batch_roc_auc_score(y_true, y_pred):
    num_classes = y_true.shape[1]
    auc_scores = []
    
    for class_idx in range(num_classes):
        y_true_class = y_true[:, class_idx]
        y_pred_class = y_pred[:, class_idx]
        auc = roc_auc_score(y_true_class, y_pred_class)
        auc_scores.append(auc)
    
    return np.mean(auc_scores)


def evaluate_model_sklearn_with_accuracy_and_reports(model, dataloader, device):
    
    model.eval()
    all_predictions_task1 = []
    all_labels_task1 = []
    all_predictions_task2 = []
    all_labels_task2 = []
    
    all_logits_task1 = []
    all_logits_task2 = []
    
    all_one_hot_labels_task1 = []
    all_one_hot_labels_task2 = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            
            batch_nn_tokens, batch_input_ids, batch_attention_masks, batch_labels_task1, batch_labels_task2 = batch

            logits_task1, logits_task2 = model(nn_embed_token_ids = batch_nn_tokens.to(device), bert_input_ids = batch_input_ids.to(device), bert_attn_masks = batch_attention_masks.to(device))
            
            _, predicted_task1 = torch.max(logits_task1, 1)
            _, predicted_task2 = torch.max(logits_task2, 1)
            _, label_task1 = torch.max(batch_labels_task1, 1)
            _, label_task2 = torch.max(batch_labels_task2, 1)
            
            all_logits_task1.extend(logits_task1.cpu().numpy())
            all_logits_task2.extend(logits_task2.cpu().numpy())
            
            all_one_hot_labels_task1.extend(batch_labels_task1.cpu().numpy())
            all_one_hot_labels_task2.extend(batch_labels_task2.cpu().numpy())
            
            all_predictions_task1.extend(predicted_task1.cpu().numpy())
            all_labels_task1.extend(label_task1.cpu().numpy())
            all_predictions_task2.extend(predicted_task2.cpu().numpy())
            all_labels_task2.extend(label_task2.cpu().numpy())
    
    # print(all_predictions_task1)
    # print(all_labels_task1)

    # Calculate accuracy for each task
    f1_task1 = f1_score(all_labels_task1, all_predictions_task1, average='macro')
    f1_task2 = f1_score(all_labels_task2, all_predictions_task2, average='macro')
    
    accuracy_task1 = accuracy_score(all_labels_task1, all_predictions_task1)
    accuracy_task2 = accuracy_score(all_labels_task2, all_predictions_task2)
    
    roc_auc_score_task1 = batch_roc_auc_score(y_pred=np.array(all_logits_task1), y_true=np.array(all_one_hot_labels_task1))
    roc_auc_score_task2 = batch_roc_auc_score(y_pred=np.array(all_logits_task2), y_true=np.array(all_one_hot_labels_task2))

    # Generate classification reports
    report_task1 = classification_report(all_labels_task1, all_predictions_task1)
    report_task2 = classification_report(all_labels_task2, all_predictions_task2)

    return roc_auc_score_task1, roc_auc_score_task2, accuracy_task1, accuracy_task2, f1_task1, f1_task2,  report_task1, report_task2
    # return accuracy_task2, report_task2