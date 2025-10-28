from math import exp,log


def softmax(scores):
    max_score = max(scores)
    exp_scores = [exp(s - max_score) for s in scores]
    total = sum(exp_scores)
    return [s/total for s in exp_scores]
def cross_entry_loss(logits, target_index):
    softmax = softmax(logits)
    p = softmax[target_index]
    eps = 1e-12
    loss = -log(max(p,eps))
    return loss
def cross_entry_loss_batch(batch_logits, batch_target_index):
    if len(batch_logits) != len(batch_target_index):
        raise ValueError("Batch logits and targets must have the same length.")
    if batch_size == 0:
        return 0.0
    
    batch_size = len(batch_logits)
    total_loss = 0.0
    for i in range(batch_size):
        logits = batch_logits[i]
        target_index = batch_target_index[i]
        loss = cross_entry_loss(logits, target_index)
        total_loss += loss
    return total_loss / batch_size

def cross_entry_grad_batch(batch_logits, batch_target_index):
    grad_batch = []
    for i in range(len(batch_logits)):
        probs_i = softmax(batch_logits[i])
        grad_i = probs_i.copy()
        grad_i[batch_target_index[i]] -= 1
        grad_batch.append(grad_i)
    grad_batch = [[g / len(batch_logits) for g in grad] for grad in grad_batch]
    return grad_batch