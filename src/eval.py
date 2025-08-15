from typing import Optional
import torch
from torch.optim import Adam
import torch.nn as nn
from src.models import LogReg
import functools
import numpy as np
from sklearn.metrics import f1_score
import torch.nn.functional as F
from sklearn import metrics

mean_min_f1 = 0

def repeat(n_times):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            global mean_min_f1

            results = [f(*args, **kwargs) for _ in range(n_times)]  
            statistics = {}
            for key in results[0].keys():  
                values = [r[key] for r in results]
                statistics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values)}
                if key == 'F1Mi':
                    mean_min_f1 = statistics[key]['mean']
            # print_statistics(statistics, f.__name__)

            return statistics
        return wrapper
    return decorator

@repeat(10)
def log_regression(z,
                   dataset,
                   evaluator,
                   num_epochs: int = 5000,
                   test_device: Optional[str] = None,
                   split: str = 'rand:0.1',
                   verbose: bool = False,
                   preload_split=None):
    test_device = z.device if test_device is None else test_device
    z = z.detach().to(test_device)
    num_hidden = z.size(1)
    y = dataset[0].y.view(-1).to(test_device)
    num_classes = dataset[0].y.max().item() + 1
    classifier = LogReg(num_hidden, num_classes).to(test_device)
    optimizer = Adam(classifier.parameters(), lr=0.01, weight_decay=0.0)
    split = get_idx_split(dataset, split, preload_split)
    split = {k: v.to(test_device) for k, v in split.items()}
    f = nn.LogSoftmax(dim=-1)
    nll_loss = nn.NLLLoss()
    # ACC
    best_test_acc = 0
    best_val_acc = 0

    for epoch in range(num_epochs):
        classifier.train()
        optimizer.zero_grad()
        output = classifier(z[split['train']])
        loss = nll_loss(f(output), y[split['train']])
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            if 'val' in split:
                test_acc = evaluator.eval({
                    'y_true': y[split['test']].view(-1, 1),
                    'y_pred': classifier(z[split['test']]).argmax(-1).view(-1, 1)
                })['acc']
                val_acc = evaluator.eval({
                    'y_true': y[split['val']].view(-1, 1),
                    'y_pred': classifier(z[split['val']]).argmax(-1).view(-1, 1)
                })['acc']
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc

            else:
                acc = evaluator.eval({
                    'y_true': y[split['test']].view(-1, 1),
                    'y_pred': classifier(z[split['test']]).argmax(-1).view(-1, 1)
                })['acc']

                if best_test_acc < acc:
                    best_test_acc = acc

            if verbose:
                print(f'logreg epoch {epoch}: best test acc {best_test_acc}, '
                      )

    # return {'acc': best_test_acc}
    # #F1
    best_test_f1 = 0
    best_val_f1 = 0
    for epoch in range(num_epochs):
        classifier.train()
        optimizer.zero_grad()
        output = classifier(z[split['train']])
        loss = nll_loss(f(output), y[split['train']])
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            if 'val' in split:
                test_f1 = evaluator.eval({
                    'y_true': y[split['test']].view(-1, 1),
                    'y_pred': classifier(z[split['test']]).argmax(-1).view(-1, 1)
                })['f1']
                val_f1 = evaluator.eval({
                    'y_true': y[split['val']].view(-1, 1),
                    'y_pred': classifier(z[split['val']]).argmax(-1).view(-1, 1)
                })['f1']
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_test_f1 = test_f1

            else:
                f1 = evaluator.eval({
                    'y_true': y[split['test']].view(-1, 1),
                    'y_pred': classifier(z[split['test']]).argmax(-1).view(-1, 1)
                })['f1']

                if best_test_f1 < f1:
                    best_test_f1 = f1

            if verbose:
                print(f'logreg epoch {epoch}: best test acc {best_test_f1}, '
                      )

    return {'acc': best_test_acc,'f1': best_test_f1}

class MulticlassEvaluator:
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def _eval(y_true, y_pred):
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        total = y_true.size(0)
        correct = (y_true == y_pred).to(torch.float32).sum()
        if y_true.is_cuda:
            y_true = y_true.cpu().numpy()
        else:
            y_true = y_true.numpy()
    
        if y_pred.is_cuda:
            y_pred = y_pred.cpu().numpy()
        else:
            y_pred = y_pred.numpy()
        test_micro = f1_score(y_true, y_pred, average='micro')
        test_macro = f1_score(y_true, y_pred, average='macro')
        return (correct / total).item(),test_micro

    def eval(self, res):
        return {'acc': self._eval(**res)[0], 'f1': self._eval(**res)[1]}