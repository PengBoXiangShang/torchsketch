import numpy as np


class EarlyStoppingOnAccuracy:
    
    def __init__(self, patience = 10, delta = 0):
        
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, acc):

        score = acc

        if self.best_score is None:
            self.best_score = score
            
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score            
            self.counter = 0

    
