from .models import CNNClassifier, save_model
from .utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb

class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        """
        Your code here

        Compute mean(-log(softmax(input)_label))

        @input:  torch.Tensor((B,C))
        @target: torch.Tensor((B,), dtype=torch.int64)

        @return:  torch.Tensor((,))

        Hint: Don't be too fancy, this is a one-liner
        """

        return torch.nn.functional.cross_entropy(input, target) 


def train(args):
    from os import path
    model = CNNClassifier()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    """
    Your code here, modify your HW1 code
    
    """
    import matplotlib.pyplot as plt
    import numpy as np
    loss = ClassificationLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
    
    # load the data; train / valid
    TRAIN_PATH = 'data/train'
    VALID_PATH = 'data/valid'
    
    train_data = load_data(TRAIN_PATH)
    valid_data = load_data(VALID_PATH)

    # Run SGD for several epochs
    train_result = []
    valid_result = []

    global_step = 0
    for epoch in range(50):
        train_accuracy = []
        loss_list_training = []
        loss_list_valid = []

        train
        for input, target in train_data:
            optimizer.zero_grad()
            output = model(input)        
            l = loss(output, target)
            l.backward()
            optimizer.step()
            loss_list_training.append(l.item())

            train_logger.add_scalar('loss', np.mean(l.item()), global_step= global_step)
            
            # somehow need to add accuracy?


            global_step += 1

        for input, target in valid_data:
            output = model(input)        
            l_v = loss(output, target)
            loss_list_valid.append(l_v.item())
        print(f"epoch =  {epoch} | training loss =  {sum(loss_list_training) / len(loss_list_training)} | valid loss =  {sum(loss_list_valid) / len(loss_list_valid)}")

        train_result.append(sum(loss_list_training) / len(loss_list_training))
        valid_result.append(sum(loss_list_valid) / len(loss_list_valid))

    plt.plot(train_result,label = 'training')
    plt.plot(valid_result,label = 'valid')
    plt.legend()
    plt.show()

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
