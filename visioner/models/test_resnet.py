import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from visioner.utils import count_trainable_parameters
from visioner.models.resnet import VisionResNet


def check_resnet(model):
    batch_size, C, H, W = (16, 3, 64, 64)
    num_class = 30
    ipt = np.random.uniform(0, 1, (batch_size, C, H, W)).astype(np.float32)
    truth = np.random.choice(num_class, batch_size).astype(np.float32)
    count_trainable_parameters(model)
    loss = run_check_net(model, ipt, truth, device='cuda')
    assert np.isclose(loss, 0), 'ResNet cannot converge'


def run_check_net(model, ipt, truth, device='cuda'):
    model = model.to(device)
    model.train()
    ipt = torch.from_numpy(ipt).float().to(device)
    truth = torch.from_numpy(truth).long().to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    # see if it can converge ...
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    for i in range(32):
        optimizer.zero_grad()
        logits = model(ipt)
        loss = criterion(logits, truth)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        if i % 10 == 0:
            print(f'Iter --> {i} | Loss --> {loss.item():.6f}')

    with torch.no_grad():
        logits = model(ipt)
        loss = criterion(logits, truth)
    print(f'Final Loss -->{loss.item():.6f}')
    return loss.cpu().numpy()
