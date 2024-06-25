import math
import torch
from torch.nn import functional as Fun
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os

torch.cuda.manual_seed_all(1)
torch.manual_seed(1)
random.seed(1)


def get_similarity(view1, view2):
    norm1 = torch.sum(torch.square(view1), dim=1)
    norm1 = norm1.reshape(-1, 1)
    norm2 = torch.sum(torch.square(view2), dim=1)
    norm2 = norm2.reshape(1, -1)
    similarity = norm1 + norm2 - 2.0 * torch.matmul(view1, view2.transpose(1, 0))
    similarity = -1.0 * torch.max(similarity, torch.zeros(1).cuda())

    return similarity


LOGDIR='/home/yosubs/koa_scratch/human_pose'

def decision_offset(view1, view2, label, name1, name2):
    sim_12 = get_similarity(view1, view2)

    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_12.cpu().detach().numpy(),
                annot=False, cmap="viridis", cbar=True)

    # Save the heatmap to a PNG file
    plt.title(f"sim_12_{name1}_{name2}")
    plt.savefig(os.path.join(LOGDIR, f"{name1}_{name2}_sim_12.png"))
    plt.close()
    
    softmaxed_sim_12 = Fun.softmax(sim_12, dim=1)

    ground = (torch.tensor([i * 1.0 for i in range(view1.size(0))]).cuda()).reshape(-1, 1)

    predict = softmaxed_sim_12.argmax(dim=1)

    plt.figure(figsize=(10, 8))
    sns.heatmap(softmaxed_sim_12.cpu().detach().numpy(),
                annot=False, cmap="viridis", cbar=True, square=True)
    plt.plot(predict.cpu(), np.arange(len(predict.cpu())), color='red',
             marker='o', linestyle='-', linewidth=2, markersize=5)

    k = label.item() * -1

    print('k', k, 'label', label)
    
    # Create the points for the line with y-intercept k
    x_line = np.arange(softmaxed_sim_12.shape[1])
    y_line = x_line + k

    valid_indices = (y_line >= 0) & (y_line < softmaxed_sim_12.shape[0])
    x_line = x_line[valid_indices]
    y_line = y_line[valid_indices]

    plt.plot(x_line, y_line, color='blue', linestyle='--',
             linewidth=2, label=f'Line with y-intercept {k}')

    plt.gca().set_aspect('equal', adjustable='box')

    # Save the heatmap to a PNG file
    plt.title(f"softmaxed_sim_12_{name1}_{name2}")
    plt.savefig(os.path.join(
        LOGDIR, f"{name1}_{name2}_softmaxed_sim_12.png"))
    plt.close()
    
    length1 = ground.size(0)

    frames = []

    for i in range(length1):
        p = predict[i].item()
        g = ground[i][0].item()

        frame_error = (p - g)
        frames.append(frame_error)

    median_frames = np.median(frames)

    num_frames = math.floor(median_frames)

    result = abs(num_frames - label)

    return result



def corresponding(view1, view2, label, name1, name2):
    result = decision_offset(view1[0], view2[0], label, name1[0][1:21], name2[0][1:21])

    return result


if __name__ == '__main__':
    output1 = torch.randn((1, 40, 2176)).cuda()
    output2 = torch.randn((1, 50, 2176)).cuda()

    label = torch.ones(1).cuda() * 10

    frames1, frames2 = corresponding(output1, output2, label)
    print(frames1, frames2.item())
