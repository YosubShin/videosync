import os
import torch
from dataloader.data_pipeline import DataPipeline
from network.adjusted_stgcn import Adjusted_GCN
from torch.backends import cudnn
from torch.utils.data import DataLoader
import datetime as time
from matching import corresponding
from tqdm import tqdm
import config_test
import wandb
import numpy as np

opt = config_test.config()

torch.cuda.manual_seed_all(opt.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(opt.seed)

from scipy.stats import t
import math

def calculate_margin_of_error(data, confidence_level=0.95):
    """
    Calculate the margin of error for the mean of a sample data using t-distribution.

    Args:
    data (list): list of sample data.
    confidence_level (float): The confidence level (0 < confidence_level < 1).

    Returns:
    float: The margin of error for the mean of the sample data.
    """
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1")

    # Calculate the sample mean and standard deviation
    std_dev = np.std(data)
    n = len(data)

    # Calculate the degrees of freedom
    df = n - 1

    # Determine the critical t-value for the given confidence level
    # We use two-tailed, hence (1 + confidence_level) / 2
    alpha = (1 - confidence_level) / 2
    t_critical = t.ppf(1 - alpha, df)

    # Calculate the margin of error
    margin_of_error = t_critical * (std_dev / math.sqrt(n))

    return margin_of_error


def test(checkpoint_path):
    test_dataset_path = opt.test_dataset.format_map({
        'dataset_name': opt.dataset_name
    })
    test_dataset = DataPipeline(test_dataset_path)

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=opt.test_batchsize,
                                 shuffle=False,
                                 num_workers=opt.workers,
                                 drop_last=True)

    length = len(test_dataset)

    checkpoint = torch.load(checkpoint_path)

    model = Adjusted_GCN(opt.in_channels, opt.layout,
                         opt.strategy, opt.edge_importance_weighting)
    model.load_state_dict(checkpoint['model'])
    model = model.cuda()
    model.eval()

    metrics = {
        "median": {
            'abs_error': 0, 'rank_1': 0, 'rank_5': 0, 'rank_10': 0,
            'relative_error_1': 0, 'relative_error_2': 0,
            'rate_10': 0, 'rate_30': 0, 'rate_50': 0, 'res': []
        },
        "dtw": {
            'abs_error': 0, 'rank_1': 0, 'rank_5': 0, 'rank_10': 0,
            'relative_error_1': 0, 'relative_error_2': 0,
            'rate_10': 0, 'rate_30': 0, 'rate_50': 0, 'res': []
        }
    }

    number = 0

    for batch_i, batch_data in enumerate(tqdm(test_dataloader)):
        view1, view2 = batch_data

        data1, label1, info1 = view1
        data2, label2, info2 = view2

        data1 = data1.cuda()
        data2 = data2.cuda()

        label1 = label1.cuda()
        label2 = label2.cuda()

        label = label1 - label2

        output1 = model(data1)
        output2 = model(data2)

        frames_dict = corresponding(output1, output2, label, info1['video_name'], info2['video_name'])

        for method, frames in frames_dict.items():
            frames = frames.item()
            metrics[method]['res'].append(frames)

            # Absolute error
            metrics[method]['abs_error'] += frames

            # Ranking metrics
            if frames <= 1.0:
                metrics[method]['rank_1'] += 1
            if frames <= 5.0:
                metrics[method]['rank_5'] += 1
            if frames <= 10.0:
                metrics[method]['rank_10'] += 1

            # Ground Truth
            GT = abs(label.item())

            # Relative error
            metrics[method]['relative_error_1'] += frames
            metrics[method]['relative_error_2'] += GT

            if GT == 0:
                number += 1
                continue

            # Rate metrics
            relative_rate = frames / GT
            if relative_rate <= 0.1:
                metrics[method]['rate_10'] += 1
            if relative_rate <= 0.3:
                metrics[method]['rate_30'] += 1
            if relative_rate <= 0.5:
                metrics[method]['rate_50'] += 1

    # Calculate final metrics for each method
    for method, method_metrics in metrics.items():
        method_metrics['res'].sort()

        # Calculate mean and margin of error for abs_error
        mean_abs_error = np.mean(method_metrics['res'])
        abs_error_margin = calculate_margin_of_error(method_metrics['res'])

        method_metrics['abs_error'] = mean_abs_error
        method_metrics['abs_error_std'] = np.std(method_metrics['res'])
        method_metrics['abs_error_margin'] = abs_error_margin  # New: margin of error

        # Calculate remaining metrics
        method_metrics['rank_1'] /= length
        method_metrics['rank_5'] /= length
        method_metrics['rank_10'] /= length

        method_metrics['relative_error'] = method_metrics['relative_error_1'] / method_metrics['relative_error_2']
        valid_length = length - number
        method_metrics['rate_10'] /= valid_length
        method_metrics['rate_30'] /= valid_length
        method_metrics['rate_50'] /= valid_length

        # Log metrics
        wandb.log({
            f'{method}/abs_error': method_metrics['abs_error'],
            f'{method}/abs_error_std': method_metrics['abs_error_std'],
            f'{method}/abs_error_margin': method_metrics['abs_error_margin'],  # Log margin of error
            f'{method}/rank_1': method_metrics['rank_1'],
            f'{method}/rank_5': method_metrics['rank_5'],
            f'{method}/rank_10': method_metrics['rank_10'],
            f'{method}/relative_error': method_metrics['relative_error'],
            f'{method}/rate_10': method_metrics['rate_10'],
            f'{method}/rate_30': method_metrics['rate_30'],
            f'{method}/rate_50': method_metrics['rate_50'],
        })

        # Print metrics
        print(f'Method: {method}')
        print(f'Abs_error: {method_metrics["abs_error"]}, Abs_error_std: {method_metrics["abs_error_std"]}, '
              f'Abs_error_margin: {method_metrics["abs_error_margin"]}')  # New print statement
        print(f'Rank_1: {method_metrics["rank_1"]}, Rank_5: {method_metrics["rank_5"]}, '
              f'Rank_10: {method_metrics["rank_10"]}')
        print(f'Relative_error: {method_metrics["relative_error"]}, Rate_10: {method_metrics["rate_10"]}, '
              f'Rate_30: {method_metrics["rate_30"]}, Rate_50: {method_metrics["rate_50"]}')

    return metrics


if __name__ == '__main__':
    wandb.init(
        # set the wandb project where this run will be logged
        project="videosync",

        # track hyperparameters and run metadata
        config=opt
    )
    checkpoint_path = os.path.join('./model/ntu_syn.pth')
    metrics = test(checkpoint_path)
    for method, method_metrics in metrics.items():
        print(f'Accuracy({method}): {method_metrics["abs_error"]}')

    wandb.finish()

    # for i in range(0, 400):
    #     print(f'epoch {i:03d}')

    #     start = time.datetime.now()

    #     pth_name = str(i).zfill(3) + '.pth'
    #     checkpoint_path = os.path.join(opt.work_dir, 'train', pth_name)

    #     accuracy = test(checkpoint_path)
    #     print('Accuracy', accuracy)

    #     end = time.datetime.now()
    #     print('Spent time: ', end - start)
