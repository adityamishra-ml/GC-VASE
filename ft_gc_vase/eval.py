import sys
import torch
import numpy as np
from split_model import SplitLatentModel
from utils import get_results, get_split_latents, CustomLoader
from collections import defaultdict
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, f1_score

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./')
parser.add_argument('--model_path', type=str, default='./model.pt')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--channels', type=int, default=256)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--latent_dim', type=int, default=64)
parser.add_argument('--recon_type', type=str, default='mse')
parser.add_argument('--content_cosine', type=int, default=1)
parser.add_argument('--data_line', type=str, default='simple')
parser.add_argument('--random_seed', type=int, default=1)
parser.add_argument('--num_heads', type=int, default=8)  # Added num_heads argument

args = parser.parse_args()

if __name__ == '__main__':
    print(args, file=sys.stdout, flush=True)
    SEED = args.random_seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    IN_CHANNELS = 30
    NUM_LAYERS = args.num_layers
    KERNEL_SIZE = 0
    ADAPTER_DIM = 64  # Define adapter dimension
    
    model = SplitLatentModel(IN_CHANNELS, args.channels, args.latent_dim, NUM_LAYERS, KERNEL_SIZE, ADAPTER_DIM, recon_type=args.recon_type, content_cosine=args.content_cosine)
    model.load_state_dict(torch.load(args.model_path))
    model = model.cuda()
    model.eval()

    with torch.no_grad():
        data_dict = torch.load(args.data_dir+f"{args.data_line}_data.pt")
        data_dict["data"] = (data_dict["data"] - data_dict["data_mean"]) / data_dict["data_std"]
        test_loader = CustomLoader(data_dict, split='test')
        del data_dict

    subject_latents, task_latents, subjects, tasks, runs, losses = get_split_latents(model, test_loader, test_loader.get_dataloader(batch_size=model.batch_size, random_sample=False))
    test_results = get_results(subject_latents, task_latents, subjects, tasks, split=test_loader.split, off_class_accuracy=1)

    # Plot confusion matrix
    print('Plotting confusion matrix...', file=sys.stdout, flush=True)
    subject_cm = test_results['XGB/' + test_loader.split + '/' + 'subject/cm']
    task_cm = test_results['XGB/' + test_loader.split + '/' + 'task/cm']
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    ax = axs.flatten()
    for i, which in enumerate(['subject', 'task', 'paradigm']):
        display_labels = []
        if which == 'subject':
            cm = subject_cm / subject_cm.sum(axis=1, keepdims=True)
            display_labels = [f'S{s}' for s in test_loader.unique_subjects]
        elif which == 'task':
            cm = task_cm / task_cm.sum(axis=1, keepdims=True)
            display_labels = [f'{test_loader.task_to_label[t]}' for t in test_loader.unique_tasks]
        else:
            cm = task_cm[::2,::2] + task_cm[1::2,::2] + task_cm[::2,1::2] + task_cm[1::2,1::2]
            cm = cm / cm.sum(axis=1, keepdims=True)
            display_labels = [f'{test_loader.task_to_label[t].split("/")[0]}' for t in test_loader.unique_tasks[::2]]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=display_labels)
        disp.plot(ax=ax[i], xticks_rotation='vertical', cmap='Blues', values_format='.2f', text_kw={'fontsize': 7} if which == 'task' else None)
        disp.ax_.get_images()[0].set_clim(0, 1)
        if which == 'subject':
            acc = test_results['XGB/test/subject/balanced_accuracy']
        elif which == 'task':
            acc = test_results['XGB/test/task/balanced_accuracy']
        else:
            acc = test_results['XGB/test/task/paradigm_wise_accuracy']
        ax[i].set_title(f'{which.capitalize()}\nBalanced Accuracy: {100*acc:.2f}%', fontsize=12)
    plt.tight_layout()
    
    # Display the plot
    plt.show()

    print(f"Subject Balanced Accuracy: {100*test_results['XGB/test/subject/balanced_accuracy']:.2f}%")
    print(f"Task Balanced Accuracy: {100*test_results['XGB/test/task/balanced_accuracy']:.2f}%")
    print(f"Paradigm-wise Balanced Accuracy: {100*test_results['XGB/test/task/paradigm_wise_accuracy']:.2f}%")

    # Print additional metrics
    print(f"Subject F1 Score: {test_results['XGB/test/subject/f1']:.2f}")
    print(f"Task F1 Score: {test_results['XGB/test/task/f1']:.2f}")
    print(f"Subject Precision: {test_results['XGB/test/subject/precision']:.2f}")
    print(f"Task Precision: {test_results['XGB/test/task/precision']:.2f}")
    print(f"Subject Recall: {test_results['XGB/test/subject/recall']:.2f}")
    print(f"Task Recall: {test_results['XGB/test/task/recall']:.2f}")

    # Print paradigm-wise accuracies
    paradigm_wise_accuracy = test_results['XGB/test/task/paradigm_wise_accuracy']
    print(f"Paradigm-wise Balanced Accuracy: {100*paradigm_wise_accuracy:.2f}%")