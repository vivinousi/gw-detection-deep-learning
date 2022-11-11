import argparse
import logging
import os
import numpy as np

np.random.seed(0)
import json
import pylab
from bisect import bisect

import torch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

from torch import nn
from torch.utils.data import DataLoader

from utils.train_utils import WarmUpLR, initialize_xavier, progress_bar
from utils.dataset import SlicerDataset, SlicerDatasetSNR
from modules.loss import reg_BCELoss
from modules.resnet import ResNet54Double
from modules.dain import DAIN_Layer
from modules.whiten import CropWhitenNet
from utils.eval_utils import get_clusters, get_triggers


def decode_snr_schedule(sch_str):
    # .e.g.: '5:1-3,5:1-1.5,5:1-1.25,10:1-1,2:0.8-1.2 means 5 epochs of SNR range (1,3),
    # followed by 5 epochs of (1, 1.5), then 5 epochs of (1,1.25), etc
    steps = sch_str.split(',')
    epochs = []
    s_ranges = []
    for step in steps:
        ep, range_ = step.split(':')
        epochs.append(int(ep))
        s_min, s_max = range_.split('-')
        s_ranges.append([float(s_min), float(s_max)])
    epochs.append(1)
    s_ranges.append(([-np.inf, np.inf]))
    epochs = np.cumsum(epochs)
    return epochs, s_ranges


def get_snr_by_epoch(sch_epochs, sch_ranges, epoch):
    print(f'Epoch: {epoch}, schedule: {[(sch_epoch, sch_range) for sch_epoch, sch_range in zip(sch_epochs, sch_ranges)]}')
    index = bisect(sch_epochs, epoch)
    if index >= len(sch_epochs):
        return sch_ranges[-1]
    return sch_ranges[index]


# Set default weights filename
default_weights_fname = 'weights.pt'

# Set data type to be used
dtype = torch.float32

sample_rate = 2048
delta_t = 1. / sample_rate
delta_f = 1 / 1.25


def main(args):
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        logging.info(f'Creating output directory {output_dir}...')
        os.makedirs(output_dir)

    # where to save/load the weights after training
    weights_path = os.path.join(output_dir, default_weights_fname)

    dataset = 4

    val_hdf = os.path.join(args.data_dir, f'dataset-{dataset}/v2/val_background_s24w6d1_1.hdf')
    val_npy = os.path.join(args.data_dir, f'dataset-{dataset}/v2/val_injections_s24w6d1_1.25s.npy')

    train_device = args.train_device

    base_model = ResNet54Double().to(train_device)
    norm = DAIN_Layer(input_dim=2).to(train_device)
    base_model.apply(initialize_xavier)

    net = CropWhitenNet(base_model, norm).to(train_device)

    if args.resume_from is not None:
        print(f'Loading weights: {args.resume_from}')
        net.load_state_dict(torch.load(args.resume_from, map_location=train_device))

    validation_dataset = SlicerDataset(val_hdf, val_npy, slice_len=int(args.slice_dur * sample_rate),
                                       slice_stride=int(args.slice_stride * sample_rate),
                                       max_seg_idx=int(np.floor(args.slice_dur)))
    val_dl = DataLoader(validation_dataset, batch_size=100, shuffle=True, num_workers=args.num_workers,
                        pin_memory=train_device)

    background_hdf = os.path.join(args.data_dir, f'dataset-{dataset}/v2/train_background_s24w61w_1.hdf')
    injections_hdf = os.path.join(args.data_dir, f'dataset-{dataset}/v2/train_injections_s24w61w_1.hdf')
    inj_npy = os.path.join(args.data_dir, f'dataset-{dataset}/v2/train_injections_s24w61w_1.25s_all.npy')

    sch_epochs, sch_ranges = decode_snr_schedule(args.snr_schedule)
    min_snr, max_snr = sch_ranges[0]
    training_dataset = SlicerDatasetSNR(background_hdf, inj_npy, slice_len=int(args.slice_dur * sample_rate),
                                        slice_stride=int(args.slice_stride * sample_rate),
                                        max_seg_idx=int(np.floor(args.slice_dur)),
                                        injections_hdf=injections_hdf, min_snr=min_snr, max_snr=max_snr,
                                        p_augment=args.p_augment)
    batch_size = args.batch_size
    train_dl = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
                          pin_memory=train_device)

    # setup loss
    loss = reg_BCELoss(dim=2)

    # setup optimizer
    learning_rate = args.learning_rate
    opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
    milestones = [int(m) for m in args.lr_milestones.split(',')]
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=args.gamma)
    n_wrm = args.warmup_epochs
    wrm = WarmUpLR(opt, int(len(train_dl) * n_wrm))

    # train/val loop
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    n_epochs = args.epochs
    # for epoch in tqdm(range(n_epochs), desc="Optimizing network"):
    for epoch in range(n_epochs):

        net.train()
        # train losses
        training_running_loss = 0.
        training_batches = 0
        # train accuracy
        total = 0
        correct = 0
        # val accuracy
        total_val = 0
        correct_val = 0

        s_min, s_max = get_snr_by_epoch(sch_epochs, sch_ranges, epoch)
        training_dataset.set_snr_range(s_min, s_max)

        for idx, (training_samples, training_labels, training_inj_times) in enumerate(train_dl):
            training_samples = training_samples.to(device=train_device)
            training_labels = training_labels.to(device=train_device)

            # Optimizer step on a single batch of training data
            opt.zero_grad()
            if epoch < n_wrm:
                wrm.step()

            training_output = net(training_samples, training_inj_times)
            training_loss = loss(training_output, training_labels)
            training_loss.backward()
            # Clip gradients to make convergence somewhat easier
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.clip_norm)
            # Make the actual optimizer step and save the batch loss
            opt.step()

            # get predictions & gt to measure accuracy
            _, predicted = training_output.max(1)
            _, gt = training_labels.max(1)
            total += training_output.size(0)
            correct += predicted.eq(gt).sum().item()
            train_acc = 100. * (correct / total)

            # update running loss
            training_running_loss += training_loss.clone().cpu().item()
            training_batches += 1

            progress_bar(idx, len(train_dl),
                         f'Epoch {epoch} | Loss {training_running_loss / training_batches:.2f} | Acc {train_acc:.2f}')

        # Evaluation on the validation dataset
        net.eval()
        with torch.no_grad():

            # error analysis in last epoch
            positive_correct = 0
            positive_total = 0
            positive_corr = []

            negative_correct = 0
            negative_total = 0
            negative_corr = []

            val_predictions = []
            val_groundtruth = []

            validation_running_loss = 0.
            validation_batches = 0
            for val_idx, (validation_samples, validation_labels, validation_inj_times) in enumerate(val_dl):
                validation_samples = validation_samples.to(device=train_device)
                validation_labels = validation_labels.to(device=train_device)

                # Evaluation of a single validation batch
                validation_output = net(validation_samples, validation_inj_times)
                validation_loss = loss(validation_output, validation_labels)

                # get predictions & gt to measure accuracy
                _, predicted_val = validation_output.max(1)
                _, gt_val = validation_labels.max(1)
                total_val += validation_output.size(0)
                correct_val += predicted_val.eq(gt_val).sum().item()
                val_acc = 100. * (correct_val / total_val)
                validation_running_loss += validation_loss.clone().cpu().item()

                # get predictions & gt to measure accuracy
                _, predicted_val = validation_output.max(1)
                val_predictions.extend(predicted_val)

                _, gt_val = validation_labels.max(1)
                val_groundtruth.extend(gt_val.cpu().numpy())
                pos_idx = gt_val == 0
                neg_idx = ~pos_idx
                positive_corr.extend(predicted_val[pos_idx].eq(gt_val[pos_idx]).cpu().numpy())
                # val_loss.extend(validation_loss[pos_idx].cpu().numpy())
                negative_corr.extend(predicted_val[neg_idx].eq(gt_val[neg_idx]).cpu().numpy())
                positive_total += pos_idx.sum()
                negative_total += neg_idx.sum()
                positive_correct += predicted_val[pos_idx].eq(gt_val[pos_idx]).sum().item()
                negative_correct += predicted_val[neg_idx].eq(gt_val[neg_idx]).sum().item()
                validation_batches += 1
                progress_bar(val_idx, len(val_dl),
                             f'Validation | Loss {validation_running_loss / validation_batches:.2f} | Acc {val_acc:.2f}'
                             f' (+:{100 * (positive_correct/positive_total):.3f}%,-:{100 * (negative_correct/negative_total):.3f}%)')

        # Print information on the training and validation loss in the current epoch and save current network state
        validation_loss = validation_running_loss / validation_batches
        training_loss = training_running_loss / training_batches
        output_string = '%04i Train Loss: %f | Val Loss: %f || Train Acc: %.3f%% | Val Acc: %.3f%% (+:%.3f%%,-:%.3f%%)' % (
            epoch, training_loss, validation_loss,
            train_acc, val_acc, 100 * (positive_correct / positive_total), 100 * (negative_correct / negative_total))
        train_losses.append(training_loss)
        val_losses.append(validation_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        logging.info(output_string)
        sch.step()

        torch.save(net.state_dict(), weights_path)
        # if (epoch+1) in sch_epochs:
        #     torch.save(net.state_dict(), os.path.join(output_dir, f'epoch_{epoch + 1}.pt'))

    # training over, save network
    torch.save(net.state_dict(), weights_path)

    # training plots
    fig, axs = pylab.subplots(1, 2, sharex=True, figsize=(10, 5))
    fig.suptitle('Training loss & acc')
    axs[0].plot(train_losses, label='train')
    axs[0].plot(val_losses, label='val')
    axs[0].title.set_text('Loss')

    axs[1].plot(train_accs, label='train')
    axs[1].plot(val_accs, label='val')
    axs[1].title.set_text('Accuracy')

    fig.savefig(f'{output_dir}/training_curves.png')

    # Print information on the training and validation loss in the current epoch and save current network state
    # validation_loss = validation_running_loss / validation_batches
    positive_acc = 100 * (positive_correct / positive_total)
    negative_acc = 100 * (negative_correct / negative_total)
    print(f'Validation accuracy: {val_acc}% (positive: {positive_acc}%, negative: {negative_acc}%)')

    # save to json for plotting later
    with open(os.path.join(output_dir, 'training_stats.json'), 'w') as f:
        train_dict = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        }
        json.dump(train_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    training_group = parser.add_argument_group('training')

    parser.add_argument('--verbose', action='store_true', help="Print update messages.")
    parser.add_argument('-o', '--output-dir', type=str, help="Path to the directory where the outputs will be stored.")
    parser.add_argument('--data-dir', type=str, help='Path to the directory where data is stored.')
    parser.add_argument('--slice-dur', type=float, default=3.25, help='Duration (in s) of original slices, e.g., 3.25.'
                                                                      'After the PSD is calculated these slices are further cropped to 1s.')
    parser.add_argument('--slice-stride', type=float, default=2., help='Slice stride.')

    training_group.add_argument('--resume-from', type=str, default=None, help='If set, weights will be loaded from this path and training will resume from these weights.')
    training_group.add_argument('--learning-rate', type=float, default=5e-5,
                                help="Learning rate of the optimizer. Default: 0.00005")
    training_group.add_argument('--lr-milestones', type=str, default='20,50', help='Epochs at which we multiply lr by gamma')
    training_group.add_argument('--gamma', type=float, default=0.5, help='Rate to multiply learning rate by at milestones.')
    training_group.add_argument('--epochs', type=int, default=10, help="Number of training epochs. Default: 10")
    training_group.add_argument('--snr-schedule', type=str, default='5:15-100,5:1-100', help='Formatted string for waveform SNR filtering.'
                                                                                             'First number is epochs, then the range is separated by -')
    training_group.add_argument('--batch-size', type=int, default=32,
                                help="Batch size of the training algorithm. Default: 32")
    training_group.add_argument('--warmup-epochs', type=float, default=0,
                                help="If >0, the learning rate will be annealed from 1e-8 to learning rate in warmup_epochs")
    training_group.add_argument('--clip-norm', type=float, default=100.,
                                help="Gradient clipping norm to stabilize the training. Default 100.")
    training_group.add_argument('--p-augment', type=float, default=0.25,
                                help="Percentage of samples where L1 noise is randomly replaced with different segment.")
    training_group.add_argument('--train-device', type=str, default='cpu',
                                help="Device to train the network. Use 'cuda' for the GPU."
                                     "Also, 'cpu:0', 'cuda:1', etc. (zero-indexed). Default: cpu")
    training_group.add_argument('--num-workers', type=int, default=8,
                                help="Number of workers to use when loading training data. Default: 8")

    args = parser.parse_args()

    # logging.basicConfig(format='%(levelname)s | %(asctime)s: %(message)s', level=logging.INFO,
    #                     datefmt='%d-%m-%Y %H:%M:%S')

    main(args)
