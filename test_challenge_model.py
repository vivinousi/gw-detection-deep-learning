#! /usr/bin/env python3
import argparse
import h5py
import torch

from utils.eval_utils import get_triggers, get_clusters
from modules.dain import DAIN_Layer
from modules.whiten import CropWhitenNet
from modules.resnet import ResNet54

# weights filename
weights_path = 'trained_models/d4_model/weights.pt'

# Set data type to be used
dtype = torch.float32

sample_rate = 2048
delta_t = 1. / sample_rate
delta_f = 1 / 1.25


def main(args):
    device = 'cuda:0' if torch.cuda.device_count() > 0 else 'cpu'

    base_model = ResNet54().to(device)
    norm = DAIN_Layer(input_dim=2).to(device)

    net = CropWhitenNet(base_model, norm).to(device)
    net.deploy = True
    net.load_state_dict(torch.load(weights_path, map_location=device))

    net.eval()

    # run on foreground
    inputfile = args.inputfile
    outputfile = args.outputfile
    step_size = 3.1
    slice_dur = 4.25
    trigger_threshold = 0.4
    cluster_threshold = 0.35
    var = 0.3

    test_batch_size = 32

    with torch.no_grad():
        triggers = get_triggers(net,
                                inputfile,
                                step_size=step_size,
                                trigger_threshold=trigger_threshold,
                                device=device,
                                verbose=True,
                                batch_size=test_batch_size,
                                whiten=False,
                                slice_length=int(slice_dur * 2048))

    time, stat, var = get_clusters(triggers, cluster_threshold, var=var)

    with h5py.File(outputfile, 'w') as outfile:
        print("Saving clustered triggers into %s." % outputfile)

        outfile.create_dataset('time', data=time)
        outfile.create_dataset('stat', data=stat)
        outfile.create_dataset('var', data=var)

        print("Triggers saved, closing file.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('inputfile', type=str, help="The path to the input data file.")
    parser.add_argument('outputfile', type=str, help="The path where to store the triggers.")

    args = parser.parse_args()

    main(args)
