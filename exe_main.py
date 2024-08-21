import argparse
import torch
import json
import yaml
import os
from main_model import WorkloadDiff
from dataset import get_dataloader
from utils import train, evaluate
import datetime

torch.set_printoptions(profile="full")
parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.1)
parser.add_argument(
    "--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])"
)
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=10)
parser.add_argument('--init-method', type=str, default='tcp://127.0.0.1:23500')
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--world-size',type=int, default=1)


def single_process(config, foldername, args):
    device = torch.device(args.device)
    model = WorkloadDiff(config, device).to(device)
    batch_size = config["train"]["batch_size"]
    train_loader, valid_loader, test_loader = get_dataloader(
        seed=args.seed,
        nfold=args.nfold,
        batch_size=batch_size,
        missing_ratio=config["model"]["test_missing_ratio"],
        config=config,
    )
    # Model training
    if args.modelfolder == "":
        train(
            model,
            config["train"],
            train_loader,
            foldername=foldername,
        )
    # Model loading
    else:
        head, tail = os.path.split(foldername)
        model.load_state_dict(torch.load(os.path.join(head, args.modelfolder + "/model.pth")))

    # Model evaluating
    evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername, config=config)


if __name__ == '__main__':
    start = datetime.datetime.now()
    print('start: ', start)
    args = parser.parse_args()

    # config
    path = "config/base.yaml"
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    config["model"]["test_missing_ratio"] = args.testmissingratio

    # model folder
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_logs = config["others"]["dir_dataset"]

    # Model storage folder
    tmp = dir_logs.split("/")[:-3]
    foldername = os.path.join('/'.join(tmp), "logs/WorkloadDiff_" + current_time)
    config["others"]["model_folder"] = foldername

    os.makedirs(foldername, exist_ok=True)
    with open(os.path.join(foldername, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    print(json.dumps(config, indent=4))
    single_process(config, foldername, args)
    end = datetime.datetime.now()
    print('end: ', end)
    print('TIME USAGE: ', end-start)
