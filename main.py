import os
import argparse

from utils.data import Nsc_Data, Data
from model.sam_model import SamModel
from model.nic_model import NicModel
from model.att_model import AttModel

parser = argparse.ArgumentParser()
# Data parameters
parser.add_argument(
    "--img_dir", type=str, default="./data/nsc_img/img")
parser.add_argument("--split_dir", default="./data/file_list/splits")
parser.add_argument("--pickles_dir", default="./data")
parser.add_argument("--vocab_min", default=5)
parser.add_argument(
    "--comm_min", type=int,
    default=5, help="Minimum number of comms to be included")  # more than 4
parser.add_argument(
    "--subset_users", 
    default=3000, help="Users with the most comments for subset")
parser.add_argument(
    "--load_prep", default=False,
    action="store_true", help="Whether to load preprocessed data from pickle")
parser.add_argument(
    "--prep_path", default=None, 
    help="(Optional) Preprocessed data path"
)
# Annotations generation
parser.add_argument(
    "--annot_gt_folder", default="./annotations"
)
parser.add_argument(
    "--prep_annot", default=False, action="store_true"
)
parser.add_argument(
    "--annot_res_folder", default="./results"
)
parser.add_argument(
    "--annot_name", default="res_1.json"
)
# Training parameters
parser.add_argument("--gpu", required=True)
parser.add_argument(
    "--num_comms", type=int,
    default=5, help="Comments used every batch, cannot be more than min_comms"
)
parser.add_argument(
    "--batch_size", type=int, default=16
)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=50)
# Mode
parser.add_argument(
    "--mode", default="train", 
    choices=["train", "eval_test"], help="Mode, train/eval_test")
# Ckeckpoints/logs
parser.add_argument(
    "--checkpoint_dir", default="./checkpoints"
)
parser.add_argument(
    "--checkpoint", default=""
)
parser.add_argument(
    "--logdir", default="./logs"
)
# Evaluation
parser.add_argument(
    "--res_save_file", default="annot_1.json"
)
parser.add_argument(
    "--eval_set", default="test", choices=["val", "test"]
)
# Experiments NIC/Attention/SAM
parser.add_argument(
    "--experiment", default="sam", choices=["nic", "att", "sam"]
)
args = parser.parse_args()


def main(args):
    data_nsc = Nsc_Data(
        args.img_dir, args.split_dir, 
        args.pickles_dir, args.vocab_min,
        args.subset_users, args.comm_min)
    comm_max_len = 25
    seqattr_max_len = 10  # WIll remain for now
    data_nsc.prepare_data(
        load_prepared=args.load_prep,
        prepared_path=args.prep_path,
        comm_maxlen=comm_max_len, 
        pname_maxlen=seqattr_max_len,
        annot_folder=args.annot_gt_folder,
        annot_prepare=args.prep_annot,
        annot_numref=4)  # 4 was used in NSC paper
    # Save dictionaries for demo usage
    data_nsc.save_dicts("dicts.pickle")
    # Separate data objects, used for batch generation, storing dicts
    train_data = Data(
        data_nsc.train, data_nsc.dictionaries, args.num_comms, 
        comm_max_len)
    val_data = Data(
        data_nsc.val, data_nsc.dictionaries, args.num_comms, 
        comm_max_len)
    test_data = Data(
        data_nsc.test, data_nsc.dictionaries, 1,
        comm_max_len)
    data = {
        "train": train_data,
        "val": val_data,
        "test": test_data}
    eval_set = test_data if args.eval_set == "test" else val_data
    if args.experiment == "sam":
        model = SamModel(args, data, comm_max_len)
        # model.test_datagen()
        if args.mode == "train":
            model.train()
        elif args.mode == "eval_test":
            model.eval_test(
                args.checkpoint,
                args.res_save_file, 
                data=eval_set, # was some mistake during saving, change it later
                checkp_dir=args.checkpoint_dir)
    elif args.experiment == "nic":
        print("Nic Resnet")
        model = NicModel(args, data, comm_max_len)
        # model.test_datagen()
        if args.mode == "train":
            model.train()
        elif args.mode == "eval_test":
            model.eval_test(
                args.checkpoint,
                args.res_save_file, 
                data=eval_set, 
                checkp_dir=args.checkpoint_dir)
    elif args.experiment == "att":
        print("Attention ResNet")
        model = AttModel(args, data, comm_max_len)
        # model.test_datagen()
        if args.mode == "train":
            model.train()
        elif args.mode == "eval_test":
            model.eval_test(
                args.checkpoint,
                args.res_save_file, 
                data=eval_set, 
                checkp_dir=args.checkpoint_dir)

if __name__ == "__main__":
    main(args)
