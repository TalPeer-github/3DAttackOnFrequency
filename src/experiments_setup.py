import numpy as np
import itertools
import config
from argparse import ArgumentParser
import pandas as pd
from preprocessing.dataset import *
from time import sleep


def parse_args(exp_params):
    """
    :return:
    """

    model_type, dataset_name, batch_size, hidden_channels, num_epochs, num_walks, walks_len, lr, wd = exp_params
    parser = ArgumentParser(description="Point Cloud Attack")
    parser.add_argument('--model_type', type=str, help="Chosen Model", default=model_type)
    parser.add_argument('--dataset_name', type=str, help="Point Cloud Datset", default="ModelNet40")
    parser.add_argument('--hidden_channels', type=int, help="Hidden dimension", default=hidden_channels)
    parser.add_argument('--pooling_type', type=str, help="Pooling method", default='max')

    parser.add_argument('--batch_size', type=int, help="Batch size", default=batch_size)
    parser.add_argument('--lr', type=float, help="Learning rate", default=lr)
    parser.add_argument('--wd', type=float, help="Weight decay", default=wd)
    parser.add_argument('--num_epochs', type=int, help="Number of epochs to run", default=num_epochs)

    parser.add_argument('--num_classes', type=int, help="Number of classes", default=int(dataset_name[-3:-1]))
    parser.add_argument('--num_walks', type=int, help="Number of random walks for each PC", default=num_walks)
    parser.add_argument('--walks_len', type=int, help="Length of random walk", default=walks_len)

    parser.add_argument('--run_mode', type=str, help="exp (Experiment) / eval (Evaluation)", default='eval')
    parser.add_argument('--env', type=str, help="Local (local) / Remote (remote)", default='local')

    return parser.parse_args()


def get_stats(array):
    return np.max(array), np.argmax(array), np.mean(array), np.var(array), np.min(array[::-1]), np.argmin(array[::-1])


if __name__ == "__main__":
    exp_models = ["RnnWalkNet"]
    exp_dataset = ["ModelNet40", "ModelNet10"]
    exp_batch_sizes = [24, 48]
    exp_epochs = [100, 150, 200]
    exp_hidden_channels = [16, 32, 48]
    exp_lrs = [3e-3]
    exp_wds = [5e-3]

    num_walks = [2, 3, 4]
    walks_len = [15, 25, 40, 48, ]

    results = []

    demo_run = None

    params_combinations = list(itertools.product(
        exp_models[:demo_run],
        exp_dataset[:demo_run],
        exp_batch_sizes[:demo_run],
        exp_hidden_channels[:demo_run],
        exp_epochs[:demo_run],
        exp_num_walks[:demo_run],
        exp_walks_len[:demo_run],
        exp_lrs[:demo_run],
        exp_wds[:demo_run],
    ))

    num_experiments = len(params_combinations)

    exp_idx = 0
    for experiment_params in params_combinations:
        args = parse_args(experiment_params)
        model_type, dataset_name, batch_size, hidden_channels, num_epochs, \
            num_walks, walks_len, lr, wd = experiment_params
        config.args = args

        train_acc, val_acc = [], []

        from experiments import run_experiment

        exp_idx += 1
        if exp_idx % 15 == 0 or exp_idx <= 5:
            print("\n=====================================================")
            print(f"\nExperiment [{exp_idx} / {num_experiments}] ->"
                  f"\n\tmodel = {args.model_type}"
                  f"\n\tdataset = {args.dataset_name}"
                  f"\n\thidden_layers = {args.hidden_channels}"
                  f"\n\tnum_walks = {args.num_walks}"
                  f"\n\twalks_len = {args.walks_len}\n"

                  f"\n\tnum_epochs = {args.num_epochs}"
                  f"\n\tbatch_size = {args.batch_size}\n"
                  f"\n\tlearning_rate = {args.lr}"
                  f"\n\tweight_decay = {args.wd}\n\n")

        train_loss, val_loss, train_acc, val_acc, test_acc, test_acc_per_class = run_experiment()
        (train_max_acc, train_max_acc_epoch, train_mean_acc, train_acc_var,
         train_min_acc, train_min_acc_epoch) = get_stats(train_acc)
        (val_max_acc, val_max_acc_epoch, val_mean_acc, val_acc_var,
         val_min_acc, val_min_acc_epoch) = get_stats(val_acc)

        results.append({
            'model': args.model_type,
            'dataset': args.num_conv_layers,
            'num_walks': args.num_walks,
            'walks_len': args.walks_len,
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'hidden_channels': args.hidden_channels,
            'lr': args.lr,
            'weight_decay': args.wd,
            'test_accuracy': test_acc,
            'test_acc_per_class': test_acc_per_class,
            'mean_train_accuracy': train_mean_acc,
            'var_train_accuracy': train_acc_var,
            'max_train_accuracy': train_max_acc,
            'max_train_accuracy_epoch': train_max_acc_epoch,
            'min_train_accuracy': train_min_acc,
            'min_train_accuracy_epoch': train_min_acc_epoch,
            'mean_val_accuracy': val_mean_acc,
            'var_val_accuracy': val_acc_var,
            'max_val_accuracy': val_max_acc,
            'max_val_accuracy_epoch': val_max_acc_epoch,
            'min_val_accuracy': val_min_acc,
            'min_val_accuracy_epoch': val_min_acc_epoch,
            'train_mean_loss': np.mean(train_loss),
            'val_mean_loss': np.mean(val_loss),
            'train_acc_array': train_acc,
            'val_acc_array': val_acc,
            'train_loss_array': train_loss,
            'val_loss_array': val_loss
        })
        if exp_idx % 15 == 0 or exp_idx <= 5:
            print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
            print(
                f"-> Train mean loss = {np.mean(train_loss):.4f} | Validation mean loss = {np.mean(val_loss):.4f}")
            print(
                f"-> Train max accuracy = {train_max_acc:.4f} | Validation max accuracy = {val_max_acc:.4f}")
            print("\n=====================================================")
        sleep(1)

    df_results = pd.DataFrame(results)
    exp_local_time = time.localtime()
    time_string = time.strftime("%m%d, %H%M%S", exp_local_time)
    file_path = f"../results/{time_string}.csv"
    df_results.to_csv(file_path, index=False)
    print(f"Experiments results saved to {file_path}")
