import torch.optim.lr_scheduler

from pyg_dataset import create_dataset, create_dataloaders
from proxy_network import *
import proxy_network


def labels_to_onehot(labels_tensor, num_classes):
    return torch.nn.functional.one_hot(labels_tensor, num_classes=num_classes)


def optimizer_lr_update(optimizer, args, loss_stablility_count=1):
    lr_period = args.cyclic_lr_period
    iteration = optimizer.iterations.numpy()
    if loss_stablility_count >= 1:
        iteration = iteration % lr_period
        far_from_mid = np.abs(iteration - lr_period / 2)
        fraction_from_mid = np.abs(lr_period / 2 - far_from_mid) / (lr_period / 2)
        factor = fraction_from_mid + (1 - fraction_from_mid) * args.min_lr_factor
        optimizer.learning_rate.assign(args.learning_rate * factor)


def lr_scaling(x):
    x_th = 500e3 / args.cycle_opt_prms.step_size
    if x < x_th:
        return 1.0
    else:
        return 0.5


def train_proxy(device='cpu'):
    def train_epoch(one_label_per_model):
        cummulative_loss = 0.
        for pc_batch in train_loader:
            _, model_features, labels_ = pc_batch  # TODO - varify correctness (of dataset attributes)
            shape = model_features.shape
            model_features = torch.reshape(model_features, (-1, shape[-2], shape[-1]))
            with torch.autograd:
                if one_label_per_model:
                    labels = torch.reshape(torch.T(torch.stack((labels_,) * args.num_walks)), (-1,))  # TODO - validate
                    predictions = proxy_model(model_features)
                else:
                    labels = torch.reshape(labels_, (-1, shape[-2]))
                    skip = args.min_seq_len
                    predictions = proxy_model(model_features)[:, skip:]
                    labels = labels[:, skip + 1:]

                if args.train_loss == ['manifold_CE']:  # TODO - check about this param
                    labels = labels_to_onehot(labels_tensor=labels, num_classes=args.num_classes)
                    acc = manifold_segmentation_train_acc(labels, predictions)
                    loss = manifold_seg_loss(labels, predictions)
                else:
                    acc = segmentation_train_acc(labels, predictions)
                    loss = segmentation_train_loss(labels, predictions)
                cummulative_loss += torch.reduce_sum(proxy_model.losses)
                # TODO - Check for a suitable torch version & choose - cummulative_loss/loss (original = loss).
                #  I think it should be "losses += torch.sum(losses)" with respect to some dimention.

            gradients = torch.autograd.grad(loss, proxy_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, proxy_model.trainable_variables))
            # TODO - Check for a suitable torch version
            lr_scheduler.step()

            train_losses.append(train_logs['segmentation_train_loss'](cummulative_loss))
            train_accuracies.append(torch.mean(acc))  # TODO - add accuracy (no rush)
        return train_losses, train_accuracies

    def test_epoch(one_label_per_model):
        test_losses, test_accuracies = [], []

        for pc_batch in val_loader:
            _, model_features, labels_ = pc_batch  # TODO - varify correctness (of dataset attributes)
            with torch.no_grad():
                shape = model_features.shape
                model_features = torch.reshape(model_features, (-1, shape[-2], shape[-1]))
                if one_label_per_model:
                    labels = torch.reshape(torch.T(torch.stack((labels_,) * args.num_walks)), (-1,))
                    predictions = proxy_model(model_features)
                else:
                    labels = torch.reshape(labels_, (-1, shape[-2]))
                    skip = args.min_seq_len
                    predictions = proxy_model(model_features, training=False)[:, skip:]
                    labels = labels[:, skip + 1:]

                if params.train_loss == ['manifold_cros_entr']:  # TODO - check about this param
                    labels = labels_to_onehot(labels_tensor=labels, num_classes=args.num_classes)

            test_accuracies.append(test_accuracy(labels, predictions))

        return test_losses, test_accuracies

    args = config.args
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_name = "ModelNet40" if args.dataset_name is None else args.dataset_name

    pre_transform = get_pre_transform()
    transform = get_transform(args.num_points_to_sample)  # TODO - set args default = 2048

    train_dataset, val_dataset = create_dataset(dataset_name, pre_transform, transform)
    # TODO - align with PointCloudDataset class, so 1 train batch tuple -> (<something>, model_features, labels)
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset)

    proxy_model = proxy_network.RnnWalkNet(args, args.n_classes, args.net_input_dim, init_net_using)
    proxy_model = proxy_model.to(device)
    optimizer = torch.optim.Adam(proxy_model.parameters(), lr=args.lr, betas=args.betas, weight_decay=args.wd)
    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer, base_lr=args.base_lr, max_lr=args.max_lr,
                                                     scale_fn=lr_scaling, step_size_up=args.scheduler_step_size,
                                                     scale_mode='cycle')

    if args.class_probabilities_target:  # TODO - check relevance
        apply_softmax = True
    else:
        apply_softmax = False

    segmentation_train_acc = torcheval.metrics.MulticlassAccuracy()
    manifold_segmentation_train_acc = torcheval.metrics.MulticlassAccuracy()
    segmentation_train_loss = torch.nn.CrossEntropyLoss(reduction=args.loss_reduction)
    # TODO - read about loss_reduction "sum" VS "mean"
    manifold_segmentation_train_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=False)
    # TODO - check about "log_target = True"

    train_logs = {"segmentation_train_loss": torcheval.functional.mean(),
                  "segmentation_train_loss_2": segmentation_train_loss,
                  "manifold_segmentation_train_loss": manifold_segmentation_train_loss,
                  "segmentation_train_acc": segmentation_train_acc}
    # TODO - understand which objective is relevant to which case (Tal)
    train_accuracies, val_accuracies = [], []
    train_losses, val_losses = [], []

    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_acc = train_epoch(one_label_per_model=one_label_per_model)
        val_loss, val_acc = test_epoch(one_label_per_model=one_label_per_model)

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    return train_losses, val_losses, train_accuracies, val_accuracies
