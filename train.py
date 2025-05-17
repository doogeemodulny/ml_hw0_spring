import torch
import torch.nn as nn
import torchvision.transforms as transforms
from comet_ml import Experiment
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from tqdm import tqdm, trange

from hparams import config

experiment = Experiment(
    api_key="### your api key ###",
    project_name="hw0_spring_2025",
    auto_param_logging=False,
    disabled=False
)

experiment.log_parameters(config)


def compute_accuracy(preds, targets):
    result = (targets == preds).float().mean()
    return result


def _transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        # transforms.Resize((224, 224)),
    ])
    return transform


def _train_dataset(transform):
    train_dataset = CIFAR10(root='CIFAR10/train',
                            train=True,
                            transform=transform,
                            download=False,
                            )
    return train_dataset


def _test_dataset(transform):
    test_dataset = CIFAR10(root='CIFAR10/test',
                           train=False,
                           transform=transform,
                           download=False,
                           )
    return test_dataset


def _dataloader(dataset, shuffle=False):
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=config["batch_size"], shuffle=shuffle)


def _model():
    model = resnet18(weights=None, num_classes=10, zero_init_residual=config["zero_init_residual"])
    return model


def _criterion():
    return nn.CrossEntropyLoss()


def _optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])


def train_one_batch(i, images, labels, device, model, criterion, optimizer, test_loader):
    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)
    loss = criterion(outputs, labels)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    metrics = {}
    if i % 100 == 0:
        all_preds = []
        all_labels = []

        for test_images, test_labels in test_loader:
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)

            with torch.inference_mode():
                outputs = model(test_images)
                preds = torch.argmax(outputs, 1)

                all_preds.append(preds)
                all_labels.append(test_labels)

        accuracy = compute_accuracy(torch.cat(all_preds), torch.cat(all_labels))

        metrics = {'test_acc': accuracy, 'train_loss': loss}
        experiment.log_metrics(metrics, step=i)
    return metrics


def main(transform, train_loader, test_loader, model, criterion, optimizer):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)
    experiment.set_model_graph(str(model))

    all_metrics = []
    for epoch in trange(config["epochs"]):
        for i, (images, labels) in enumerate(tqdm(train_loader)):
            metrics = train_one_batch(i, images, labels, device, model, criterion, optimizer, test_loader)
            if i % 100 == 0:
                all_metrics.append(metrics)

    torch.save(model.state_dict(), "model.pt")
    experiment.log_model("ResNet18", "model.pt")

    with open("run_id.txt", "w+") as f:
        print(experiment.get_key(), file=f)
        experiment.end()
    return all_metrics


if __name__ == '__main__':
    transform = _transform()
    train_dataset = _train_dataset(transform)
    test_dataset = _test_dataset(transform)

    train_loader = _dataloader(train_dataset, shuffle=True)
    test_loader = _dataloader(test_dataset)
    model = _model()
    criterion = _criterion()
    optimizer = _optimizer(model)
    main(transform, train_loader, test_loader, model, criterion, optimizer)