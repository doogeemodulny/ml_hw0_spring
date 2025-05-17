import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import torch.nn as nn
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10

from train import compute_accuracy, \
    _model, \
    _criterion, \
    _optimizer, \
    _dataloader, \
    _test_dataset, \
    _train_dataset, \
    _transform, \
    train_one_batch, \
    main


config = dict(
    batch_size=64,
    learning_rate=1e-5,
    weight_decay=0.01,
    epochs=2,
    zero_init_residual=False,
)


@pytest.fixture
def train_dataset():
    # note: реализуйте и протестируйте подготовку данных (скачиание и препроцессинг)
    transform = _transform()
    assert transform is not None, "Преобразования не должны быть пустыми"

    train_dataset = _train_dataset(transform)
    assert len(train_dataset) > 0, "Тренировочный набор данных должен содержать элементы"

    return train_dataset


@pytest.fixture
def test_dataset():
    # note: реализуйте и протестируйте подготовку данных (скачиание и препроцессинг)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        # transforms.Resize((224, 224)),
    ])

    test_dataset = _test_dataset(transform)

    assert len(test_dataset) > 0, "Тренировочный набор данных должен содержать элементы"
    return test_dataset


@pytest.mark.parametrize(["device"], [["cpu"]])  # , ["cuda"]])
def test_train_on_one_batch(device, train_dataset, test_dataset):
    # note: реализуйте и протестируйте один шаг обучения вместе с метрикой

    assert len(train_dataset) > 0, "Тренировочный набор данных пуст"
    assert len(test_dataset) > 0, "Тестовый набор данных пуст"

    train_loader = _dataloader(train_dataset, shuffle=True)
    assert len(train_loader) > 0, "Загрузчик данных должен содержать элементы"

    test_loader = _dataloader(test_dataset)
    assert len(test_loader) > 0, "Загрузчик данных должен содержать элементы"

    model = _model()
    assert isinstance(model, nn.Module), "Модель не является экземпляром nn.Module"
    assert model.fc.out_features == 10, "Количество выходов модели должно быть 10"

    model.to(device)
    # wandb.watch(model)

    criterion = _criterion()
    assert isinstance(criterion, nn.CrossEntropyLoss), "Критерий должен быть CrossEntropyLoss"

    optimizer = _optimizer(model)
    assert optimizer is not None, "Оптимизатор не должен быть None"

    (images, labels) = next(iter(train_loader))
    images = images.to(device)
    labels = labels.to(device)

    metrics = train_one_batch(0, images, labels, device, model, criterion, optimizer, test_loader)
    print(metrics)


def test_model_initialization():
    model = _model()
    assert isinstance(model, nn.Module), "Модель не инициализирована корректно"
    assert hasattr(model, 'fc'), "Модель не содержит слой fc"


def test_learning_rate():
    model = _model()
    optimizer = _optimizer(model)
    for param_group in optimizer.param_groups:
        assert param_group['lr'] == 1e-5, "Начальная скорость обучения должна быть 1e-5"


def test_compute_accuracy():
    dummy_output = torch.tensor([2, 3, 2, 5])
    dummy_labels = torch.tensor([2, 3, 2, 4])

    accuracy = compute_accuracy(dummy_output, dummy_labels)

    assert accuracy == 0.75, f"Ожидалось 0.75, но получено {accuracy}"


def test_model_save_and_load(tmp_path):
    model = _model()
    save_path = tmp_path / "model.pth"

    torch.save(model.state_dict(), save_path)
    assert save_path.is_file(), "Файл модели не был создан"

    model_loaded = _model()
    model_loaded.load_state_dict(torch.load(save_path))
    assert model_loaded is not None, "Загруженная модель не должна быть None"


def test_training():
    # note: реализуйте и протестируйте полный цикл обучения модели (обучение, валидацию, логирование, сохранение артефактов)
    transform = _transform()
    train_dataset = _train_dataset(transform)
    test_dataset = _test_dataset(transform)

    train_loader = _dataloader(train_dataset, shuffle=True)
    test_loader = _dataloader(test_dataset)
    model = _model()
    criterion = _criterion()
    optimizer = _optimizer(model)
    metrics_log = main(transform, train_loader, test_loader, model, criterion, optimizer)
    print(metrics_log)
    assert metrics_log[-1]["train_loss"] < metrics_log[0]["train_loss"] # происходит ли вообще обучение
    assert metrics_log[-1]["test_acc"] > metrics_log[0]["test_acc"] # адекватная accuracy
    assert (metrics_log[0]["train_loss"] - metrics_log[10]["train_loss"]) > \
           (metrics_log[-10]["train_loss"] - metrics_log[-1]["train_loss"]) #адекватная форма кривой обучения
    
