import unittest
import torch
import torch.nn as nn
from src.model import get_model
from src.dataset import get_transforms


class TestModel(unittest.TestCase):
    def test_get_model_resnet18(self):
        """Verify resnet18 initialization with custom number of classes."""
        num_classes = 10
        model = get_model(architecture="resnet18", num_classes=num_classes)
        self.assertIsInstance(model, nn.Module)
        self.assertEqual(model.fc.out_features, num_classes)

    def test_model_forward_pass(self):
        """Verify that a batch of CIFAR-10 images produces expected output logits shape."""
        model = get_model(architecture="resnet18", num_classes=10)
        model.eval()
        batch_size = 4
        dummy_input = torch.randn(batch_size, 3, 32, 32)
        with torch.no_grad():
            output = model(dummy_input)
        self.assertEqual(output.shape, (batch_size, 10))

    def test_unsupported_architecture(self):
        """Verify ValueError is raised when requesting an unsupported architecture."""
        with self.assertRaises(ValueError):
            get_model(architecture="transformer_v1", num_classes=10)

    def test_dataset_transforms(self):
        """Verify train and validation transforms pipeline."""
        train_transform = get_transforms(train=True)
        val_transform = get_transforms(train=False)
        self.assertIsNotNone(train_transform)
        self.assertIsNotNone(val_transform)


if __name__ == "__main__":
    unittest.main()
