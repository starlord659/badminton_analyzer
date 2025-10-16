import os
import json
import time
import random
import copy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Device setup
# -----------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# -----------------------------------------------------------------------------
# Dataset class
# -----------------------------------------------------------------------------
class BadmintonKeypointDataset(Dataset):
    def __init__(self, data_dir, split='train', transforms=None):
        self.data_dir = data_dir
        self.split = split
        self.transforms = transforms

        # Load COCO-style annotations
        annotation_file = os.path.join(data_dir, split, '_annotations.coco.json')
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)

        # Organize annotations per image
        self.annotations = {}
        for ann in self.coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in self.annotations:
                self.annotations[image_id] = []
            self.annotations[image_id].append(ann)

        # Image metadata
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.image_ids = list(self.annotations.keys())

        print(f"Loaded {len(self.image_ids)} images for {split} split")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.images[image_id]
        annotations = self.annotations[image_id]

        # Load image
        image_path = self._get_image_path(image_info['file_name'])
        image = Image.open(image_path).convert('RGB')
        image = transforms.ToTensor()(image)

        # Prepare targets
        targets = self._prepare_targets(annotations, image_info)

        if self.transforms:
            image, targets = self.transforms(image, targets)

        return image, targets

    def _get_image_path(self, file_name):
        """Try multiple path variations to locate the image file."""
        possible_paths = [
            os.path.join(self.data_dir, self.split, file_name),
            os.path.join(self.data_dir, self.split, file_name.replace('_jpg', '.jpg')),
            os.path.join(self.data_dir, self.split, os.path.basename(file_name)),
        ]

        clean_name = os.path.basename(file_name)
        possible_paths.append(os.path.join(self.data_dir, self.split, clean_name))

        for path in possible_paths:
            if os.path.exists(path):
                return path

        raise FileNotFoundError(f"Could not find image: {file_name}")

    def _prepare_targets(self, annotations, image_info):
        num_objs = len(annotations)

        boxes, keypoints, labels = [], [], []

        for ann in annotations:
            # Bounding box
            x1, y1, w, h = ann['bbox']
            x2, y2 = x1 + w, y1 + h
            boxes.append([x1, y1, x2, y2])

            # Keypoints
            kpts = np.array(ann['keypoints']).reshape(-1, 3)
            keypoints.append(kpts)

            # Labels (only one class: court)
            labels.append(1)

        boxes = torch.FloatTensor(boxes)
        keypoints = torch.FloatTensor(keypoints)
        labels = torch.LongTensor(labels)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros(num_objs, dtype=torch.int64)
        image_id = torch.tensor([image_info['id']])

        return {
            'boxes': boxes,
            'labels': labels,
            'keypoints': keypoints,
            'area': area,
            'iscrowd': iscrowd,
            'image_id': image_id
        }


# -----------------------------------------------------------------------------
# Model definition
# -----------------------------------------------------------------------------
def get_model(num_keypoints=22, num_classes=2):
    model = keypointrcnn_resnet50_fpn(pretrained=True)

    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )

    # Replace keypoint predictor
    in_features_kp = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
    model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(in_features_kp, num_keypoints)

    return model


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def collate_fn(batch):
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    metric_logger = defaultdict(list)

    for i, (images, targets) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch}")):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        for k, v in loss_dict.items():
            metric_logger[k].append(v.item())
        metric_logger['total_loss'].append(losses.item())

        if i % print_freq == 0:
            avg_loss = np.mean(metric_logger['total_loss'][-print_freq:])
            print(f"Batch {i}/{len(data_loader)}, Loss: {avg_loss:.4f}")

    return {k: np.mean(v) for k, v in metric_logger.items()}


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    metric_logger = defaultdict(list)

    for images, targets in tqdm(data_loader, desc="Evaluating"):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Predictions (not used here but could be for metrics)
        _ = model(images)

        # Compute validation loss
        model.train()
        loss_dict = model(images, targets)
        model.eval()

        for k, v in loss_dict.items():
            metric_logger[k].append(v.item())

    return {k: np.mean(v) for k, v in metric_logger.items()}


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def visualize_keypoints(image, keypoints, boxes=None, skeleton=None):
    if torch.is_tensor(image):
        image = image.permute(1, 2, 0).cpu().numpy()

    if torch.is_tensor(keypoints):
        keypoints = keypoints.cpu().numpy()

    if torch.is_tensor(boxes):
        boxes = boxes.cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)

    # Draw boxes
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 fill=False, color='red', linewidth=2)
            ax.add_patch(rect)

    # Draw keypoints
    if keypoints is not None:
        for kpts in keypoints:
            for i, (x, y, v) in enumerate(kpts):
                if v > 0:
                    color = 'red' if v == 2 else 'yellow'
                    ax.plot(x, y, 'o', color=color, markersize=8)
                    ax.text(x, y, str(i), fontsize=8, color='white',
                            bbox=dict(boxstyle="round,pad=0.1",
                                      facecolor='black', alpha=0.7))

    ax.set_title("Badminton Court Keypoints")
    plt.tight_layout()
    plt.show()


def show_sample_data(dataset, num_samples=3):
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    for idx in indices:
        image, target = dataset[idx]
        print(f"Sample {idx}:")
        print(f"  Image shape: {image.shape}")
        print(f"  Objects: {len(target['boxes'])}")
        print(f"  Keypoints shape: {target['keypoints'].shape}")

        skeleton = None
        if 'categories' in dataset.coco_data:
            for cat in dataset.coco_data['categories']:
                if 'skeleton' in cat:
                    skeleton = cat['skeleton']
                    break

        visualize_keypoints(image, target['keypoints'], target['boxes'], skeleton)


# -----------------------------------------------------------------------------
# Main training logic
# -----------------------------------------------------------------------------
def main():
    CONFIG = {
        'data_dir': 'D:/capstone/badminton_analyzer/data/Court',
        'batch_size': 4,
        'num_epochs': 30,
        'learning_rate': 0.001,
        'num_keypoints': 22,
        'num_classes': 2,  # background + court
        'save_dir': 'D:/capstone/badminton_analyzer/models',
        'print_freq': 10,
    }

    os.makedirs(CONFIG['save_dir'], exist_ok=True)

    print("Configuration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")

    # Datasets
    print("\nLoading datasets...")
    train_dataset = BadmintonKeypointDataset(CONFIG['data_dir'], 'train')
    valid_dataset = BadmintonKeypointDataset(CONFIG['data_dir'], 'valid')

    # Show samples
    print("\nSample training data:")
    show_sample_data(train_dataset, num_samples=2)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                              shuffle=True, collate_fn=collate_fn, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['batch_size'],
                              shuffle=False, collate_fn=collate_fn, num_workers=2)

    print(f"Train batches: {len(train_loader)}")
    print(f"Valid batches: {len(valid_loader)}")

    # Model
    print("\nCreating model...")
    model = get_model(CONFIG['num_keypoints'], CONFIG['num_classes'])
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Training
    print("\nStarting training...")
    best_loss = float('inf')
    train_losses, valid_losses = [], []

    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{CONFIG['num_epochs']}")

        train_metrics = train_one_epoch(model, optimizer, train_loader, device,
                                        epoch, CONFIG['print_freq'])
        valid_metrics = evaluate(model, valid_loader, device)

        scheduler.step()

        train_loss = train_metrics['total_loss']
        valid_loss = sum(valid_metrics.values())
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # Save checkpoints
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'config': CONFIG
            }, os.path.join(CONFIG['save_dir'], 'best_model.pth'))
            print("Saved new best model!")

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'valid_losses': valid_losses,
                'config': CONFIG
            }, os.path.join(CONFIG['save_dir'], f'checkpoint_epoch_{epoch + 1}.pth'))

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(CONFIG['save_dir'], 'training_curves.png'))
    plt.show()

    print("\nTraining completed!")
    return model, train_losses, valid_losses


# -----------------------------------------------------------------------------
# Inference helpers
# -----------------------------------------------------------------------------
@torch.no_grad()
def predict_keypoints(model, image_path, device, confidence_threshold=0.5):
    model.eval()

    image = Image.open(image_path).convert('RGB')
    image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)

    predictions = model(image_tensor)
    pred = predictions[0]

    scores = pred['scores'].cpu().numpy()
    keep = scores > confidence_threshold

    return {
        'boxes': pred['boxes'][keep].cpu().numpy(),
        'keypoints': pred['keypoints'][keep].cpu().numpy(),
        'scores': scores[keep],
        'image': np.array(image)
    }


def test_model_inference(model, test_dataset, num_samples=3):
    model.eval()
    indices = random.sample(range(len(test_dataset)), min(num_samples, len(test_dataset)))

    for idx in indices:
        image, target = test_dataset[idx]
        with torch.no_grad():
            prediction = model([image.to(device)])

        pred = prediction[0]

        print(f"Ground Truth - Sample {idx}:")
        visualize_keypoints(image, target['keypoints'], target['boxes'])

        print(f"Prediction - Sample {idx}:")
        if len(pred['boxes']) > 0:
            visualize_keypoints(image, pred['keypoints'], pred['boxes'])
            print(f"Confidence scores: {pred['scores'].cpu().numpy()}")
        else:
            print("No detections found")


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    model, train_losses, valid_losses = main()

    print("\nTesting model inference...")
    try:
        test_dataset = BadmintonKeypointDataset('D:/capstone/badminton_analyzer/data/Court', 'test')
        test_model_inference(model, test_dataset, num_samples=2)
    except Exception as e:
        print(f"Could not load test dataset: {e}")
        print("Testing on validation dataset instead...")
        valid_dataset = BadmintonKeypointDataset('D:/capstone/badminton_analyzer/data/Court', 'valid')
        test_model_inference(model, valid_dataset, num_samples=2)

    print("\nNotebook execution completed!")
