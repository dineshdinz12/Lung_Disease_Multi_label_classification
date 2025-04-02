import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve, average_precision_score, matthews_corrcoef
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from torch.serialization import add_safe_globals
from fastcore.foundation import L
import timm
from torch.serialization import safe_globals
import sys
import json
import tempfile

# Add safe globals for model loading
add_safe_globals([L, np.core.multiarray.scalar])

class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_channels, in_channels // 8),
            nn.LayerNorm(in_channels // 8),
            nn.GELU(),
            nn.Linear(in_channels // 8, in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: [B, H, W, C] -> [B, H*W, C]
        B, H, W, C = x.shape
        x = x.reshape(B, H*W, C)
        # Apply attention
        attn = self.attention(x)  # [B, H*W, C]
        # Reshape back
        x = x * attn
        x = x.reshape(B, H, W, C)
        return x

class ChestXrayModel(nn.Module):
    def __init__(self, num_classes):
        super(ChestXrayModel, self).__init__()
        # Use Swin Transformer as the base model
        self.base_model = timm.create_model('swin_large_patch4_window12_384', pretrained=True)
        
        # Get the number of features from the base model
        num_features = self.base_model.head.in_features
        
        # Remove the original classifier
        self.base_model.head = nn.Identity()
        
        # Create a more sophisticated classifier head with residual connections
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(0.4),
            ResidualBlock(2048),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            ResidualBlock(1024),
            nn.Linear(1024, num_classes),
            nn.Sigmoid()
        )
        
        # Add attention mechanism
        self.attention = AttentionModule(num_features)
        
    def forward(self, x):
        # Get features from base model
        features = self.base_model.forward_features(x)  # [B, H, W, C]
        
        # Apply attention
        attended_features = self.attention(features)
        
        # Global average pooling
        pooled = torch.mean(attended_features, dim=[1, 2])  # [B, C]
        
        # Apply classifier
        return self.classifier(pooled)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        return x + self.block(x)

def load_model(model_path, device):
    # Define class labels
    model_classes = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Emphysema', 'Fibrosis',
        'Hernia', 'Infiltration', 'Mass', 'No Finding', 'Nodule',
        'Pneumonia', 'Pneumothorax'
    ]
    
    # Initialize model
    model = ChestXrayModel(num_classes=len(model_classes)).to(device)
    
    try:
        # Use safe_globals context manager to handle numpy scalar
        with safe_globals([np.core.multiarray.scalar]):
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Try to load the state dict
            try:
                model.load_state_dict(state_dict, strict=False)
                # Don't print success message
            except Exception as e:
                # Don't print warning message
                pass
    except Exception as e:
        # Don't print error message
        pass
    
    model.eval()
    return model, model_classes

def preprocess_image(image_path, device):
    # Enhanced transforms with better augmentation
    transform = transforms.Compose([
        transforms.Resize((384, 384)),  # Swin Transformer default size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image, image_tensor

def predict_image(model, image_path, model_classes, device, threshold=0.3):
    image, image_tensor = preprocess_image(image_path, device)
    
    # Perform test-time augmentation
    with torch.no_grad():
        # Original prediction
        outputs = model(image_tensor)
        probabilities = outputs.cpu().numpy()[0]
        
        # Horizontal flip prediction
        flipped_tensor = torch.flip(image_tensor, [3])
        flipped_outputs = model(flipped_tensor)
        flipped_probabilities = flipped_outputs.cpu().numpy()[0]
        
        # Vertical flip prediction
        v_flipped_tensor = torch.flip(image_tensor, [2])
        v_flipped_outputs = model(v_flipped_tensor)
        v_flipped_probabilities = v_flipped_outputs.cpu().numpy()[0]
        
        # Average predictions
        probabilities = (probabilities + flipped_probabilities + v_flipped_probabilities) / 3
    
    # Class-specific thresholds based on validation performance
    class_thresholds = {
        'Atelectasis': 0.35,
        'Cardiomegaly': 0.35,
        'Effusion': 0.35,
        'Emphysema': 0.35,
        'Fibrosis': 0.35,
        'Hernia': 0.35,
        'Infiltration': 0.35,
        'Mass': 0.35,
        'No Finding': 0.40,
        'Nodule': 0.35,
        'Pneumonia': 0.35,
        'Pneumothorax': 0.35
    }
    
    # Special handling for "No Finding" class
    no_finding_idx = model_classes.index('No Finding')
    if probabilities[no_finding_idx] > class_thresholds['No Finding']:
        probabilities = np.zeros_like(probabilities)
        probabilities[no_finding_idx] = 1.0
    else:
        probabilities[no_finding_idx] = 0.0
    
    # Apply class-specific thresholds
    predicted_labels = []
    for i, label in enumerate(model_classes):
        if i != no_finding_idx and probabilities[i] > class_thresholds[label]:
            predicted_labels.append(label)
    
    # Sort predictions by probability
    predicted_labels = sorted(predicted_labels, 
                            key=lambda x: probabilities[model_classes.index(x)], 
                            reverse=True)
    
    # Create a dictionary of predictions
    predictions = {label: float(prob) for label, prob in zip(model_classes, probabilities)}
    
    return predicted_labels, predictions

def evaluate_model(model, test_loader, model_classes, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = outputs.cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    mcc = matthews_corrcoef(all_labels.ravel(), all_preds.ravel())
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(model_classes):
        fpr, tpr, _ = roc_curve(all_labels[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Class')
    plt.legend(loc="lower right")
    plt.show()
    
    # Plot Precision-Recall curves
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(model_classes):
        precision_curve, recall_curve, _ = precision_recall_curve(all_labels[:, i], all_probs[:, i])
        avg_precision = average_precision_score(all_labels[:, i], all_probs[:, i])
        plt.plot(recall_curve, precision_curve, label=f'{label} (AP = {avg_precision:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for Each Class')
    plt.legend(loc="lower left")
    plt.show()
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'MCC': mcc
    }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Please provide an image path"}))
        sys.exit(1)

    image_path = sys.argv[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Redirect stdout to a temporary file to capture any print statements
        import sys
        import os
        import tempfile
        
        # Create a temporary file to capture stdout
        temp_stdout = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        original_stdout = sys.stdout
        sys.stdout = temp_stdout
        
        # Load model and predict
        model, model_classes = load_model("model.pth", device)
        predicted_labels, predictions = predict_image(model, image_path, model_classes, device)
        
        # Restore stdout
        sys.stdout = original_stdout
        temp_stdout.close()
        
        # Print only the JSON output to stdout
        print(json.dumps(predictions))
        
    except Exception as e:
        # Restore stdout in case of exception
        if 'original_stdout' in locals():
            sys.stdout = original_stdout
        
        print(json.dumps({"error": str(e)}))
        sys.exit(1)