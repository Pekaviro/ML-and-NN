from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision import datasets
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPModel, CLIPProcessor


dataset = datasets.Food101(root='./data', split='test', download=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется: {device}")
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)

test_dataset = datasets.Food101(
    root='./data',
    split='test',
    download=True,
    transform=None
    )

def collate_fn(batch):
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    return images, labels

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

class_names = test_dataset.classes
print(f"Классов: {len(class_names)}")
print(f"Примеры: {class_names[:5]}")

# text_descriptions = [f"a photo of a {name}" for name in class_names]  # 80.70%
# text_descriptions = [f"{name}" for name in class_names]      77.64%
text_descriptions = [f"a photograph of a {name}" for name in class_names]
text_inputs = processor(
    text=text_descriptions,
    return_tensors="pt",
    padding=True
    ).to(device)

with torch.no_grad():
    text_features = model.get_text_features(**text_inputs).pooler_output
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)


all_preds = []
all_labels = []
model.eval()

with torch.no_grad():
    for images, labels in test_loader:
        image_inputs = processor(
            images=list(images),
            return_tensors="pt",
            padding=True
            ).to(device)
        
        image_features = model.get_image_features(**image_inputs).pooler_output
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        similarity = image_features @ text_features.T
        preds = similarity.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
accuracy = np.mean(all_preds == all_labels)
print(f"Zero-shot accuracy: {accuracy * 100:.2f}%")



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)

n_classes = len(class_names)
figsize = (max(10, n_classes * 0.3), max(10, n_classes * 0.3))

fig, ax = plt.subplots(figsize=figsize)
disp.plot(ax=ax, include_values=False)
ax.set_aspect('auto')

ax.set_xticklabels([]) 
ax.set_yticklabels([]) 

ax.set_xlabel('') 
ax.set_ylabel('') 
plt.title("Confusion Matrix (CLIP zero-shot)")
plt.savefig(f'скрины/confusion_matrix_photograph.png')
plt.show()

for i, class_name in enumerate(class_names):
    mask = all_labels == i
    if mask.sum() == 0:
        continue
    class_acc = np.mean(all_preds[mask] == all_labels[mask])
    print(f"{class_name:30s}: {class_acc * 100:.1f}%")

# templates = [
#     "a photo of a {}",
#     "a photograph of a {}",
#     "an image of a {}",
#     "a picture of a {}",
#     "a {} in a photo",
#     ]
# text_features_ensemble = []
# for class_name in class_names:
#     descriptions = [t.format(class_name) for t in templates]
#     text_inputs = processor(
#         text=descriptions,
#         return_tensors="pt",
#         padding=True
#         ).to(device)
#     with torch.no_grad():
#         features = model.get_text_features(**text_inputs)
#         features = features / features.norm(dim=-1, keepdim=True)
#         avg_feature = features.mean(dim=0)
#         avg_feature = avg_feature / avg_feature.norm()
#     text_features_ensemble.append(avg_feature)
# text_features_ensemble = torch.stack(text_features_ensemble)