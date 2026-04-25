import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt  


RANDOM_SEED = 42

# Воспроизводимость
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Устройство
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {device}")

vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device)

# Заморозить все параметры — сеть используется только как экстрактор признаков
for param in vgg.parameters():
    param.requires_grad = False
vgg.eval()

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

def load_image(path, size=256):
    image = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std), 
    ])
    return transform(image).unsqueeze(0) # [1, 3, H, W]

content_path = "лодочка.jpg"
style_path = "prud-s-kuvshinkami-mone+.jpg"

content_img = load_image(content_path).to(device)
style_img = load_image(style_path).to(device)


def denormalize(tensor):
    """Обратная нормализация ImageNet для отображения изображения."""
    mean = torch.tensor(imagenet_mean).view(3, 1, 1)
    std = torch.tensor(imagenet_std).view(3, 1, 1)
    img = tensor.clone().detach().squeeze(0).cpu() 
    img = img * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()

def get_features(image, model, layers):
    # Инвертируем словарь один раз, чтобы не делать поиск на каждом слое
    idx_to_name = {v: k for k, v in layers.items()}
    max_idx = max(idx_to_name.keys())
    features = {}
    x = image
    for i, layer in enumerate(model):
        x = layer(x)
        if i in idx_to_name:
            features[idx_to_name[i]] = x
        if i == max_idx:
            break # все нужные слои извлечены — выходим раньше
    return features

style_layers = {
    'conv1_1': 0,
    'conv2_1': 5,
    'conv3_1': 10,
    'conv4_1': 19,
    'conv5_1': 28,
    }

content_layers = {
    'conv4_2': 21,
    }

all_layers = {**style_layers, **content_layers}

def content_loss(generated_features, content_features):
    return F.mse_loss(generated_features, content_features)

def gram_matrix(features):
    b, c, h, w = features.shape
    F_ = features.view(b, c, h * w) # [batch, C, H*W]
    G = torch.bmm(F_, F_.transpose(1, 2)) # [batch, C, C]
    return G / (c * h * w) # нормализация по числу элементов

def style_loss(generated_features, style_features):
    G_gen = gram_matrix(generated_features)
    G_style = gram_matrix(style_features)
    return F.mse_loss(G_gen, G_style)

generated = content_img.clone().requires_grad_(True)

with torch.no_grad():
    content_features = get_features(content_img, vgg, all_layers)
    style_features = get_features(style_img, vgg, all_layers)

optimizer = torch.optim.LBFGS([generated], lr=1.0, max_iter=20)

num_steps = 300
alpha = 1
beta = 1e5
# Веса для разных уровней стиля
style_weights = {
    'conv1_1': 1.0,
    'conv2_1': 0.8,
    'conv3_1': 0.5,
    'conv4_1': 0.3,
    'conv5_1': 0.1,
    }

# Для корректного зажима в нормализованном пространстве
mean = torch.tensor(imagenet_mean, device=device).view(1, 3, 1, 1)
std = torch.tensor(imagenet_std, device=device).view(1, 3, 1, 1)

# def run_nst(beta_value, num_steps=300, save_interval=50):
#     """Запускает NST с заданным значением beta и возвращает финальное изображение и историю."""
#     print(f"\n--- Запуск с beta = {beta_value} ---")
    
#     generated = content_img.clone().requires_grad_(True)
    
#     with torch.no_grad():
#         content_features = get_features(content_img, vgg, all_layers)
#         style_features = get_features(style_img, vgg, all_layers)
    
#     optimizer = torch.optim.LBFGS([generated], lr=1.0, max_iter=20)
#     alpha = 1
    
#     saved_images = []
    
#     def closure():
#         optimizer.zero_grad()
#         gen_features = get_features(generated, vgg, all_layers)
#         c_loss = content_loss(gen_features['conv4_2'], content_features['conv4_2'])
        
#         s_loss = 0
#         for layer in style_layers:
#             weight = style_weights[layer]
#             s_loss += weight * style_loss(gen_features[layer], style_features[layer])
        
#         total_loss = alpha * c_loss + beta_value * s_loss
#         total_loss.backward()
        
#         closure.last_losses = (total_loss, c_loss, s_loss)
#         return total_loss
    
#     for step in range(num_steps):
#         optimizer.step(closure)
        
#         with torch.no_grad():
#             generated.data = (generated.data * std + mean).clamp(0, 1)
            
#             if step % save_interval == 0:
#                 temp_np = generated.data.clone().detach().cpu().squeeze(0).permute(1,2,0).clamp(0,1).numpy()
#                 saved_images.append(temp_np)
            
#             generated.data = (generated.data - mean) / std
        
#         if step % save_interval == 0:
#             total_loss, c_loss, s_loss = closure.last_losses
#             print(f"  Step {step}: total={total_loss.item():.4f}, content={c_loss.item():.4f}, style={s_loss.item():.8f}")
    
#     final_np = denormalize(generated)
#     return final_np, saved_images

# # Исследуемые значения beta
# beta_values = [1e3, 1e5, 1e7]
# results = {}

# for beta in beta_values:
#     final_img, history = run_nst(beta, num_steps=300, save_interval=100)
#     results[beta] = {
#         'final': final_img,
#         'history': history
#     }

# # ВИЗУАЛИЗАЦИЯ
# fig, axes = plt.subplots(2, len(beta_values) + 1, figsize=(4 * (len(beta_values) + 1), 8))

# # Показываем исходные изображения в первой колонке
# content_np = denormalize(content_img)
# style_np = denormalize(style_img)

# axes[0, 0].imshow(content_np)
# axes[0, 0].set_title(f"Content\n{content_path}", fontsize=10)
# axes[0, 0].axis('off')

# axes[1, 0].imshow(style_np)
# axes[1, 0].set_title(f"Style\n{style_path}", fontsize=10)
# axes[1, 0].axis('off')

# # Показываем результаты для разных beta
# for idx, beta in enumerate(beta_values):
#     # Верхний ряд: промежуточный результат (например, шаг 100)
#     if len(results[beta]['history']) > 1:
#         axes[0, idx + 1].imshow(results[beta]['history'][1])  # шаг 100
#         axes[0, idx + 1].set_title(f"β={beta:.0e}\n(step 100)", fontsize=10)
#     else:
#         axes[0, idx + 1].imshow(results[beta]['final'])
#         axes[0, idx + 1].set_title(f"β={beta:.0e}", fontsize=10)
#     axes[0, idx + 1].axis('off')
    
#     # Нижний ряд: финальный результат
#     axes[1, idx + 1].imshow(results[beta]['final'])
#     axes[1, idx + 1].set_title(f"β={beta:.0e}\n(final)", fontsize=10)
#     axes[1, idx + 1].axis('off')

# plt.suptitle(f"Влияние баланса α/β (α=1) | Content: {content_path} | Style: {style_path}", 
#              fontsize=14, y=0.98)
# plt.tight_layout()
# plt.savefig("beta_comparison.png", dpi=150, bbox_inches='tight')
# plt.show()


def run_nst_with_init(content_img, style_img, vgg, beta_value, init_type='content', 
                      num_steps=300, save_interval=50):
    """
    Запускает NST с разными типами инициализации.
    init_type: 'content' (копия контента) или 'noise' (случайный шум)
    """
    print(f"\n--- Запуск с beta = {beta_value}, инициализация = {init_type} ---")
    
    # Инициализация
    if init_type == 'content':
        generated = content_img.clone().requires_grad_(True)
        print("  Инициализация: копия контентного изображения")
    elif init_type == 'noise':
        generated = torch.randn_like(content_img).requires_grad_(True)
        # Нормализуем шум, чтобы он был в том же диапазоне, что и нормализованные изображения
        with torch.no_grad():
            generated.data = (generated.data - generated.data.mean()) / generated.data.std()
        print("  Инициализация: случайный шум")
    else:
        raise ValueError("init_type должен быть 'content' или 'noise'")
    
    with torch.no_grad():
        content_features = get_features(content_img, vgg, all_layers)
        style_features = get_features(style_img, vgg, all_layers)
    
    optimizer = torch.optim.LBFGS([generated], lr=1.0, max_iter=20)
    alpha = 1
    
    # Для корректного зажима в нормализованном пространстве
    mean = torch.tensor(imagenet_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(imagenet_std, device=device).view(1, 3, 1, 1)
    
    saved_images = []
    
    def closure():
        optimizer.zero_grad()
        gen_features = get_features(generated, vgg, all_layers)
        c_loss = content_loss(gen_features['conv4_2'], content_features['conv4_2'])
        
        s_loss = 0
        for layer in style_layers:
            weight = style_weights[layer]
            s_loss += weight * style_loss(gen_features[layer], style_features[layer])
        
        total_loss = alpha * c_loss + beta_value * s_loss
        total_loss.backward()
        
        closure.last_losses = (total_loss, c_loss, s_loss)
        return total_loss
    
    for step in range(num_steps):
        optimizer.step(closure)
        
        with torch.no_grad():
            generated.data = (generated.data * std + mean).clamp(0, 1)
            
            if step % save_interval == 0:
                temp_np = generated.data.clone().detach().cpu().squeeze(0).permute(1,2,0).clamp(0,1).numpy()
                saved_images.append(temp_np)
            
            generated.data = (generated.data - mean) / std
        
        if step % save_interval == 0:
            total_loss, c_loss, s_loss = closure.last_losses
            print(f"  Step {step}: total={total_loss.item():.4f}, content={c_loss.item():.4f}, style={s_loss.item():.8f}")
    
    final_np = denormalize(generated)
    return final_np, saved_images


# ============ ИССЛЕДОВАНИЕ ВЛИЯНИЯ ИНИЦИАЛИЗАЦИИ ============

def compare_initializations():
    """Сравнивает два типа инициализации для выбранного beta."""
    
    # Пути к изображениям
    content_path = "лодочка.jpg"
    style_path = "prud-s-kuvshinkami-mone+.jpg"
    
    # Загрузка изображений
    print("Загрузка изображений...")
    content_img = load_image(content_path).to(device)
    style_img = load_image(style_path).to(device)
    
    beta_value = 1e5
    
    # Запускаем с разной инициализацией
    init_types = ['content', 'noise']
    results = {}
    
    for init_type in init_types:
        final_img, history = run_nst_with_init(
            content_img, style_img, vgg, beta_value, 
            init_type=init_type, num_steps=300, save_interval=100
        )
        results[init_type] = {
            'final': final_img,
            'history': history
        }
    
    # ВИЗУАЛИЗАЦИЯ СРАВНЕНИЯ
    fig, axes = plt.subplots(3, len(init_types) + 1, figsize=(5 * (len(init_types) + 1), 12))
    
    # Первая колонка - исходные изображения
    content_np = denormalize(content_img)
    style_np = denormalize(style_img)
    
    axes[0, 0].imshow(content_np)
    axes[0, 0].set_title(f"Content\n{content_path}", fontsize=10)
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(style_np)
    axes[1, 0].set_title(f"Style\n{style_path}", fontsize=10)
    axes[1, 0].axis('off')
    
    axes[2, 0].axis('off')  # пустая ячейка
    
    # Заголовки для колонок с инициализациями
    for idx, init_type in enumerate(init_types):
        init_name = "Копия контента" if init_type == 'content' else "Случайный шум"
        axes[0, idx + 1].set_title(f"{init_name}\n(начальное состояние)", fontsize=10)
        axes[1, idx + 1].set_title(f"{init_name}\n(шаг 100)", fontsize=10)
        axes[2, idx + 1].set_title(f"{init_name}\n(финал)", fontsize=10)
    
    # Заполняем результаты
    for idx, init_type in enumerate(init_types):
        # Начальное состояние
        if init_type == 'content':
            init_np = content_np
        else:
            # Визуализируем шум (первое сохранённое изображение или генерируем)
            if len(results[init_type]['history']) > 0:
                init_np = results[init_type]['history'][0]
            else:
                # Создаём визуализацию шума
                noise_vis = torch.randn_like(content_img).cpu().squeeze(0)
                noise_vis = (noise_vis - noise_vis.min()) / (noise_vis.max() - noise_vis.min())
                init_np = noise_vis.permute(1,2,0).clamp(0,1).numpy()
        axes[0, idx + 1].imshow(init_np)
        axes[0, idx + 1].axis('off')
        
        # Шаг 100 (промежуточный)
        if len(results[init_type]['history']) > 1:
            axes[1, idx + 1].imshow(results[init_type]['history'][1])
        axes[1, idx + 1].axis('off')
        
        # Финальный результат
        axes[2, idx + 1].imshow(results[init_type]['final'])
        axes[2, idx + 1].axis('off')
    
    plt.suptitle(f"Сравнение инициализаций (β = {beta_value:.0e}, α = 1)\nContent: {content_path} | Style: {style_path}", 
                 fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig("init_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()

    
    print("\nРезультаты сохранены в:")
    print("  - init_comparison.png (сравнение инициализаций)")


# Запуск сравнения
if __name__ == "__main__":
    compare_initializations()




# список для хранения промежуточных изображений
# saved_images = []

# def closure():
#     optimizer.zero_grad()
#     gen_features = get_features(generated, vgg, all_layers)
#     c_loss = content_loss(
#         gen_features['conv4_2'],
#         content_features['conv4_2']
#         )
    
#     s_loss = 0
#     for layer in style_layers:
#         weight = style_weights[layer]
#         s_loss += weight * style_loss(
#             gen_features[layer],
#             style_features[layer]
#             )
        
#     total_loss = alpha * c_loss + beta * s_loss
#     total_loss.backward()

#     # Сохраняем для логирования
#     closure.last_losses = (total_loss, c_loss, s_loss)
#     return total_loss

# for step in range(num_steps):
#     optimizer.step(closure)

#     with torch.no_grad():
#         # Временно денормализуем для зажима в [0,1]
#         generated.data = (generated.data * std + mean).clamp(0, 1)
        
#         # сохраняем промежуточный результат, если шаг кратен 50 
#         if step % 50 == 0:
#             # generated.data сейчас в [0,1] (денормализованное)
#             temp_np = generated.data.clone().detach().cpu().squeeze(0).permute(1,2,0).clamp(0,1).numpy()
#             saved_images.append(temp_np)
        
#         # Возвращаем в нормализованное пространство для следующей итерации
#         generated.data = (generated.data - mean) / std

#     if step % 50 == 0:
#         total_loss, c_loss, s_loss = closure.last_losses
#         print(f"Step {step}: total={total_loss.item():.4f}, content={c_loss.item():.4f}, style={s_loss.item():.8f}")

# # Сохраняем финальное изображение
# final_np = denormalize(generated)
# saved_images.append(final_np)

# # ВИЗУАЛИЗАЦИЯ
# num_images = len(saved_images)
# plt.figure(figsize=(3 * num_images, 3.5))

# # Заголовок над всей фигурой
# plt.suptitle(f"Content: {content_path}\nStyle: {style_path}", fontsize=12, y=0.98)

# for idx, img_np in enumerate(saved_images):
#     plt.subplot(1, num_images, idx+1)
#     plt.imshow(img_np)
#     plt.title(f"Шаг {idx * 50}")
#     plt.axis('off')
# plt.tight_layout()
# plt.savefig("evolution_2.png", dpi=150, bbox_inches='tight')
# plt.show()