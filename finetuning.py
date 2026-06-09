import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, \
    get_linear_schedule_with_warmup
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from itertools import combinations
from tqdm import tqdm

MODEL_NAME = 'DeepPavlov/rubert-base-cased'

# ============================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================

df = pd.read_csv('courses_labeled.csv', sep=';')
print(f"Всего курсов: {len(df)}")
print(f"Кластеров:    {df['cluster'].nunique()}")
print(f"Распределение:\n{df['cluster'].value_counts()}")

df_train, df_val = train_test_split(
    df, test_size=0.2, stratify=df['cluster'], random_state=42
)
print(f"\nTrain: {len(df_train)}, Val: {len(df_val)}")


# ============================================================
# 2. МОДЕЛЬ С MEAN POOLING
# ============================================================

class BertEmbedder(nn.Module):
    """BERT + Mean Pooling → единый вектор для текста"""

    def __init__(self, model_name=MODEL_NAME):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size  # 312

    def mean_pooling(self, token_embeddings, attention_mask):
        """
        Усредняет эмбеддинги токенов, игнорируя [PAD]

        token_embeddings: (batch, seq_len, hidden_size)
        attention_mask:    (batch, seq_len)
        return:            (batch, hidden_size)
        """
        mask = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counted = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counted

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        embeddings = self.mean_pooling(
            outputs.last_hidden_state,
            attention_mask
        )
        # L2-нормализация
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def encode(self, texts, tokenizer, batch_size=32, device='cpu'):
        """Кодирование списка текстов (для оценки)"""
        self.eval()
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(device)

            with torch.no_grad():
                emb = self.forward(
                    encoded['input_ids'],
                    encoded['attention_mask']
                )
            all_embeddings.append(emb.cpu().numpy())

        return np.vstack(all_embeddings)


# ============================================================
# 3. TRIPLET LOSS
# ============================================================

class TripletLoss(nn.Module):
    """
    L = max(0, d(anchor, positive) - d(anchor, negative) + margin)

    Цель: положительный пример ближе к якорю,
          чем отрицательный, с запасом margin
    """

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        anchor:   (batch, hidden_size)
        positive: (batch, hidden_size)
        negative: (batch, hidden_size)
        """
        # Косинусные расстояния
        d_pos = 1 - F.cosine_similarity(anchor, positive)  # маленькое
        d_neg = 1 - F.cosine_similarity(anchor, negative)  # большое

        losses = F.relu(d_pos - d_neg + self.margin)
        return losses.mean()


class MultipleNegativesRankingLoss(nn.Module):
    """
    L = - 1/B * Σ_i log[ exp(sim(a_i, p_i)/τ) / Σ_j exp(sim(a_i, p_j)/τ) ]

    В качестве негативов используются все положительные примеры
    других пар в батче (in-batch negatives).
    Симметричная версия: уредняется по a→p и p→a.

    Цель: положительный пример p_i должен быть ближе к якорю a_i,
          чем все остальные p_j, j≠i, с учётом температуры τ.
    """

    def __init__(self, temperature=0.05):
        """
        Args:
            temperature: коэффициент температуры для Softmax.
                         Чем ниже, тем жёстче разделение (обычно 0.05-0.1).
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negative=None):
        """
        anchor:   (batch, hidden_size)
        positive: (batch, hidden_size)
        negative: игнорируется – оставлен для бесшовной замены TripletLoss.

        Returns:
            scalar loss
        """
        # 1. Нормализация для получения косинусного сходства
        anchor   = F.normalize(anchor,   p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)

        # 2. Матрица косинусных сходств anchor (B) x positive (B)
        #    sim[i][j] = cos(anchor_i, positive_j)
        sim = torch.matmul(anchor, positive.T)  # (B, B)

        # 3. Применяем температуру
        scores = sim / self.temperature

        # 4. Правильные метки: для i-го якоря правильный ответ – i-й позитив
        batch_size = scores.size(0)
        labels = torch.arange(batch_size, device=scores.device)

        # 5. Кросс-энтропия в обе стороны (симметричный MNR)
        loss_a = F.cross_entropy(scores, labels)      # anchor → positive
        loss_b = F.cross_entropy(scores.T, labels)    # positive → anchor

        return (loss_a + loss_b) / 2


# ============================================================
# 4. DATASET ДЛЯ ТРИПЛЕТОВ
# ============================================================

class TripletDataset(Dataset):
    """
    Генерирует тройки (anchor, positive, negative)
    из таблицы (description, cluster)
    """

    def __init__(self, df, tokenizer, max_length=256,
                 max_triplets_per_anchor=10):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.triplets = self._create_triplets(df, max_triplets_per_anchor)
        print(f"  Создано триплетов: {len(self.triplets)}")

    def _create_triplets(self, df, max_per_anchor):
        cluster_texts = df.groupby('cluster')['description'].apply(
            list).to_dict()
        clusters = list(cluster_texts.keys())

        triplets = []
        for cluster_id in clusters:
            texts = cluster_texts[cluster_id]

            other_texts = []
            for other_id in clusters:
                if other_id != cluster_id:
                    other_texts.extend(cluster_texts[other_id])

            if len(texts) < 2 or len(other_texts) == 0:
                continue

            pos_pairs = list(combinations(texts, 2))
            if len(pos_pairs) > max_per_anchor * len(texts):
                pos_pairs = random.sample(
                    pos_pairs,
                    max_per_anchor * len(texts)
                )

            for text_a, text_b in pos_pairs:
                negative = random.choice(other_texts)
                triplets.append((text_a, text_b, negative))

        random.shuffle(triplets)
        return triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor, positive, negative = self.triplets[idx]
        return {
            'anchor': anchor,
            'positive': positive,
            'negative': negative
        }

    def collate_fn(self, batch):
        """Токенизация внутри collate для батчевого паддинга"""
        anchors = [b['anchor'] for b in batch]
        positives = [b['positive'] for b in batch]
        negatives = [b['negative'] for b in batch]

        enc_anchor = self.tokenizer(
            anchors, padding=True, truncation=True,
            max_length=self.max_length, return_tensors='pt'
        )
        enc_positive = self.tokenizer(
            positives, padding=True, truncation=True,
            max_length=self.max_length, return_tensors='pt'
        )
        enc_negative = self.tokenizer(
            negatives, padding=True, truncation=True,
            max_length=self.max_length, return_tensors='pt'
        )

        return enc_anchor, enc_positive, enc_negative


# ============================================================
# 5. ОЦЕНКА КАЧЕСТВА
# ============================================================

def evaluate(model, tokenizer, df_eval, n_clusters, device):
    """Кодирует → кластеризует → считает метрики"""

    descriptions = df_eval['description'].tolist()
    true_labels = df_eval['cluster'].tolist()

    embeddings = model.encode(descriptions, tokenizer, device=device)

    pred_labels = KMeans(
        n_clusters=n_clusters, random_state=42, n_init=10
    ).fit_predict(embeddings)

    sil = silhouette_score(embeddings, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)

    return sil, ari


# ============================================================
# 6. ЦИКЛ ОБУЧЕНИЯ
# ============================================================

def train(
        model_name=MODEL_NAME,
        num_epochs=10,
        batch_size=16,
        learning_rate=2e-5,
        margin=0.3,
        max_length=256,
        max_triplets_per_anchor=10,
        save_path='./finetuned-bert',
        weight_decay=0.01,
):
    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Модель и токенизатор
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertEmbedder(model_name).to(device)
    #criterion = TripletLoss(margin=margin)
    criterion = MultipleNegativesRankingLoss()

    # Данные
    print("\nСоздание обучающего датасета...")
    train_dataset = TripletDataset(
        df_train, tokenizer, max_length, max_triplets_per_anchor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )

    # Оптимизатор и планировщик
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * 0.1)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    n_clusters = df['cluster'].nunique()

    # Метрики до обучения
    print("\n=== Метрики ДО обучения ===")
    sil_before, ari_before = evaluate(
        model, tokenizer, df, n_clusters, device
    )
    print(f"  Silhouette: {sil_before:.4f}")
    print(f"  ARI:        {ari_before:.4f}")

    # Обучение
    best_sil = -1
    history = []

    print(f"\n=== Обучение: {num_epochs} эпох ===")
    print(f"  Батчей за эпоху: {len(train_loader)}")
    print(f"  Всего шагов:     {total_steps}")
    print(f"  Warmup:           {warmup_steps}")

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for enc_anchor, enc_positive, enc_negative in pbar:
            # Переносим на устройство
            enc_anchor = {k: v.to(device) for k, v in enc_anchor.items()}
            enc_positive = {k: v.to(device) for k, v in enc_positive.items()}
            enc_negative = {k: v.to(device) for k, v in enc_negative.items()}

            # Forward
            emb_anchor = model(
                enc_anchor['input_ids'],
                enc_anchor['attention_mask']
            )
            emb_positive = model(
                enc_positive['input_ids'],
                enc_positive['attention_mask']
            )
            emb_negative = model(
                enc_negative['input_ids'],
                enc_negative['attention_mask']
            )

            # Loss
            loss = criterion(emb_anchor, emb_positive, emb_negative)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Оценка после каждой эпохи
        avg_loss = np.mean(epoch_losses)
        sil_val, ari_val = evaluate(
            model, tokenizer, df_val, n_clusters, device
        )
        sil_all, ari_all = evaluate(
            model, tokenizer, df, n_clusters, device
        )

        history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'sil_val': sil_val,
            'ari_val': ari_val,
            'sil_all': sil_all,
            'ari_all': ari_all,
        })

        is_best = sil_val > best_sil
        if is_best:
            best_sil = sil_val
            # Сохраняем лучшую модель
            model.bert.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

        print(
            f"  Epoch {epoch + 1}: loss={avg_loss:.4f} | "
            f"val_sil={sil_val:.4f} val_ari={ari_val:.4f} | "
            f"all_sil={sil_all:.4f} all_ari={ari_all:.4f}"
            f"{'  ★ best' if is_best else ''}"
        )

    return model, tokenizer, history


# ============================================================
# 7. ЗАПУСК

model, tokenizer, history = train(
    model_name=MODEL_NAME,
    num_epochs=12,
    batch_size=16,
    learning_rate=2e-5,
    #margin=0.3,
    margin=0.5,
    max_length=256,
    save_path='./finetuned-bert-triplet'
)

# ============================================================
# 8. ЗАГРУЗКА И ИСПОЛЬЗОВАНИЕ ЛУЧШЕЙ МОДЕЛИ
# ============================================================

# Загрузка
best_tokenizer = AutoTokenizer.from_pretrained('./finetuned-bert-triplet')
best_model = BertEmbedder('./finetuned-bert-triplet')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_model = best_model.to(device)

# # Финальные эмбеддинги
all_descriptions = df['description'].tolist()
embeddings_final = best_model.encode(all_descriptions, best_tokenizer,
                                     device=device)

# Финальная кластеризация
n_clusters = df['cluster'].nunique()
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
pred_labels = kmeans.fit_predict(embeddings_final)

final_sil = silhouette_score(embeddings_final, pred_labels)
final_ari = adjusted_rand_score(df['cluster'].tolist(), pred_labels)

print(f"\n{'=' * 50}")
print(f"Финальные метрики на лучшей модели:")
print(f"  Silhouette: {final_sil:.4f}")
print(f"  ARI:        {final_ari:.4f}")
print(f"{'=' * 50}")

# ============================================================
# 9. ВИЗУАЛИЗАЦИЯ ОБУЧЕНИЯ
# ============================================================

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

epochs = [h['epoch'] for h in history]

# Loss
axes[0].plot(epochs, [h['loss'] for h in history], 'b-o')
axes[0].set_title('Training Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Triplet Loss')
axes[0].grid(True, alpha=0.3)

# Silhouette
axes[1].plot(epochs, [h['sil_val'] for h in history], 'g-o', label='Validation')
axes[1].plot(epochs, [h['sil_all'] for h in history], 'b-s', label='All data')
axes[1].set_title('Silhouette Score')
axes[1].set_xlabel('Epoch')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# ARI
axes[2].plot(epochs, [h['ari_val'] for h in history], 'r-o', label='Validation')
axes[2].plot(epochs, [h['ari_all'] for h in history], 'm-s', label='All data')
axes[2].set_title('Adjusted Rand Index')
axes[2].set_xlabel('Epoch')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
plt.show()

# ============================================================
# 10. ВИЗУАЛИЗАЦИЯ КЛАСТЕРОВ: ДО И ПОСЛЕ
# ============================================================

from sklearn.manifold import TSNE

# Эмбеддинги до обучения
base_model = BertEmbedder(MODEL_NAME).to(device)
base_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
emb_before = base_model.encode(all_descriptions, base_tokenizer, device=device)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, emb, title in [
    (axes[0], emb_before, "До fine-tuning"),
    (axes[1], embeddings_final, "После fine-tuning"),
]:
    tsne = TSNE(n_components=2, random_state=42,
                perplexity=min(30, len(emb) - 1))
    emb_2d = tsne.fit_transform(emb)

    scatter = ax.scatter(
        emb_2d[:, 0], emb_2d[:, 1],
        c=df['cluster'].tolist(),
        cmap='tab10', s=60, alpha=0.7,
        edgecolors='black', linewidths=0.5
    )
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

plt.colorbar(scatter, ax=axes, label='Кластер')
plt.tight_layout()
plt.savefig('clusters_before_after.png', dpi=150)
plt.show()

# ============================================================
# 11. ПРЕДСКАЗАНИЕ ДЛЯ НОВЫХ КУРСОВ
# ============================================================

import pickle

# Сохраняем KMeans
pickle.dump(kmeans, open('kmeans_model.pkl', 'wb'))


def predict_cluster(text, model, tokenizer, kmeans_model, device):
    embedding = model.encode([text], tokenizer, device=device)
    cluster = kmeans_model.predict(normalize(embedding))[0]

    # Расстояния до всех центроидов
    distances = kmeans_model.transform(normalize(embedding))[0]
    confidence = 1 / (1 + distances[cluster])

    return cluster


# with open('description.txt', 'r', encoding='utf-8') as f:
#     new_courses = [x.replace('\n', '') for x in f.readlines()]

dataframe = pd.read_csv('dataframe.csv', sep=";")
dataframe['cluster'] = dataframe['full'].apply(lambda row:
                               predict_cluster(row, best_model, best_tokenizer, kmeans, device))
dataframe.to_csv("clustered_courses.csv", encoding="utf-8")

# cluster_texts = {0: [], 1: [], 2: [], 3: [], 4: [],
#                  5: [], 6: [], 7: [], 8: [], 9: []}

# for course in dataframe:
#     cluster, confidence = predict_cluster(
#         course, best_model, best_tokenizer, kmeans, device
#     )
#     cluster_texts[cluster].append(course)
    #print(f"  '{course[:50]}...' → Кластер {cluster} ({confidence:.2%})")
#TODO: записать датафрейм в файл


# with open('kmeans_bert_clusters.txt', 'w', encoding='utf-8') as f:
#     f.write("="*80 + "\n\n")
#     for cluster_id in range(n_clusters):
#         f.write(f"--- Кластер {cluster_id} ---\n")
#         for idx, text in enumerate(cluster_texts[cluster_id]):
#             f.write(f"[{idx}] {text}\n")
#         f.write("\n")
# print("\nКластеры сохранены в 'kmeans_bert_clusters.txt'")
