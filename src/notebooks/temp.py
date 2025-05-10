# %%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# %%
years = range(2018, 2024)
dfs = []

base_path = (
    "../../data/clean/complex/final/player_data_{}.csv"  # Update with actual path
)
for year in years:
    df = pd.read_csv(base_path.format(year))
    df["Year"] = year
    df["Score"] = 101 - df["Rank"]  # convert rank to score
    dfs.append(df)

train_df = pd.concat(dfs, ignore_index=True)

train_df.head()


# %%
# Load 2024 test data
test_df = pd.read_csv("../../data/clean/complex/final/player_data_2024.csv")
test_df["Year"] = 2024
test_df["Score"] = 0  # Placeholder
test_df.head()


# %%
# Handle different rating column versions
def unify_rating_columns(df):
    # Priority: use rating_2.1 if available, else rating_2.0
    if "rating_2.1" in df.columns:
        df["rating"] = df["rating_2.1"]
    elif "rating_2.0" in df.columns:
        df["rating"] = df["rating_2.0"]
    elif "rating_1.0" in df.columns:
        df["rating"] = df["rating_1.0"]
    else:
        df["rating"] = np.nan  # fallback

    # Drop older versions if present
    for col in ["rating_2.1", "rating_2.0", "rating_1.0"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    return df


# Apply to both training and test sets
train_df = unify_rating_columns(train_df)
test_df = unify_rating_columns(test_df)


# %%
# Preprocess
id_cols = ["Player", "HLTV_ID", "Rank", "Year"]
feature_cols = [col for col in train_df.columns if col not in id_cols + ["Score"]]
shared_cols = list(set(feature_cols) & set(test_df.columns))

train_df[shared_cols] = train_df[shared_cols].replace(-1.0, np.nan)
train_df[shared_cols] = train_df[shared_cols].fillna(train_df[shared_cols].mean())

test_df[shared_cols] = test_df[shared_cols].replace(-1.0, np.nan)
test_df[shared_cols] = test_df[shared_cols].fillna(train_df[shared_cols].mean())

scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[shared_cols])
X_test = scaler.transform(test_df[shared_cols])
y_train = train_df["Score"].values


# %%
class DeepRankNet(nn.Module):
    def __init__(self, input_dim):
        super(DeepRankNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


# %% [markdown]
# Training Loop

# %%
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

train_ds = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

model = DeepRankNet(input_dim=X_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Train loop
for epoch in range(10000):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {total_loss / len(train_loader):.4f}")


# %%
model.eval()
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    preds_2024 = model(X_test_tensor).numpy()

test_df["PredictedScore"] = preds_2024
top_20_2024 = test_df.sort_values(by="PredictedScore", ascending=False).head(20)
top_20_2024.reset_index(inplace=True)

print(top_20_2024[["Player", "PredictedScore"]])


# %%
# Load actual HLTV 2024 rankings
actual_df = pd.read_csv("../../rankings/ranking_2024.csv")  # Update path


# Normalize nicknames in both DataFrames (for easier comparison)
def normalize(name):
    return name.strip().lower().replace("⁠", "").replace("’", "'").replace("`", "'")


actual_df["Nickname"] = actual_df["Nickname"].apply(normalize)
top_20_2024["Player"] = top_20_2024["Player"].apply(normalize)

# Map: nickname -> actual rank
actual_ranks = {row["Nickname"]: row["Rank"] for _, row in actual_df.iterrows()}


# Evaluation function
def score_ranking(pred_df, actual_rank_dict):
    score = 0
    graded = []

    for pred_rank, row in enumerate(pred_df["Player"].values, 1):
        actual_rank = actual_rank_dict.get(row)

        if actual_rank:
            diff = abs(actual_rank - pred_rank)
            if diff == 0:
                pts = 5
            elif diff == 1:
                pts = 4
            elif diff == 2:
                pts = 3
            elif diff == 3:
                pts = 2
            elif diff <= 5:
                pts = 1
            else:
                pts = 0
        else:
            pts = 0

        graded.append((pred_rank, row, actual_rank, pts))
        score += pts

    return score, graded


# %%
# Run scoring
total_score, breakdown = score_ranking(top_20_2024, actual_ranks)

# Display summary
print(f"🏆 Total Ranking Score: {total_score}/100\n")
print("🔍 Breakdown:")
for pred_rank, nickname, actual_rank, pts in breakdown:
    print(
        f"Predicted #{pred_rank:>2}: {nickname:<15} | Actual: {actual_rank if actual_rank else 'N/A':<2} | +{pts} pts"
    )
