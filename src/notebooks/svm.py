# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# %%
years = range(2018, 2024)
dfs = []

base_path = "../../data/clean/complex/final/player_data_{}.csv"
for year in years:
    df = pd.read_csv(base_path.format(year))
    df["Year"] = year
    df["Score"] = 101 - df["Rank"]  # Higher score = better player
    dfs.append(df)

train_df = pd.concat(dfs, ignore_index=True)
print(f"Combined training shape: {train_df.shape}")
train_df.head()


# %%
# Load 2024 test data
test_df = pd.read_csv("../../data/clean/complex/final/player_data_2024.csv")
test_df["Year"] = 2024
test_df["Score"] = 0  # Placeholder


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
y_train = train_df["Score"].values
X_test = scaler.transform(test_df[shared_cols])


# %%
# 6
from sklearn.feature_selection import SelectKBest, f_regression

# Manually ensure inclusion of important stat families (base, b_, m_ versions)
always_include_keywords = [
    "rating",
    "dpr",
    "kast",
    "impact",
    "adr",
    "kpr",
    "vs_top_5",
    "vs_top_10",
    "vs_top_20",
]

# Create whitelist of columns matching keywords
whitelist_features = [
    col
    for col in shared_cols
    if any(
        kw in col.lower() and col.lower().startswith(prefix)
        for kw in always_include_keywords
        for prefix in ["", "b_", "m_"]
    )
]

print(f"Manually included important features: {len(whitelist_features)}")

# Run SelectKBest
kbest = SelectKBest(score_func=f_regression, k=40)
X_train_kbest = kbest.fit_transform(X_train, y_train)
selected_kbest_cols = [shared_cols[i] for i in kbest.get_support(indices=True)]

# Combine k-best + always-include
final_selected_features = list(set(selected_kbest_cols + whitelist_features))

print(
    f"\nFinal selected features (after combining KBest + manual whitelist): {len(final_selected_features)}"
)

# Re-transform X_train and X_test to use only selected features
X_train_selected = pd.DataFrame(X_train, columns=shared_cols)[
    final_selected_features
].values
X_test_selected = pd.DataFrame(X_test, columns=shared_cols)[
    final_selected_features
].values


# %%
from sklearn.svm import LinearSVR
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {"C": [0.01, 0.1, 1, 10, 100], "epsilon": [0.01, 0.1, 0.5]}

# Perform 5-fold CV
svm_model = LinearSVR(max_iter=10000)
grid = GridSearchCV(
    svm_model, param_grid, cv=5, scoring="neg_mean_squared_error", verbose=1
)
grid.fit(X_train_selected, y_train)

print("Best Params:", grid.best_params_)
print("Best CV Score (MSE):", -grid.best_score_)


# %% [markdown]
# Training Loop

# %%
# Train best model on all training data
best_svm = grid.best_estimator_
best_svm.fit(X_train_selected, y_train)


# %%
preds_2024 = best_svm.predict(X_test_selected)
test_df["PredictedScore"] = preds_2024

top_20_2024 = test_df.sort_values(by="PredictedScore", ascending=False).head(20)
top_20_2024[["Player", "PredictedScore"]]


# %%
# Load actual HLTV 2024 rankings
actual_df = pd.read_csv("../../rankings/ranking_2024.csv")


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
