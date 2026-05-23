"""
=============================================================================
  WORKOUT RECOMMENDER — RESEARCH-READY ML PIPELINE
  Target: Predict workout type (strength / hiit / flexibility)
  Dataset: scraped_style_noisy_workout_dataset
=============================================================================
"""

# ── 0. Imports ────────────────────────────────────────────────────────────
import warnings
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection   import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing     import LabelEncoder, StandardScaler
from sklearn.impute             import SimpleImputer
from sklearn.pipeline           import Pipeline
from sklearn.metrics            import (classification_report, confusion_matrix,
                                        accuracy_score, f1_score,
                                        precision_score, recall_score)
from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors          import KNeighborsClassifier
from xgboost                    import XGBClassifier
from scipy                      import stats

warnings.filterwarnings("ignore")
np.random.seed(42)

OUTDIR = "/mnt/user-data/outputs"

# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  STEP 1 — LOADING DATA")
print("=" * 65)

df_raw = pd.read_csv("/mnt/user-data/uploads/scraped_style_noisy_workout_dataset__1_.csv")
print(f"Raw shape: {df_raw.shape}")
print(f"Columns  : {df_raw.columns.tolist()}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. DATA CLEANING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  STEP 2 — DATA CLEANING")
print("=" * 65)

df = df_raw.copy()

# ── 2a. Drop columns with >95 % missing or near-zero variance ────────────
# sleep_hours: 10,670 / 10,800 missing → drop
df.drop(columns=["sleep_hours", "row_id", "timestamp", "data_source",
                  "workout_name", "muscle_groups", "equipment_needed"],
        inplace=True)
print("Dropped low-signal / ID columns")

# ── 2b. Clean TARGET: target_workout_type ────────────────────────────────
# Valid classes: strength | hiit | flexibility
valid_types = {"strength", "hiit", "flexibility"}

def clean_target(val):
    if pd.isna(val):       return np.nan
    v = str(val).strip().lower()
    if v in valid_types:   return v
    if "strength" in v:    return "strength"
    if "hiit"     in v:    return "hiit"
    if "flex"     in v:    return "flexibility"
    return np.nan           # garbage: missing, ???, CARDIOOOO, mixed_type

df["target_workout_type"] = df["target_workout_type"].apply(clean_target)
before = len(df)
df.dropna(subset=["target_workout_type"], inplace=True)
print(f"Removed {before - len(df)} rows with unresolvable target labels")
print(f"Target distribution:\n{df['target_workout_type'].value_counts()}")

# ── 2c. Clean fitness_level ──────────────────────────────────────────────
# Valid values: 1-10; garbage: NoneType, 404_error, [], {}, <html>
def clean_fitness_level(val):
    try:
        v = int(float(str(val).strip()))
        return v if 1 <= v <= 10 else np.nan
    except (ValueError, TypeError):
        return np.nan

df["fitness_level"] = df["fitness_level"].apply(clean_fitness_level)
print(f"fitness_level nulls after cleaning: {df['fitness_level'].isna().sum()}")

# ── 2d. Clean equipment_access ───────────────────────────────────────────
valid_equip = {"none", "minimal", "home_gym", "outdoor", "full_gym"}

def clean_equipment(val):
    if pd.isna(val): return np.nan
    v = str(val).strip().lower()
    return v if v in valid_equip else np.nan

df["equipment_access"] = df["equipment_access"].apply(clean_equipment)

# ── 2e. Clean numeric outliers ───────────────────────────────────────────
# user_age: cap at [10, 80]; impossible values like 5 or 120
df.loc[df["user_age"] < 10,  "user_age"] = np.nan
df.loc[df["user_age"] > 80,  "user_age"] = np.nan

# user_bmi: valid range 10-60
df.loc[df["user_bmi"] < 10,  "user_bmi"] = np.nan
df.loc[df["user_bmi"] > 60,  "user_bmi"] = np.nan

# resting_heart_rate: 30-120
df.loc[df["resting_heart_rate"] < 30, "resting_heart_rate"] = np.nan
df.loc[df["resting_heart_rate"] > 120,"resting_heart_rate"] = np.nan

# target_duration_min: 5-180
df.loc[df["target_duration_min"] <  5,  "target_duration_min"] = np.nan
df.loc[df["target_duration_min"] > 180, "target_duration_min"] = np.nan

print("Outlier cleaning done")

# ── 2f. Clean stress_level (874 missing → impute with mode later) ────────
valid_stress = {"low", "moderate", "high", "chronic"}
df["stress_level"] = df["stress_level"].apply(
    lambda v: v if pd.notna(v) and str(v).strip().lower() in valid_stress else np.nan
)

print(f"\nShape after cleaning: {df.shape}")
print(f"Missing values per column:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  STEP 3 — FEATURE ENGINEERING")
print("=" * 65)

# Derived physiological features
df["cardio_capacity_index"]  = df["vo2_max_estimate"] / (df["resting_heart_rate"] + 1)
df["heart_rate_efficiency"]  = df["heart_rate_reserve"] / (df["max_heart_rate"] + 1)
df["fitness_experience_ratio"] = df["fitness_level"].fillna(1) / (df["experience_years"] + 1)

# Lifestyle composite score  (0–1 scale)
stress_map  = {"low": 3, "moderate": 2, "high": 1, "chronic": 0}
sleep_map   = {"excellent": 3, "good": 2, "fair": 1, "poor": 0}

df["stress_score"] = df["stress_level"].map(stress_map).fillna(1)
df["sleep_score"]  = df["sleep_quality"].map(sleep_map).fillna(1)
df["wellness_score"] = (
    df["stress_score"] / 3 * 0.3
  + df["sleep_score"]  / 3 * 0.3
  + df["diet_quality_score"] / 10 * 0.2
  + df["hydration_liters"]  / 4   * 0.2
)

# Time efficiency ratio
df["time_efficiency"] = df["workout_adherence_pct"] / (df["time_available_min"] + 1)

print("New features: cardio_capacity_index, heart_rate_efficiency,")
print("              fitness_experience_ratio, wellness_score, time_efficiency")

# ═══════════════════════════════════════════════════════════════════════════
# 4. FEATURE SELECTION & ENCODING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  STEP 4 — ENCODING & FEATURE SELECTION")
print("=" * 65)

# Ordinal encodings
ORDINAL_MAPS = {
    "fitness_goal": {
        "rehabilitation": 1, "flexibility": 2, "general_health": 3,
        "weight_loss": 4, "endurance": 5, "muscle_gain": 6,
        "strength": 7, "athletic_performance": 8
    },
    "activity_level": {
        "sedentary": 1, "lightly_active": 2, "moderately_active": 3,
        "very_active": 4, "extremely_active": 5
    },
    "equipment_access": {
        "none": 1, "minimal": 2, "outdoor": 3, "home_gym": 4, "full_gym": 5
    },
    "injury_history": {
        "none": 0, "minor": 1, "major": 2, "chronic": 3
    },
    "preferred_timing": {
        "morning": 1, "afternoon": 2, "evening": 3, "night": 4
    },
    "social_preference": {
        "solo": 1, "partner": 2, "group": 3, "online_class": 4
    },
}

for col, mapping in ORDINAL_MAPS.items():
    df[col + "_enc"] = df[col].map(mapping)

# Target encoding
TARGET_MAP = {"strength": 0, "hiit": 1, "flexibility": 2}
df["label"] = df["target_workout_type"].map(TARGET_MAP)

# Final feature set
FEATURE_COLS = [
    # User physiology
    "user_age", "user_bmi", "user_body_fat_pct",
    "fitness_level", "experience_years",
    "resting_heart_rate", "max_heart_rate",
    "heart_rate_reserve", "vo2_max_estimate",
    # User behaviour
    "time_available_min", "weekly_consistency_days",
    "workout_adherence_pct", "motivation_score",
    "diet_quality_score", "hydration_liters",
    # Ordinal encoded
    "fitness_goal_enc", "activity_level_enc", "equipment_access_enc",
    "injury_history_enc", "preferred_timing_enc", "social_preference_enc",
    # Derived
    "cardio_capacity_index", "heart_rate_efficiency",
    "fitness_experience_ratio", "wellness_score",
    "time_efficiency", "stress_score", "sleep_score",
]

X = df[FEATURE_COLS].copy()
y = df["label"].copy()

print(f"Feature matrix shape: {X.shape}")
print(f"Label distribution  : {y.value_counts().to_dict()}")

# ═══════════════════════════════════════════════════════════════════════════
# 5. PREPROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  STEP 5 — TRAIN / TEST SPLIT + IMPUTATION")
print("=" * 65)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Impute → Scale
imputer = SimpleImputer(strategy="median")
scaler  = StandardScaler()

X_train_imp = imputer.fit_transform(X_train)
X_test_imp  = imputer.transform(X_test)

X_train_sc  = scaler.fit_transform(X_train_imp)
X_test_sc   = scaler.transform(X_test_imp)

print(f"Train size: {X_train_sc.shape[0]:,} | Test size: {X_test_sc.shape[0]:,}")

# ═══════════════════════════════════════════════════════════════════════════
# 6. MODEL TRAINING — FIVE CLASSIFIERS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  STEP 6 — TRAINING MODELS")
print("=" * 65)

MODELS = {
    "Logistic Regression":      LogisticRegression(max_iter=1000, random_state=42, C=1.0),
    "K-Nearest Neighbors":      KNeighborsClassifier(n_neighbors=7),
    "Random Forest":            RandomForestClassifier(n_estimators=300, max_depth=15,
                                                        random_state=42, n_jobs=-1),
    "Gradient Boosting":        GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                                            max_depth=5, random_state=42),
    "XGBoost":                  XGBClassifier(n_estimators=300, learning_rate=0.05,
                                              max_depth=6, use_label_encoder=False,
                                              eval_metric="mlogloss", random_state=42,
                                              n_jobs=-1),
}

results     = {}
cv          = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for name, model in MODELS.items():
    print(f"\n  ▶ Training {name}...")

    # 10-fold cross-validation on training set
    cv_scores = cross_val_score(model, X_train_sc, y_train,
                                cv=cv, scoring="f1_weighted", n_jobs=-1)

    # Final fit & predict
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="weighted")
    prec = precision_score(y_test, y_pred, average="weighted")
    rec  = recall_score(y_test, y_pred, average="weighted")

    results[name] = {
        "model":       model,
        "y_pred":      y_pred,
        "CV_F1_mean":  cv_scores.mean(),
        "CV_F1_std":   cv_scores.std(),
        "Test_Acc":    acc,
        "Test_F1":     f1,
        "Precision":   prec,
        "Recall":      rec,
    }

    print(f"    CV F1  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"    Test Acc: {acc:.4f}  |  Test F1: {f1:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# 7. RESULTS TABLE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  STEP 7 — RESULTS SUMMARY")
print("=" * 65)

metrics_df = pd.DataFrame({
    name: {
        "CV F1 (mean)": f"{v['CV_F1_mean']:.4f}",
        "CV F1 (±std)": f"{v['CV_F1_std']:.4f}",
        "Test Accuracy": f"{v['Test_Acc']:.4f}",
        "Test F1 (w)":   f"{v['Test_F1']:.4f}",
        "Precision (w)": f"{v['Precision']:.4f}",
        "Recall (w)":    f"{v['Recall']:.4f}",
    }
    for name, v in results.items()
}).T

print(metrics_df.to_string())
metrics_df.to_csv(f"{OUTDIR}/model_results_table.csv")

# Best model
best_name  = max(results, key=lambda k: results[k]["Test_F1"])
best_model = results[best_name]["model"]
print(f"\n🏆 Best Model: {best_name}  (Test F1 = {results[best_name]['Test_F1']:.4f})")

# Per-class report for best model
CLASS_NAMES = ["Strength", "HIIT", "Flexibility"]
print(f"\nDetailed Classification Report — {best_name}:")
print(classification_report(y_test, results[best_name]["y_pred"],
                             target_names=CLASS_NAMES))

# ═══════════════════════════════════════════════════════════════════════════
# 8. STATISTICAL SIGNIFICANCE TEST  (paired t-test vs Logistic Regression)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  STEP 8 — STATISTICAL SIGNIFICANCE (t-test vs Logistic Regression)")
print("=" * 65)

baseline_scores = cross_val_score(
    MODELS["Logistic Regression"], X_train_sc, y_train,
    cv=cv, scoring="f1_weighted", n_jobs=-1
)

for name, model in MODELS.items():
    if name == "Logistic Regression":
        continue
    model_scores = cross_val_score(model, X_train_sc, y_train,
                                   cv=cv, scoring="f1_weighted", n_jobs=-1)
    t_stat, p_val = stats.ttest_rel(model_scores, baseline_scores)
    sig = "✅ Significant" if p_val < 0.05 else "❌ Not significant"
    print(f"  {name:<25}  t={t_stat:+.3f}  p={p_val:.4f}  {sig}")

# ═══════════════════════════════════════════════════════════════════════════
# 9. VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  STEP 9 — SAVING FIGURES")
print("=" * 65)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# ── Figure 1: Model Comparison Bar Chart ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold")

model_names   = list(results.keys())
test_acc_vals = [results[n]["Test_Acc"] for n in model_names]
test_f1_vals  = [results[n]["Test_F1"]  for n in model_names]

bars1 = axes[0].bar(model_names, test_acc_vals, color=sns.color_palette("muted", len(model_names)))
axes[0].set_title("Test Accuracy")
axes[0].set_ylim(0, 1.05)
axes[0].set_xticklabels(model_names, rotation=20, ha="right")
for bar, val in zip(bars1, test_acc_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)

bars2 = axes[1].bar(model_names, test_f1_vals, color=sns.color_palette("muted", len(model_names)))
axes[1].set_title("Weighted F1-Score")
axes[1].set_ylim(0, 1.05)
axes[1].set_xticklabels(model_names, rotation=20, ha="right")
for bar, val in zip(bars2, test_f1_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig1_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: fig1_model_comparison.png")

# ── Figure 2: Confusion Matrix (Best Model) ───────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
cm = confusion_matrix(y_test, results[best_name]["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
ax.set_title(f"Confusion Matrix — {best_name}", fontweight="bold")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig2_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: fig2_confusion_matrix.png")

# ── Figure 3: CV Score Distribution (Box Plot) ───────────────────────────
cv_all = {}
for name, model in MODELS.items():
    cv_all[name] = cross_val_score(model, X_train_sc, y_train,
                                   cv=cv, scoring="f1_weighted", n_jobs=-1)

fig, ax = plt.subplots(figsize=(10, 5))
ax.boxplot(cv_all.values(), labels=cv_all.keys(), patch_artist=True,
           boxprops=dict(facecolor="#AED6F1"))
ax.set_title("10-Fold CV F1-Score Distribution per Model", fontweight="bold")
ax.set_ylabel("Weighted F1-Score")
ax.set_xticklabels(cv_all.keys(), rotation=20, ha="right")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig3_cv_boxplot.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: fig3_cv_boxplot.png")

# ── Figure 4: Feature Importance (Best Tree Model) ───────────────────────
tree_models = {k: v["model"] for k, v in results.items()
               if hasattr(v["model"], "feature_importances_")}
best_tree_name = max(tree_models, key=lambda k: results[k]["Test_F1"])
importances    = tree_models[best_tree_name].feature_importances_
feat_imp_df    = (pd.DataFrame({"Feature": FEATURE_COLS, "Importance": importances})
                  .sort_values("Importance", ascending=True)
                  .tail(20))

fig, ax = plt.subplots(figsize=(9, 8))
ax.barh(feat_imp_df["Feature"], feat_imp_df["Importance"],
        color=sns.color_palette("viridis", len(feat_imp_df)))
ax.set_title(f"Top 20 Feature Importances — {best_tree_name}", fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig4_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: fig4_feature_importance.png")

# ── Figure 5: Class Distribution (Target) ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Dataset Class Distribution", fontweight="bold")

df["target_workout_type"].value_counts().plot(
    kind="bar", ax=axes[0], color=["#2E86AB", "#A23B72", "#F18F01"],
    edgecolor="white", rot=0)
axes[0].set_title("Raw Counts")
axes[0].set_xlabel("Workout Type")
axes[0].set_ylabel("Count")
for p in axes[0].patches:
    axes[0].annotate(f"{int(p.get_height()):,}",
                     (p.get_x() + p.get_width()/2, p.get_height()),
                     ha="center", va="bottom")

df["target_workout_type"].value_counts(normalize=True).mul(100).plot(
    kind="pie", ax=axes[1], autopct="%1.1f%%",
    colors=["#2E86AB", "#A23B72", "#F18F01"], startangle=90)
axes[1].set_ylabel("")
axes[1].set_title("Percentage Split")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig5_class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: fig5_class_distribution.png")

# ── Figure 6: Correlation Heatmap ────────────────────────────────────────
numeric_cols = ["user_age", "user_bmi", "fitness_level", "vo2_max_estimate",
                "resting_heart_rate", "motivation_score", "wellness_score",
                "workout_adherence_pct", "time_available_min", "label"]

corr = df[numeric_cols].dropna().corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True, ax=ax, linewidths=0.5)
ax.set_title("Feature Correlation Matrix", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig6_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: fig6_correlation_heatmap.png")

# ── Figure 7: EDA — Fitness Goal vs Workout Type ─────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
ct = pd.crosstab(df["fitness_goal"], df["target_workout_type"], normalize="index") * 100
ct.plot(kind="bar", ax=ax, colormap="tab10", edgecolor="white", rot=20)
ax.set_title("Fitness Goal vs Recommended Workout Type (%)", fontweight="bold")
ax.set_xlabel("Fitness Goal")
ax.set_ylabel("Proportion (%)")
ax.legend(title="Workout Type", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig7_goal_vs_workout.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: fig7_goal_vs_workout.png")

# ═══════════════════════════════════════════════════════════════════════════
# 10. ABLATION STUDY  (remove feature groups one at a time)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  STEP 10 — ABLATION STUDY")
print("=" * 65)

FEATURE_GROUPS = {
    "Physiology":   ["user_age", "user_bmi", "user_body_fat_pct",
                     "resting_heart_rate", "max_heart_rate",
                     "heart_rate_reserve", "vo2_max_estimate"],
    "Fitness":      ["fitness_level", "experience_years",
                     "fitness_goal_enc", "activity_level_enc"],
    "Lifestyle":    ["wellness_score", "stress_score", "sleep_score",
                     "diet_quality_score", "hydration_liters"],
    "Behavioural":  ["time_available_min", "weekly_consistency_days",
                     "workout_adherence_pct", "motivation_score",
                     "time_efficiency"],
    "Derived":      ["cardio_capacity_index", "heart_rate_efficiency",
                     "fitness_experience_ratio"],
}

best_rf = results["Random Forest"]["model"]
ablation_rows = []

# Baseline
X_all_imp = imputer.transform(X)
X_all_sc  = scaler.transform(X_all_imp)
base_f1 = cross_val_score(best_rf, X_train_sc, y_train,
                           cv=StratifiedKFold(5, shuffle=True, random_state=42),
                           scoring="f1_weighted").mean()

ablation_rows.append({"Removed Group": "None (Baseline)", "CV F1": base_f1,
                      "Drop in F1": 0.0})

for group_name, group_cols in FEATURE_GROUPS.items():
    remaining = [c for c in FEATURE_COLS if c not in group_cols]
    X_ab = X[remaining].copy()

    imp_ab = SimpleImputer(strategy="median")
    sc_ab  = StandardScaler()
    X_tr_ab, _, y_tr_ab, _ = train_test_split(X_ab, y, test_size=0.2,
                                               random_state=42, stratify=y)
    X_tr_ab = sc_ab.fit_transform(imp_ab.fit_transform(X_tr_ab))

    f1_ab = cross_val_score(
        RandomForestClassifier(n_estimators=100, random_state=42),
        X_tr_ab, y_tr_ab,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="f1_weighted"
    ).mean()

    drop = base_f1 - f1_ab
    ablation_rows.append({"Removed Group": group_name,
                           "CV F1": f1_ab, "Drop in F1": drop})
    print(f"  Without {group_name:<15}: F1 = {f1_ab:.4f}  (drop = {drop:+.4f})")

ablation_df = pd.DataFrame(ablation_rows)
ablation_df.to_csv(f"{OUTDIR}/ablation_study.csv", index=False)
print("\nAblation study saved.")

# ── Figure 8: Ablation Bar Chart ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#2ECC71" if d <= 0 else "#E74C3C"
          for d in ablation_df["Drop in F1"]]
ax.bar(ablation_df["Removed Group"], ablation_df["CV F1"], color=colors, edgecolor="white")
ax.axhline(y=base_f1, color="navy", linestyle="--", label=f"Baseline F1 = {base_f1:.4f}")
ax.set_title("Ablation Study — Impact of Removing Feature Groups", fontweight="bold")
ax.set_ylabel("CV Weighted F1-Score")
ax.set_xticklabels(ablation_df["Removed Group"], rotation=20, ha="right")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig8_ablation_study.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: fig8_ablation_study.png")

# ═══════════════════════════════════════════════════════════════════════════
# 11. SAVE CLEAN DATASET
# ═══════════════════════════════════════════════════════════════════════════
df.to_csv(f"{OUTDIR}/cleaned_workout_dataset.csv", index=False)
print(f"\n✅ Clean dataset saved — {df.shape[0]:,} rows × {df.shape[1]} columns")

# ═══════════════════════════════════════════════════════════════════════════
# 12. FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  PIPELINE COMPLETE — SUMMARY")
print("=" * 65)
print(f"  Dataset rows (clean)  : {df.shape[0]:,}")
print(f"  Features used         : {len(FEATURE_COLS)}")
print(f"  Models trained        : {len(MODELS)}")
print(f"  CV folds              : 10")
print(f"  Best model            : {best_name}")
print(f"  Best Test Accuracy    : {results[best_name]['Test_Acc']:.4f}")
print(f"  Best Test F1 (w)      : {results[best_name]['Test_F1']:.4f}")
print(f"  Outputs saved to      : {OUTDIR}")
print("=" * 65)
