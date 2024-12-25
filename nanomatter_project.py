import pandas as pd
import numpy as np
from scipy.stats import skewnorm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# Loading setpoints from the provided file
setpoints = pd.read_csv("/Users/navin/Downloads/setpoint_publish(in)-3.csv")
setpoints.columns = setpoints.columns.str.strip()  # Clean column names

# Converting numerical columns
setpoints.iloc[:, 1:] = setpoints.iloc[:, 1:].replace(',', '', regex=True).apply(pd.to_numeric, errors='coerce')

# Generating synthetic data
def generate_synthetic_data(mean, std_dev, size=5000, skew=None, seed=None):
    np.random.seed(seed)
    if skew is None:
        return np.random.normal(loc=mean, scale=std_dev, size=size)
    return skewnorm.rvs(a=skew, loc=mean, scale=std_dev, size=size)

# Creating datasets
datasets = {}
for dataset_name in ["opt", "drift", "excursion"]:
    rows = []
    for _, row in setpoints.iterrows():
        mean = row[f"{dataset_name}_mean"]
        std = row[f"{dataset_name}_sig"]
        skew = None
        if dataset_name == "excursion" and row["var"] in ["x4", "x5"]:
            skew = 200
        elif dataset_name == "excursion" and row["var"] == "x8":
            skew = -100
        elif dataset_name == "excursion" and row["var"] == "x9":
            skew = -200
        rows.append(generate_synthetic_data(mean, std, skew=skew))
    datasets[dataset_name] = pd.DataFrame(np.column_stack(rows), columns=setpoints["var"])

# Function to calculate outcome
def calculate_outcome(data, coefficients, intercept):
    return (
        coefficients["x1"] * data["x1"] +
        coefficients["x2"] * (data["x2"] ** 2) +
        coefficients["x4_x5"] * data["x4"] * data["x5"] +
        coefficients["x17"] * data["x17"] +
        coefficients["x7"] * data["x7"] +
        intercept
    )

# Defining coefficients
coefficients_opt = {"x1": -5 * np.pi * 10**4, "x2": 770, "x4_x5": 58, "x17": -890, "x7": 730.5}
coefficients_drift = coefficients_opt.copy()
coefficients_excursion = coefficients_opt.copy()
coefficients_drift.update({"x5_x8_x9": 39 / (10**11)})
coefficients_excursion.update({"x5_x8_x9": 39 / (10**11), "x4_x5_x17": 0.8 / (10**13)})

# Computing outcomes
datasets["opt"]["outcome"] = calculate_outcome(datasets["opt"], coefficients_opt, intercept=7.5)
datasets["drift"]["outcome"] = calculate_outcome(datasets["drift"], coefficients_drift, intercept=7)
datasets["excursion"]["outcome"] = calculate_outcome(datasets["excursion"], coefficients_excursion, intercept=7)

# Kernel density plots
plt.figure(figsize=(10, 6))
for name, df in datasets.items():
    sns.kdeplot(df["outcome"], label=name, fill=True)
plt.title("Kernel Density Plot for Outcomes", fontsize=16)
plt.xlabel("Outcome", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# Correlation heatmap with enhanced readability
corr = datasets["drift"].corr()

# Filter for strong correlations
threshold = 0.5  # Only show correlations greater than this value
filtered_corr = corr[(corr > threshold) | (corr < -threshold)]

plt.figure(figsize=(12, 8))
sns.heatmap(
    filtered_corr,
    annot=True,
    fmt=".2f",
    cmap="RdBu",
    annot_kws={"size": 10},
    cbar_kws={"shrink": 0.8},
    linewidths=0.5
)
plt.title("Correlation Heatmap (Filtered for Strong Relationships)", fontsize=16)
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

# Split data
X = datasets["drift"].drop(columns=["outcome"])
y = datasets["drift"]["outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluating model
y_pred = model.predict(X_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R-squared: {r2_score(y_test, y_pred):.2f}")

# Feature importance plot
plt.figure(figsize=(10, 6))
sns.barplot(x=model.feature_importances_, y=X.columns)
plt.title("Feature Importance (XGBoost)", fontsize=16)
plt.xlabel("Importance", fontsize=14)
plt.ylabel("Features", fontsize=14)
plt.tight_layout()
plt.show()

# Interaction heatmap for x5 and x9
interaction = pd.pivot_table(
    datasets["drift"], values="outcome", 
    index=pd.cut(datasets["drift"]["x5"], bins=10),
    columns=pd.cut(datasets["drift"]["x9"], bins=10), aggfunc="mean"
)

plt.figure(figsize=(10, 8))
sns.heatmap(
    interaction,
    cmap="coolwarm",
    cbar_kws={'label': 'Outcome'},
    linewidths=0.5
)
plt.title("Interaction Effect of x5 and x9", fontsize=16)
plt.xlabel("x9 (Binned)", fontsize=14)
plt.ylabel("x5 (Binned)", fontsize=14)
plt.tight_layout()
plt.show()
