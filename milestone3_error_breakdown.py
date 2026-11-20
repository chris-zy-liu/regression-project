# Milestone 3: AI Adoption Forecast Error Analysis
# Goal: Compare baseline and AI-release forecasting models

from pathlib import Path
import pandas as pd

# Step 1: Load existing prediction results
DATA_PATH = Path(__file__).resolve().parent / "results" / "releases" / "adoption_predictions.csv"

df = pd.read_csv(DATA_PATH)

# Step 2: Inspect the dataset
print("Dataset shape:", df.shape)
print("Available columns:")
print(df.columns.tolist())

# Step 3: Compare prediction errors between two models
# Calculate absolute errors for each prediction
df["baseline_error"] = abs(df["target"] - df["control_ridge"])
df["two_stage_error"] = abs(df["target"] - df["nhpp_ridge"])

# Calculate Mean Absolute Error (MAE)
baseline_mae = df["baseline_error"].mean()
two_stage_mae = df["two_stage_error"].mean()

# Calculate improvement (positive means two-stage is better)
improvement = baseline_mae - two_stage_mae

# Display results
print("\n--- Model Performance Comparison ---")
print(f"Baseline Ridge MAE: {baseline_mae:.3f}")
print(f"Two-Stage Ridge MAE: {two_stage_mae:.3f}")
print(f"MAE Improvement: {improvement:.3f}")

if improvement > 0:
    print("Two-stage forecasting performs better.")
elif improvement < 0:
    print("Baseline forecasting performs better.")
else:
    print("Both models have equal MAE.")


# Step 4: Compare model performance by industry

industry_results = (
    df.groupby("Sector")
    .agg(
        observations=("target", "count"),
        baseline_mae=("baseline_error", "mean"),
        two_stage_mae=("two_stage_error", "mean")
    )
    .reset_index()
)

# Positive improvement means two-stage performs better
industry_results["improvement"] = (
    industry_results["baseline_mae"]
    - industry_results["two_stage_mae"]
)

# Sort industries from most improved to least improved
industry_results = industry_results.sort_values(
    by="improvement",
    ascending=False
)

print("\n--- MAE Comparison by Industry ---")
print(industry_results.round(3).to_string(index=False))


# Step 5: Compare model performance by business size

size_results = (
    df.groupby("Empsize")
    .agg(
        observations=("target", "count"),
        baseline_mae=("baseline_error", "mean"),
        two_stage_mae=("two_stage_error", "mean")
    )
    .reset_index()
)

# Calculate the improvement for each business size
size_results["improvement"] = (
    size_results["baseline_mae"]
    - size_results["two_stage_mae"]
)

# Sort by improvement
size_results = size_results.sort_values(
    by="improvement",
    ascending=False
)

print("\n--- MAE Comparison by Business Size ---")
print(size_results.round(3).to_string(index=False))


# Step 6: Export results and create visualization

import matplotlib.pyplot as plt

# Create an output folder
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "milestone3_error_breakdown"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Save comparison tables
industry_results.to_csv(
    OUTPUT_DIR / "industry_comparison.csv",
    index=False
)

size_results.to_csv(
    OUTPUT_DIR / "business_size_comparison.csv",
    index=False
)

# Prepare data for visualization
plot_data = industry_results.sort_values("improvement")

# Green = improved accuracy; red = worse accuracy
colors = [
    "seagreen" if value > 0 else "indianred"
    for value in plot_data["improvement"]
]

# Create a horizontal bar chart
plt.figure(figsize=(10, 7))

plt.barh(
    plot_data["Sector"].astype(str),
    plot_data["improvement"],
    color=colors
)

# Add reference line
plt.axvline(0, color="black", linewidth=1)

plt.xlabel("MAE Improvement (percentage points)")
plt.ylabel("Industry Sector")
plt.title("Impact of AI Release Features on Forecast Accuracy")

plt.tight_layout()

# Save the figure
plt.savefig(
    OUTPUT_DIR / "industry_improvement.png",
    dpi=300
)

plt.close()

print("\nResults and visualization saved successfully!")



