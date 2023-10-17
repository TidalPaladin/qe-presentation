import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Read table
df = pd.read_csv("segmentation_results.csv", index_col="Step")
df = df.loc[:, ("seg-baseline - val/seg_acc_no_bg", "seg-finetune-pretrain - val/seg_acc_no_bg")]
df.columns = ["Baseline", "Pretrained"]

# Set up basic plot
plt.figure(figsize=(10, 6))
max_baseline = df["Baseline"].max()
max_pretrained = df["Pretrained"].max()
plt.plot(df.index, df["Baseline"], label=f"Baseline (Max: {max_baseline:.4f})")
plt.plot(df.index, df["Pretrained"], label=f"Pretrained (Max: {max_pretrained:.4f})")
plt.xlabel("Step")
plt.ylabel("Accuracy")
plt.title("Segmentation Accuracy (validation set, excl. background class)")
plt.legend()

# Draw intersection point where baseline and pretrained meet
max_baseline = df["Baseline"].max()
f = interp1d(df["Pretrained"], df.index)
intersect_step = float(f(max_baseline))
print(max_baseline, intersect_step)

plt.hlines(y=max_baseline, colors='gray', linestyle='--', xmin=intersect_step, xmax=100000)
plt.vlines(x=intersect_step, colors='gray', linestyle='--', ymin=0.4, ymax=max_baseline)
plt.savefig("segmentation_results.png")

percent_improvement = ((max_pretrained - max_baseline) / max_baseline) * 100
print(f"Percent improvement in maximum accuracy for the pretrained model over the baseline model: {percent_improvement:.2f}%")

# Read train and validation accuracies
df1 = pd.read_csv("run-version_1344-tag-train_cls_acc.csv", index_col="Step")
df1 = df1.loc[:, "Value"]
df1.name = "Train"
df2 = pd.read_csv("run-version_1344-tag-val_cls_acc.csv", index_col="Step")
df2 = df2.loc[:, "Value"]
df2.name = "Val"
df = pd.concat([df1, df2], axis=1)

# Cleanup
train = df["Train"].dropna()
train = train.rolling(window=100).mean().dropna()
val = df["Val"].dropna()
val = val.sort_index()


plt.figure(figsize=(10, 6))
max_train = train.max()
max_val = val.max()
plt.plot(train.index, train, label=f"Train (Max: {max_train:.4f})")
plt.plot(val.index, val, label=f"Validation (Max: {max_val:.4f})")
plt.xlabel("Step")
plt.ylabel("Accuracy")
plt.title("Accuracy (Linear Probe)")
plt.legend()
plt.savefig("train_val_accuracies.png")
