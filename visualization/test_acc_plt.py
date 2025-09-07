import glob
import re
import matplotlib.pyplot as plt

# Define pairs and their corresponding dataset names
pairs = {
    (1, 2): "cifar10",
    (3, 4): "cifar100",
    (5, 6): "mnist",
    (7, 8): "fmnist",
    (9, 10): "tinyIN",
    (11, 12): "flowers",
    (13, 14): "pets",
    (15, 16): "stl10"
}

# Process each pair
for (run1, run2), dataset_name in pairs.items():
    plt.figure(figsize=(10, 6))
    
    # Process first run in pair
    file1 = f"../results/FINAL_V1_ID{run1:03d}/accuracy_stats/report_val.txt"
    epochs1, top1_vals1 = [], []
    
    try:
        with open(file1, 'r') as f:
            for line in f:
                match = re.search(r'Test epoch (\d+):.*?top1%=([0-9.]+)', line)
                if match:
                    epochs1.append(int(match.group(1)))
                    top1_vals1.append(float(match.group(2)))
    except FileNotFoundError:
        print(f"Warning: File not found - {file1}")

    # Process second run in pair
    file2 = f"../results/FINAL_V1_ID{run2:03d}/accuracy_stats/report_val.txt"
    epochs2, top1_vals2 = [], []
    
    try:
        with open(file2, 'r') as f:
            for line in f:
                match = re.search(r'Test epoch (\d+):.*?top1%=([0-9.]+)', line)
                if match:
                    epochs2.append(int(match.group(1)))
                    top1_vals2.append(float(match.group(2)))
    except FileNotFoundError:
        print(f"Warning: File not found - {file2}")

    # Plot both runs
    plt.plot(epochs1, top1_vals1, 'b-', linewidth=2, label=f'Run {run1}')
    plt.plot(epochs2, top1_vals2, 'r--', linewidth=2, label=f'Run {run2}')
    
    # Configure plot
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Top1 Accuracy', fontsize=12)
    plt.title(f'{dataset_name.upper()} - Top1 Accuracy Comparison', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'{dataset_name}_top1_accuracy.png', dpi=300)
    plt.close()
    print(f"Saved plot for {dataset_name} as {dataset_name}_top1_accuracy.png")

print("All plots generated successfully!")
