from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

path = Path("package")

plt.figure(figsize=(8, 4))

for folder in path.iterdir():
    if folder.stem.startswith("skopt_"):
        results = pd.read_csv(folder / "optimization_history.csv")

        plt.plot(results["Performance"], label=folder.stem.split("_")[1])

# change legend location to right outside of the plot
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel("Iteration")
plt.ylabel("Val accuracy")
plt.title("Optimization results")
plt.tight_layout()
plt.savefig("optimization_results.png")