# NAS_Challenge_AutoML_2024

This is the repository of EG-ENAS, an efficient and generalizable NAS framework for image classification based on evolutionary computation, which uses a small version of the RegNet search space and reuses available prior knowledge across tasks and  proxies to reduce redundant computations. It aligns with the constraints set by the [NAS Unseen Data Challenge](https://github.com/Towers-D/NAS-Unseen-Datasets), which share similar goals and motivations, allowing for a
consistent framework that can be easily used by other researchers and users. our low-cost (T0) and full EG-ENAS (T6) configurations achieve robust performance across 11 different datasets, with competitive results in under 24 hours on the seven validation datasets used for the NAS Unseen Data Challenge 2023. In Figure 1 we showed the aggregated relative score for datasets Language, Gutenberg, CIFARTile, AddNIST, MultNIST, GeoClassing and Chesseract [1].

![Figure 1: Model Architecture](images/relative_scores_b.png)

*Figure 1: Total Relative versus time in hours for each study on \textbf{(left)} seven validation datasets and \textbf{(right)} four test datasets*

# Installation
Follow these steps to set up the project in a virtual environment:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ankilab/NAS_Challenge_AutoML_2024.git
   cd repo-name
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python3 -m venv .venv
   ```

3. **Activate the virtual environment**:
   - **On Windows**:
     ```bash
     .venv\Scripts\activate
     ```
   - **On macOS and Linux**:
     ```bash
     source .venv/bin/activate
     ```

4. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Deactivate the virtual environment** (when finished, optional):
   ```bash
   deactivate
   ```

Now, your environment is set up, and you’re ready to run the project.

# Usage 

# Configuration

## References

1. Geada, Rob, et al. "Insights from the Use of Previously Unseen Neural Architecture Search Datasets." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024. [Link](https://arxiv.org/abs/2404.02189)