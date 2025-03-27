Create folder with your dataset name and add inside the following files:
metadata
test_x.npy
test_y.npy
train_x.npy
train_y.npy
valid_x.npy
valid_y.npy

where metadata contains the following information:
{
  "num_classes": 20,
  "input_shape": [50000, 3, 28, 28],
  "codename": "Adaline",
  "benchmark": 89.850
}

You can download the datasets from the UNSEEN-NAS-CHALLENGE following the links avaialible in the NAS CHALLENGE repository:
https://github.com/Towers-D/NAS-Unseen-Datasets
