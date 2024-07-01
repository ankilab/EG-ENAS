The CIFARTile dataset is constructed using batches of image from the popular CIFAR-10 dataset (available at https://www.cs.toronto.edu/~kriz/cifar.html)

Each CIFARTile image is a joining of four CIFAR-10 Images in a grid pattern. The solution to this dataset is for a model to identify the number of classes from the original CIFAR-10 that appear in the CIFARTile image. The original CIFAR-10 has 10 classes: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, and Truck. A example CIFARTile image might include two images of horses, a car, and a bird.

The data is in a channels-first format with a shape of (n, 3, 64, 64) where n is the number of samples in the corresponding set (45,000 for training, 15,000 for validation, and 10,000 for testing).

There are four classes in the dataset, with 17,500 examples of each distributed as evenly as possible between the three subsets.

Each images label represents the number of CIFAR-10 classes that appear minus one, for zero-indexing. This means the above example of two horse images, one car, and one bird would have a final label of 2, as there are three different classes.

0: All sub-images belong to the same CIFAR-10 class.
1: There are two CIFAR-10 classes among the sub-images.
2: There are three CIFAR-10 classes among the sub-images
3: All usb-images belong to different CIFAR-10 classes.