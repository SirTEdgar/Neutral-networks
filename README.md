Digit Recognition with Neural Networks (MLP)

A Jupyter notebook that trains and compares basic neural networks (multilayer perceptrons) to classify handwritten digits, using both TensorFlow/Keras and PyTorch.

Notebook: neural_networks_digits_mlp.ipynb

Scenario

A team is experimenting with a digit recognition system, similar to classifying handwritten forms or invoice numbers, and needs a basic neural network model to classify images.

Objectives
Prepare image data and split it into training, validation, and test sets.
Build and train an MLP in Keras, and find which activation function (relu, tanh, sigmoid) works best.
Build the same model in PyTorch and compare the two workflows.
Evaluate with accuracy and a confusion matrix, then improve the model by changing layers, hyperparameters, and optimizers.
Data
Item	Detail
Dataset	scikit-learn load_digits (bundled with the library, no download needed)
Size	1,797 images of 8x8 pixels (64 features), 10 classes (digits 0-9)
Scaling	Pixel values (0-16) divided by 16 to give the range [0, 1]
Data splits
Set	Share	Samples	Purpose
Train	64%	1,149	Fit model weights
Validation	16%	288	Choose activation, architecture, optimizer
Test	20%	360	Final unbiased evaluation

All splits are stratified (each digit is represented proportionally) and use random_state=42. The assignment specifies an 80/20 train-test split. The validation set is carved from the 80% training portion, so model selection never touches the test set.

Requirements
Python 3.9 or newer
Packages:
bash
pip install tensorflow torch scikit-learn seaborn matplotlib pandas numpy

A GPU is not needed. The dataset is small and the notebook runs on a CPU.

How to run
Install the packages above.
Open the notebook in Jupyter, VS Code, or Google Colab (Colab already includes TensorFlow and PyTorch).
Choose Run All. The experiment sections train many small models (3 seeds each), so expect a few minutes on a CPU.
Notebook structure
Section	What it does
1. Setup	Imports and random seeds
2. Prepare data and split	Loads digits, scales to [0, 1], shows sample images and class balance, creates the splits
3.1 Baseline Keras MLP	64 → 128 → 64 → 10 (softmax) network with relu, trained for 30 epochs, with loss and accuracy curves
3.2 Activation comparison	relu vs tanh vs sigmoid, mean validation accuracy over 3 seeds
4. PyTorch MLP	Same architecture, optimizer, batch size, epochs, and split, with a hand-written training loop and a Keras vs PyTorch curve comparison
5.1 Baseline evaluation	Test accuracy for both frameworks, confusion matrix, classification report
5.2 Experiments	Layer sizes and depth, dropout, optimizers (Adam, SGD with momentum, RMSprop), and learning rates
5.3 Final model	Best configuration (chosen on validation accuracy) evaluated once on the test set, with confusion matrix and misclassified examples
6. Findings and recommendations	Editable summary of conclusions
Model details
Setting	Value
Hidden layers	128 and 64 units (baseline)
Output layer	10 units, softmax (Keras); raw logits with CrossEntropyLoss (PyTorch)
Loss	sparse_categorical_crossentropy (Keras), CrossEntropyLoss (PyTorch)
Optimizer	Adam, learning rate 1e-3 (baseline)
Batch size / epochs	32 / 30
Keras vs PyTorch: key differences
Keras uses compile() and fit(), so training is concise.
PyTorch requires a model class with a forward method and a manual training loop (forward pass, loss, backward(), optimizer step).
PyTorch's CrossEntropyLoss includes softmax, so the PyTorch model has no softmax layer.
With the same architecture and settings, both should reach similar accuracy. Small differences come from initialization and batching.
Results

Results are not recorded here because they depend on your run. After running the notebook, fill in the table below and the findings section (section 6) in the notebook.

Model	Validation accuracy	Test accuracy
Keras baseline (relu)		
PyTorch baseline (relu)		
Best configuration		

Best activation: ____________ Best configuration: ____________

Limitations
The digits are small (8x8) and clean. Real forms and invoices have more variation, noise, and resolution.
The test set has only 360 images, so one misclassified image changes accuracy by about 0.3 percentage points. Differences between close models may not be meaningful.
Experiments use 30 epochs without early stopping, so slow optimizers (for example, low learning rates or sigmoid) may look worse than they would with longer training.
Seed-averaged validation accuracy reduces, but does not remove, run-to-run noise.
Troubleshooting
Problem	Fix
ImportError: cannot import name 'layers' from 'keras'	The notebook uses tensorflow.keras. Make sure TensorFlow 2.x is installed.
Very slow training	Reduce the seeds in run_experiment (for example, seeds=(0,)) or shorten the experiment list.
input_shape warning in Keras	The notebook already uses an explicit Input layer, so this indicates an older copy of the code.
Different numbers each run	Seeds are set, but GPU operations and library versions can still cause small differences.
PyTorch install problems	Use the install selector at pytorch.org for your operating system.
Possible extensions
CNN (convolutional network) that keeps the spatial image structure and scales to larger images.
Early stopping and data augmentation (small shifts and rotations) for robustness.
Cross-validation for more reliable comparisons than a single split.
Larger datasets such as MNIST for a harder, more realistic test.
Regression with neural networks on a numeric dataset (see the earlier neural_networks_digits_wine.ipynb).
Related files
sp500_time_series_forecasting.ipynb and README_time_series.md: companion time series notebook and its README.
neural_networks_digits_wine.ipynb: earlier version covering digits classification plus wine regression.
