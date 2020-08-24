# Incremental-Learning-with-Keras-and-Creme

1. First performed transfer learning and extract features from Covid-19 chest X Rays dataset.Used the Keras deep learning library and the ResNet50 network (pre-trained on ImageNet). Used ResNet50 to forward propagate images to a pre-specified layer. This output activations of that layer and treat them as a feature vector.
2. The output volume of ResNet50 is 7 x 7 x 2048 = 100,352-dim. Assuming 32-bit floats for our 100,352-dim feature vectors, that implies that trying to store the entire dataset in memory at once would require 10.03GB of RAM.
3. Used the Creme library for Incremental Learning we trained a multi-class Logistic Regression classifier one sample at a time, obtained 97.6% accuracy on the Dogsvs.Cats dataset.
4. The model trained on all 25,000 samples, we reach 97.6% accuracy which is quite respectable.
