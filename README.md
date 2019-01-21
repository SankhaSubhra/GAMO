# GAMO: Generative Adversarial Minority Oversampling
The following is an implementation of a smart deep oversampling approach where a classifier attempts to learn on an imbalanced dataset. The theory can be explained as a game between three players, where a classifier performs its usual actions, a generator attempts to create convex combination of points inside a class which are likely to be misclassified by the classifier, and a discriminator enforces the generator to adhere the class distribution. 

