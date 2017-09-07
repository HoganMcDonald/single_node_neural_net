# single_node_neural_net
This is an implimentation of a single node of a deep learning network. It is able to leverage the numpy library to carry out the steps involved in logistic regression to identify if images contain cats.

This was built as part of the deeplearning.ai level 1 certification in deep learning. This type of algorithm could be used in a network to implement a true neural net. Logistic regression is able to take in very large sets of data (such as images with many pixels arranged in 3 color channels), and output a single bit (1 or 0). In this case, it is able to learn how to identify a cat using sample data and identify if an input image matches generally the parameters present in a training set. 

A more well refined network of neurons would be able to be fine tuned to identify different aspects of images(such as recognizing the edges of objects, etc.) and create a more accurate method for prediction.

This repo does not contain a training set, so it is more for reference, but future projects will have working front ends for users to easily interact with.