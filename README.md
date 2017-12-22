MNIST (Modified National Institute of Standards and Technology database) is a data set which contains thousands images of handwritten numbers in 28x28 format. It is considered as the helloworld of Machine learning i.e when someone starts learning programming , he/she prints hello world as the first program, when someone starts learning Machine learning , he/she start with using MNIST data set to write a program that can correctly predict the hand written number in the images.

The program is written in python and uses following frameworks :
1-TenserFlow ( google's powerful framework for ML )
2-matplotlib ( for data and graph visualizing )
3-python-numpy (for mathematical and numerical computing)

The program uses logistic Regression to build the hypothesis function which outputs probability (between 0 to 1) for an image to be one of the ten classes i.e [0,1,2,3,4,5,6,7,8,9] e.g if an image contains a handwritten number 7 , the output will be something like this 
[ 0.0 , 0.4 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.9 ,0.0 , 0.0 ]

To minimize the cost function Gradient Descent Algorithm is used.

For detailed tutorial on follow official tensorFlow MNIST tutorial :
https://www.tensorflow.org/get_started/mnist/beginners

I found another article which explains the step by step MNIST turorial and contains better explanations for beginners :
https://www.oreilly.com/learning/not-another-mnist-tutorial-with-tensorflow 

some more useful reads:
deep learning intro
https://tech.io/playgrounds/9487/deep-learning-from-scratch---theory-and-implementation
genetic algorithm intro:
https://tech.io/playgrounds/334/genetic-algorithms/algori

useful Books :

Machine learning bible www.deeplearningbook.org

Introduction to Stastical Learning http://www-bcf.usc.edu/~gareth/ISL/ISLR%20Sixth%20Printing.pdf

Course:
Andrew Ng's Machine Learning :https://www.coursera.org/learn/machine-learning
