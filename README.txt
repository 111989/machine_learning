Datasets for GMM were collected from UC Irvine's Machine Learning Repository. 
The training and testing datasets for MNIST are from the official website. 
These are called mnist_train.csv and mnist_test.csv respectively. 

Running the codes:

python Perceptron.py --dataset linearly-separable-dataset.csv --mode erm
python Perceptron.py --dataset linearly-separable-dataset.csv --mode nfold
python Perceptron.py --dataset Breast_cancer_data.csv --mode erm
python Perceptron.py --dataset Breast_cancer_data.csv --mode nfold

python Adaboost.py --dataset Breast_cancer_data.csv --mode erm
python Adaboost.py --dataset Breast_cancer_data.csv --mode nfold

python svm.py --train ./mnist_train.csv --test ./mnist_test.csv --output ./w.txt --dataset mnist --kernel linear
python svm.py --train ./mnist_train.csv --test ./mnist_test.csv --output ./w.txt --dataset mnist --kernel rbf
python svm.py --train ./Breast_cancer_data.csv --test ./Breast_cancer_data.csv --output ./w.txt --dataset bcd --kernel rbf
python svm.py --train ./Breast_cancer_data.csv --test ./Breast_cancer_data.csv --output ./w.txt --dataset bcd --kernel linear

python3 gmm.py --components 1 --train /path/to/optdigits.train --test /path/to/optdigits.test
python3 gmm.py --components 3 --train /path/to/optdigits.train --test /path/to/optdigits.test
python3 gmm.py --components 4 --train /path/to/optdigits.train --test /path/to/optdigits.test
