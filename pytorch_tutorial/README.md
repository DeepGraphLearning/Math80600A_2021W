# 1 Environment Set-up

## 1.1 Colab

```bash
!pip install -q torch torchvision
```

## 1.2 Config on your own server/laptop

**Step 1: Install conda**.
 
 Aanaconda or Miniconda is a package management platform for Python and R language.
It provides a nice ecosystem for Python users.
You can find the latest installation website [Anaconda](https://docs.anaconda.com/anaconda/install/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

Example on Linux:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

**Step 2: Create and activate virtual environment**
```bash
conda create -n XXX python=3.7
conda activate XXX
```

where `XXX` is the name of your conda environment.
Usually we have one conda environment for each project, and we can install all the required packages in each conda environment.
Thus, we can avoid package conflicting among different projects.
To deactivate the current virtual environment, run `conda deactivate`.

You can check the default packages installed in the current virtual environment by `conda list`,
it shows the following (can be different when using different versions of conda):

```
# packages in environment at path-to-your-virtual-env:
#
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main  
ca-certificates           2020.12.8            h06a4308_0  
certifi                   2020.12.5        py37h06a4308_0  
ld_impl_linux-64          2.33.1               h53a641e_7  
libedit                   3.1.20191231         h14c3975_1  
libffi                    3.3                  he6710b0_2  
libgcc-ng                 9.1.0                hdf63c60_0  
libstdcxx-ng              9.1.0                hdf63c60_0  
ncurses                   6.2                  he6710b0_1  
openssl                   1.1.1i               h27cfd23_0  
pip                       20.3.3           py37h06a4308_0  
python                    3.7.9                h7579374_0  
readline                  8.0                  h7b6447c_0  
setuptools                51.0.0           py37h06a4308_2  
sqlite                    3.33.0               h62c20be_0  
tk                        8.6.10               hbc83047_0  
wheel                     0.36.2             pyhd3eb1b0_0  
xz                        5.2.5                h7b6447c_0  
zlib                      1.2.11               h7b6447c_3  
```

`pip list` is another option, you can play around with it on your end.

**Step 3: Install packages**

Install using `conda`:

```bash
conda install scikit-learn
conda install matplotlib
conda install -c pytorch pytorch=1.6.0
conda install -c pytorch torchvision
```

Or install using `pip`:

```bash
pip install scikit-learn
pip install torch==1.6.0 torchvision
```

You can see the packages installed when doing `conda list` again:

```
# Name                    Version                   Build                               Channel
.
.
.
numpy                     1.19.2                    py37h54aff64_0  
pytorch                   1.6.0                     py3.7_cuda10.2.89_cudnn7.6.5_0      pytorch
scikit-learn              0.23.2                    py37h0573a6f_0  
scipy                     1.5.2                     py37h0b6359f_0  
torchvision               0.7.0                     py37_cu102                          pytorch
matplotlib                3.3.2                     h06a4308_0
.
.
.
```

# 2 Examples

## 2.1 GPU, Check GPU

Check if we can run Pytorch on GPU (if the cuda is available).

`python example_0_GPU_test.py`

## 2.2 Tensor Basics, Some Warm-ups

Basic tensor operation on an image data.

```bash
wget https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png -O lenna.png
python example_1_basic_tensor_operation_on_image.py.py
```

## 2.3 Example 1, Iris Classification

Play around with classification on [Iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html).

### 2.3.1, Step 1: Playing around with Logistic Regression Scikit-Learn on Iris

Recall that Logistic Regression was first proposed for binary classification, and it can be generalized for multi-class classification by using one-vs-rest (OvR).
Feel free to check the [Scikit-Learn API](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

```bash
python example_2_Iris_step_1_Logistic_Regression.py
```

### 2.3.2, Step 2: Linear Model, trained with Gradient Descent (GD)

```bash
python example_2_Iris_step_2_GD.py
```

### 2.3.3, Step 3: Multi-Layer Perceptron (MLP), trained with Gradient Descent (GD)

```bash
python example_2_Iris_step_3_MLP.py
```

### 2.3.4, Step 4: Multi-Layer Perceptron (MLP), trained with Stochastic Gradient Descent (SGD)

```bash
python example_2_Iris_step_4_SGD.py
```

### 2.3.5, Step 5: Wrap-ups

A more formal version which includes more advanced tricks/packages:

```bash
python example_2_Iris_step_5_complete.py \
--batch_size=10 \
--lr=0.03 \
--epochs=200 \
--optimizer=sgd \
--model=mlp
```

## 2.4 Example 2, MNIST Digit Classification

Adapted from the pytorch official github [example](https://github.com/pytorch/examples/blob/master/mnist/main.py).

## 2.5 Example 3, 

```bash
python example_3.py
```

# Related Materials

- GitHub Repository [Linear Algebra With Python](https://github.com/MacroAnalyst/Linear_Algebra_With_Python)
- [Scikit-Learn SVM on Iris](https://scikit-learn.org/stable/auto_examples/svm/plot_iris_svc.html)
- [Scikit-Learn KNN, Logistic Regression, SVM on Iris](https://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html)
