# 欢迎来到CS224N！

本课程将全程使用Python。如果您已经有一个良好的Python环境，那就太好了！但请确保它至少是Python 3.8版本。如果不是，最简单的做法是确保您的电脑上至少有3GB的空闲空间，然后访问（https://www.anaconda.com/download/）并安装Anaconda的Python 3版本。它可以在任何操作系统上运行。

安装conda后，关闭您可能打开的所有终端。然后打开一个新的终端，并运行以下命令：

## 1. 使用env.yml中指定的依赖项创建一个环境：
    conda env create -f env.yml

## 2. 激活新环境：
    conda activate cs224n
    
## 3. 在新环境中安装IPython内核，这样我们就可以在Jupyter Notebook中使用此环境：

    python -m ipykernel install --user --name cs224n


## 4. 作业1（仅此一次）是一个Jupyter Notebook。完成上述步骤后，您可以通过输入以下命令开始工作：

    jupyter notebook exploring_word_vectors.ipynb
    
## 5. 要确保我们使用的是正确的环境，请转到exploring_word_vectors.ipynb的工具栏，点击“Kernel”->“更改内核”，您应该能在下拉菜单中看到并选择cs224n。

## 要取消激活一个活动环境，请使用
    conda deactivate

# Welcome to CS224N!

We'll be using Python throughout the course. If you've got a good Python setup already, great! But make sure that it is at least Python version 3.8. If not, the easiest thing to do is to make sure you have at least 3GB free on your computer and then to head over to (https://www.anaconda.com/download/) and install the Python 3 version of Anaconda. It will work on any operating system.

After you have installed conda, close any open terminals you might have. Then open a new terminal and run the following command:

## 1. Create an environment with dependencies specified in env.yml:
    conda env create -f env.yml

## 2. Activate the new environment:
    conda activate cs224n
    
## 3. Inside the new environment, install IPython kernel so we can use this environment in jupyter notebook: 
 
    python -m ipykernel install --user --name cs224n


## 4. Homework 1 (only) is a Jupyter Notebook. With the above done you should be able to get underway by typing:

    jupyter notebook exploring_word_vectors.ipynb
    
## 5. To make sure we are using the right environment, go to the toolbar of exploring_word_vectors.ipynb, click on Kernel -> Change kernel, you should see and select cs224n in the drop-down menu.

## To deactivate an active environment, use
    conda deactivate
