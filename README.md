# Introdeeplearning Lab 1

Please copy the following commands to your bash terminal:

## Pip Instructions
If you are using pip:
```
git clone https://github.com/yala/introdeeplearning;
cd introdeeplearning;
pip install virtualenv;
virtualenv venv --python=python2.7;
source venv/bin/activate;
pip install --upgrade pip;
source venv/bin/activate
pip install tensorflow;
pip install word2vec;
pip install jupyter;
echo 'done';
jupyter notebook
```
## Conda Instructions
```
git clone https://github.com/yala/introdeeplearning;
cd introdeeplearning;
conda create -n introdl python=2.7;
source activate introdl;
conda install tensorflow;
conda install word2vec;
conda install jupyter;
echo 'done';
jupyter notebook
```
