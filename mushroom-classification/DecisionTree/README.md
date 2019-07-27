# Mushroom classification

## Introduction

Predict mushrooms if they are poisonous or edible by using the decision tree algorithm.

## Test environment

Python 2.7 on Windows 10

## Step-by-step guide

**1. Download mushroom dataset.**

Mushroom dataset can be downloaded from the URL below:

https://www.kaggle.com/uciml/mushroom-classification

Extract `mushrooms.csv` file from the downloaded ZIP file.

**2. Clone example source files.**

This project uses the work from the URL below:

https://github.com/arthur-e/Programming-Collective-Intelligence

To get the code, clone the repository by running a command like the following:

    > git clone https://github.com/arthur-e/Programming-Collective-Intelligence.git

**3. Clone the project source files.**

    > git clone https://github.com/trvoid/mushroom-classification

**4. Copy `treepredict.py` from example source files to the project folder.**

    > copy Programming-Collective-Intelligence\chapter7\treepredict.py mushroom-classification\

**5. Predict observations if they are poisonous or edible.**

Observations are read from `observations.csv`.

    > cd mushroom-classification
    > python mushroom_predict.py <path-to-dataset-file>
