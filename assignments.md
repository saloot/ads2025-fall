---
hide_hero: true
layout: page
hide_hero: true
show_sidebar: false
---

# How It Works!
Homeworks will be announced regularly here and in the lectures. For the homeworks, the goal is to perform what we have learned during the lactures and lab sessions on the dataset of your choice. 

* A list of available datasets are listed [here](/ADS2025/resources), but if you are interested to work on a dataset of your own, it can be arranged too. Simply send us an email and we will discuss the details. 
* After a homework is announced, you have one week to submit your results.
* The results should be in the form of a pre-compiled Jupyter notebook. For submitting, you have two options:
    * Create a Github repository and submit each week's assignment there (this is the **strongly recommended** option).
    * Otherwise, you can submit it via a Google Colab and share it with us.

* Homeworks should be done indvidually. 

* You will be graded according to the following criteria:
    * Notebook runs without a problem: 10%
    * It solves/addresses the problem: 50%
    * It is clear and well-commented: 40% 
        * This last part is crucial and detailed explanations are required for the submitted notebooks


* Grades will be available within 10 days.

* Late policy: For each late day, 10% penalty 

* Make sure to spend sufficient time on the assignments, since you will be asked to reproduce one of the assignments you had submitted in an in-person session during the class as well. This will account for 5% of final grade.

**Generative AI policy**: Use of ChatGPT, Bard or other similar tools are allowed and *encouraged*. However, please try to solve the assignments by yourself first, and then use prompts for ChatGPT/Bard/etc. and compare your results. If interested, you can submit both results :)

# Homeworks and Due Dates

### Uploading... someday

### Assignment 1

The main goal of this homework is to apply what you have learned in lectures and **lab sessions** on a dataset of your choice.

---

### Dataset Selection

* Choose a suitable dataset for your homework.
* We strongly recommend using **real-world or industrial datasets**.
* You can find great options on [Kaggle](https://www.kaggle.com/).
* If possible, try using **Iranian datasets** to better understand real data challenges in local industries.

---

### Submission Instructions

Your results must be presented in a well-documented **Jupyter Notebook (.ipynb)**.

When submitting, please do the following:

1. Create a **GitHub repository** for your homework and upload your notebook(s) there.
2. You can also share your notebook via **Google Colab** (make sure link access is open).
3. In your final submission file, include:

   * GitHub link
   * Colab link
   * The actual `.ipynb` notebook file

>  You may write your code modularly (by splitting parts into `.py` files and importing them) instead of putting everything inside a single notebook.

After submission, a short in-person session will be scheduled for you to explain and review your assignment.

---

### Collaboration Policy

All homeworks must be done **individually**.

---

### Evaluation Criteria

You will be graded qualitatively based on the following:

* The analysis solves or meaningfully addresses the problem
* The notebook is clear, readable, and well-commented
* Explanations are concise, insightful, and easy to follow

---

### Bonus Points

If you go beyond the basic requirements and add something interesting, you can earn extra credit. For example:

* Having a clear and informative `README.md` file in your GitHub repository
* Creating interactive visualizations (e.g., using Plotly or Bokeh)
* Including small but meaningful creative touches related to your data or analysis

However, avoid unnecessary long reports, complexity, or flashy additions.
Focus on producing a notebook that is **executable, readable, and educational**, not overloaded or messy.

---

### Late Submission Policy

A **10% penalty** will be applied for each late day.

---

### Generative AI Policy

The use of tools such as **ChatGPT, Claude, Bard**, or other similar AI assistants is allowed and encouraged — but use them wisely.

* Try to solve each problem yourself first.
* Then, you may use AI tools to check, improve, or compare your results.
* Remember: what matters most is **understanding the code you submit**, not who wrote it.
* Be cautious — large models often *hallucinate* or produce inaccurate results.

The goal is to help you become confident in solving real data science problems independently.
**Important: It is recommended that you use the course's Ai teaching assistant before the deadline and upload your answers, approximate scores, and suggestions for improving your implementations.**
---

### Homework Components

#### 1. Exploratory Analysis and Data Cleaning

On the dataset of your choice, perform:

* Comprehensive **Exploratory Data Analysis (EDA)** with meaningful insights
* **Data cleaning:** handling missing, invalid, or duplicate values
* **Preprocessing:** converting all features to numerical values
* **Normalization or standardization**
* Clear explanations and overall notebook readability

---

#### 2. Data Visualization

Practice different visualization techniques using your dataset.
Choose appropriate plot types based on your analysis goals.

Include:

* Pie charts and box plots
* Line charts and multi-line charts
* Bar charts, grouped and stacked bar charts
* Scatter plots and bubble charts
* Charts showing uncertainty (error bars)
* Interactive charts using **Plotly** or **Bokeh**

Each chart must have:

* Proper title
* Axis labels and ranges
* Legends

---

#### 3. Feature Engineering

Using your chosen dataset:

* Create new features based on ratios, binning, mathematical functions, and feature combinations
* Perform date/time transformations
* Calculate counts or aggregation statistics
* Perform **feature selection** using *Mutual Information*
* Apply **dimensionality reduction** using *PCA*
* Provide clear explanations and a readable notebook

> Reflective Question (answer in English or Persian):
> **“When is feature engineering a nice-to-have option, and when is it a must-have?”**

---

#### 4. Web Scraping (Bonus)

Write a simple scraper to extract data for **50 “Samand” cars manufactured after 1385** from [bama.ir](https://bama.ir).

Extract the following fields:

* Price
* Mileage
* Color
* Production year
* Transmission type (manual/automatic)
* Description

Submit:

* Your scraper code
* The collected data in an **Excel (.xlsx)** file

---

### Contact & Questions

If you have any questions about the assignment, feel free to ask in the **Telegram group**.

If you prefer to contact me directly:

* **Telegram:** [t.me/peyman886](https://t.me/peyman886)
* **Email:** [peyman.75.naseri@gmail.com](mailto:peyman.75.naseri@gmail.com)

You can usually find me in the **LLM Lab** during the afternoons :)

---

### Final Note

Grading will be **qualitative** rather than checklist-based.
Don’t focus on filling boxes — focus on **understanding, creativity, and problem-solving ability**.

By the end of this course, you should be able to **independently design and execute a complete data science project**.

**Due date:** Wed, Aban 28, 23:59
{% comment %}

### Assignment 1: Pandas, Colab and Kaggle
* 70 Points: Get familiar with Pandas and Jupyter/Colab/Kaggle Notebooks by completing the exercises on [this mini tutorial](https://www.kaggle.com/learn/pandas) on Kaggle (you can use [Lab Session 1's notebook](https://colab.research.google.com/drive/1BoWlL7S1yZkw3q4tKTGG8ZiXXEcMmJws?usp=sharing)) as an additional reference.
    * Once completed, email us the certificate so that we can celeberate together :)
* 30 Points: Pick a dataset to work on for your homeworks (see some suggestions [here](/ADS2025/resources)).

**Due date:** Tuesday, Bahman 30, 23:59




### Assignment 2: Exploratory Analysis and Data Cleaning
* On the database of your choice, perform Exploratory Data Analysis, Cleaning and Preprocessing.
    * 25 Points on the notebook running correctly.
    * 10 Points on the variety of topics explored on the dataset (EDA)
    * 20 Points on data cleaning (handling all missing, invalid or duplicate values)
    * 20 Points on data preprocessing and converting everything to numerical values
    * 10 Points on data normalization/standardization
    * 15 Points on having sufficient explanations and overall readability of the notebook

* **Bonus 40 Points**: Complete the exercises on [this mini tutorial](https://www.kaggle.com/learn/data-cleaning) on Kaggle.
    * Once completed, send us the certificate via email or through Sharif Course-ware so that we can celeberate together :)

* You can use the [lab sessions's notebook](https://colab.research.google.com/drive/1tgSjMu0var9LQUNG8VymHV47NI7WfDAH?usp=sharing) as a guideline.


Please hand in the *compiled* notebook (or the link to your *compiled* notebook on Google Colab/Github/Kaggle).

**Note:**  
If you are using platforms such as Google Colab, Github, or Kaggle, please ensure that you also submit the `.ipynb` file. 

**Due date:** Tuesday, Esfand 7, 23:59


### Assignment 3: Data Visualization + Web-scraping
* On the database of your choice, practice different data visualization techniques
    * 5 Points on the notebook running correctly.
    * 10 points: pie charts (5 points) and box plots (5 points)
    * 10 Points: line charts (5 points) and stacked (multiple) line charts (5 points)
    * 15 Points: bar charts (5 points), multiple bar charts (5 points) and stacked bar charts (5 points)
    * 10 points: scatter plots (5 points) and bubble charts (5 points)
    * 5 Points on showing the uncertainty (error bars) on a chart of your choice
    * 10 Points on interactive charts using Plotly and/or Bokeh
    * 15 Points on having sufficient explanations and overall readability of the notebook

    * Please Make sure that all your charts have proper **title, axis range, axis labels and legends**.
    * You can use the [lab sessions's notebook](https://colab.research.google.com/drive/1didsHn5Hn3QkeLNuneGsS1Tp3jpuT1nI?usp=sharing) as a guideline.

* Web-scraping (20 points): Please write a simpel scraper to extract the following data for 50 "Samand" cars, manufactured after 1385 from the site: https://bama.ir
  
    * Price 
    * Milage
    * Color
    * Production year
    * Transmission type (manual or automatic)
    * Description

    * Plase submit your code as well as results in the form of an excel file.
    * You can use the [web-scraping notebook](https://colab.research.google.com/drive/1hmaWqEw2WIbrLBynaTo7L1Yn4oWgyHqG?usp=sharing#scrollTo=3H2H6IbNVdfR) as a guideline.

* **Bonus 40 Points**: Complete the exercises on [this mini tutorial](https://www.kaggle.com/learn/data-visualization) on Kaggle.
    * Once completed, send us the certificate via Sharif Coursware so that we can celeberate together :)


Please hand in the *compiled* notebook (or the link to your *compiled* notebook on Google Colab/Github/Kaggle).

**Due date:** Friday, Esfand 17, 23:59


### Assignment 4: Feature Engineering
* On the database of your choice, practice different data visualization techniques
    * 10 Points on the notebook running correctly.
    * Create new features based on:
        * 15 points: ratio, binning, function of a column and combining columns
        * 10 points: date/time
        * 10 points: counts and aggregation
    * 15: feature selection based on Mutual Information
    * 10: dimensonality reduction using PCA
    * 20 Points on having sufficient explanations and overall readability of the notebook
    * 10 Points for answering this question (in Farsi or English in your notebooks): when is feature engineering a "nice to have option" and in what situations it is a "must to have"?

* **Bonus 30 Points**: Complete the exercises on [this mini tutorial](https://www.kaggle.com/learn/feature-engineering) on Kaggle.
    * Once completed, send us the certificate via **Sharif CW page** so that we can celeberate together :)

* You can use the [lab sessions's notebook](https://colab.research.google.com/drive/1WLqnFcHbjC8YaHNsSy2Zt9bZWociDWUi?usp=sharing) as a guideline.


Please hand in the *compiled* notebook (or the link to your *compiled* notebook on Google Colab/Github/Kaggle) **on the assignment page of Sharif CW**.

**Due date:** Friday, Esfand 24, 23:59


### Assignment 5: Accuracy Measures
* On the database of your choice, practice measuring the following accuracy measures:
    * 15 Points on the notebook running correctly.
    * 20 Points: Regression Accuracy Metrics:
        * 5 Points: measuring Mean Squared Error (MSE)
        * 5 Points: measuring Mean Absolute Error (MAE)
        * 5 Points: measuring Mean Absolute Percentage Error (MAPE)
        * 5 Points: measuring R2 Score (MAPE)
    * 15 Points: Binary Classification Accuracy Metrics:
        * 5 Points: Precision
        * 5 Points: Recall
        * 5 Points: F1-Score
    * 25 Points: Multi-class Classification Accuracy Metrics:
        * 5 Points: Precision for each class
        * 5 Points: Recall for each class
        * 15 Points: Macro, Weigthed, Micro-averaged F1-Score
    * 15 Points on having sufficient explanations and overall readability of the notebook
    * 10 Points: Suppose we have a multi-label classification problem in the field of football, where each sample (player) can belong to some of 4 classes that we have:
        * Class 1: The player has played for the national team before
        * Class 2: The player had previous history of heart problems
        * Class 3: The player had knee injuries before
        * Class 4: The player has been the captain of the team in the past 

    What accuracy metric do you use to best capture the accuracy of classification algorithm which predicts the above classes based on some data from each player and why?
 
* To measure the above metric, you can either perform simple regression/classification using scikit-learn modules, or simply generate a random vector as the prediction and measure the difference (accuracy) between this "prediction" vector and the actual values (one of the columns of your choice).

* You can use the [lab sessions's notebook](https://colab.research.google.com/drive/1PYDfB4pAs0Hv2RY55Jsq4bsw9_ByNjsd?usp=sharing) as a guideline.

**IMPORTANT NOTE**: In your notebook, **per cell**, please explain why you are doing that part (in natural language, Farsi or English). Also, you need to explain what you have gained/understood from that part. If you only provide code without the comments, *you will not get the full mark*.


Please hand in the *compiled* notebook (or the link to your *compiled* notebook on Google Colab/Github/Kaggle) **on the assignment page of Sharif CW**.

**Due date:** Friday 7 Farvardin, 23:59


### Assignment 6: Regression Methods
* On the database of your choice, practice measuring the following accuracy measures:
    * 15 Points on the notebook running correctly.
    * 10 Points: Linear Regression (**try** to reach R2-score above 0.8)
    * 10 Points: Linear Regression (**try** to reach R2-score above 0.85)
    * 10 Points: Kernel Regression (R2-score not important!)
    * 10 Points: Logistic Regression (**try** to reach with R2-score above 0.8)
    * 10 Points: Ridge Regression (**try** to reach with R2-score above 0.75)
    * 10 Points: LASSO Regression (**try** to reach with R2-score above 0.75)
    * 15 Points on having sufficient explanations and overall readability of the notebook
    * 10 Points: explain kernel trick in a few sentences and how it can be helpful in achieving better regression results.
 

* You can use the [lab sessions's notebook](https://colab.research.google.com/drive/1HRfgAfs94rtK9WdEUpO9Os43yEt-x7zd?usp=sharing) as a guideline.

**IMPORTANT NOTE**: In your notebook, **per cell**, please explain why you are doing that part (in natural language, Farsi or English). Also, you need to explain what you have gained/understood from that part. If you only provide code without the comments, *you will not get the full mark*.


Please hand in the *compiled* notebook (or the link to your *compiled* notebook on Google Colab/Github/Kaggle) **on the assignment page of sharif CW**.

**Due date:** Friday, Farvardin 21, 23:59


### Assignment 7: Binary Classification Methods
* On the database of your choice, practice measuring the following accuracy measures:
    * 15 Points on the notebook running correctly.
    * 5 Points: Logistic Regression for Classification (with F1-score above 0.75)
    * 10 Points: SVM (with F1-score above 0.8)
    * 5 Points: Kernel SVM (with F1-score above 0.8)
    * 15 Points: KNN (with F1-score above 0.8)
        * 10 Points on the algorithm implementation
        * 5 Points: Tune for best number of neighbors (K)
    * 15 Points: Decision Trees (with F1-score above 0.8)
        * 10 Points on the algorithm implementation
        * 5 Points: Tune for the best maximum depth to avoid overfitting
    * 10 Points: Random Forests (with F1-score above 0.85)
    * 15 Points on having sufficient explanations and overall readability of the notebook
    * 10 Points: explain 3 techniques regualarize the training process for decision trees.

* 40 Bonus points: On the dataset we used in the lab session (for detecting diabetes), achieve an F1 score above 0.9 (on the test set) using whatever classification method you like (test set should be 20% of the whole data).

* All scores are measured on test set, which should be 20% of the whole dataset
* You can use the [lab sessions's notebook](https://colab.research.google.com/drive/1uQRWXCp8o9tSJaRR8C31K2ZPMsHtbfOD?usp=sharing as a guideline.

**IMPORTANT NOTE**: In your notebook, **per cell**, please explain why you are doing that part (in natural language, Farsi or English). Also, you need to explain what you have gained/understood from that part. If you only provide code without the comments, *you will not get the full mark*.


Please hand in the *compiled* notebook (or the link to your *compiled* notebook on Google Colab/Github/Kaggle) **on the assignment page of Sharif CW**.

**Due date:** Friday, Farvardin 29, 23:59


### Assignment 8: Multiclass Classification Methods
* On the database of your choice, perform **multiclass** classification (with at least 4 classes):
    * 15 Points on the notebook running correctly.
    * 10 Points: Multiclass SVM (with F1-score above 2.5/number_of_classes)
    * 15 Points: Multiclass Logistic Regression (with F1-score above 2.5/number_of_classes)
        * 5 Points: using OVR technique
        * 5 Points: using multinomial approach
        * 5 Points: calculate log loss for the output
    * 10 Points: Multiclass KNN (with F1-score above 2.5/number_of_classes)
        * 5 Points on the algorithm implementation
        * 5 Points: Tune for best number of neighbors (K)
    * 5 Points: Multiclass Decision Trees (with F1-score above 2.5/number_of_classes)
    * 15 Points: Boosting Techniques (with F1-score above 2.5/number_of_classes)
        * 5 Points: XGBoost
        * 5 Points: LightGBM
        * 5 Points: Adaboost or Catboost
    * 5 Points: Grid search to tune one of the boosting methods above
    * 15 Points on having sufficient explanations and overall readability of the notebook
    * 10 Points: Please explain how KNN and decision trees can be extended to multi-label classification problems.
 
* 30 Bonus points: On the dataset we used in the lab session (for predicting the position of the players being one of possible *12*), achieve an F1 score above 0.6 (on the test set) using whatever classification method you like (test set should be 20% of the whole data).

* All scores are measured on test set, which should be 20% of the whole dataset
* You can use the [lab sessions's notebook](https://colab.research.google.com/drive/1hqOlp9ps9Umby2Q3FZKEa6glS-g2Hkw-?usp=sharing) as a guideline.

**IMPORTANT NOTE**: In your notebook, **per cell**, please explain why you are doing that part (in natural language, Farsi or English). Also, you need to explain what you have gained/understood from that part. If you only provide code without the comments, *you will not get the full mark*.


Please hand in the *compiled* notebook (or the link to your *compiled* notebook on Google Colab/Github/Kaggle) **on the assignment page of Sharif CW**.

**Due date:** Friday, Ordibehesht 5, 23:59



### Assignment 9: Neural Networks
* On the database of your choice, practice using neural networks:
    * 15 Points on the notebook running correctly.
    * 10 Points: Multilayer Perceptron with Scikit-Learn
        * 5 Points: binary classification with F1-score above 0.75
        * 5 Points: regression with R2-score above 0.8
    * 15 Points: 4-layer feedforward network with Keras
        * 10 Points: binary classification with F1-score above 0.75
        * 5 Points: regression with R2-score above 0.8
    * 20 Points: 4-layer feedforward network with PyTorch
        * 10 Points: binary classification with F1-score above 0.75
        * 10 Points: regression with R2-score above 0.8
    * 15 Points: 4-layer **non-sequential** feedforward network with Keras
        * 5 Points: binary classification with F1-score above 0.75
        * 5 Points: regression with R2-score above 0.8
    
    * 15 Points on having sufficient explanations and overall readability of the notebook

    * 10 Points: Explain why neural networks are so powerful and what the diffcult part is in designing neural networks.
    
    * Bonus 15 Points (if dataset has time-series like features) 3-layer Recurrent Neural Network with Keras
        * 10 Points: binary classification with F1-score above 0.75
        * 5 Points: regression with R2-score above 0.8

* All scores are measured on test set, which should be 20% of the whole dataset
* You can use the [lab sessions's notebook](https://colab.research.google.com/drive/1EuCbMozXH19FbIKmFJtyiCqkpLv5-gh9?usp=sharing) as a guideline.

**IMPORTANT NOTE**: In your notebook, **per cell**, please explain why you are doing that part (in natural language, Farsi or English). Also, you need to explain what you have gained/understood from that part. If you only provide code without the comments, *you will not get the full mark*.


Please hand in the *compiled* notebook (or the link to your *compiled* notebook on Google Colab/Github/Kaggle) **on the assignment page of Sharif CW**.

**Due date:** Friday, Ordibehesht 12, 23:59



### Assignment 10: Deep Neural Networks
* On the database of your choice, practice working with neural networks and tuning them
* You can use the neural network you developed in the previous assignment (using Keras)
* The problem you work on in this exercise can be either regression or classification, whichever you find more suitable (one of them is sufficient)
* For each of the following tasks, please **try at least 5 different options**
* In all cases, please use 4-fold cross validation and use the average validation accuracy as the measure to tune.
    * 15 Points on the notebook running correctly.
    * 10 Points: Tuning for optimization algorithm (e.g. SGD, ADAM, etc.)
    * 5 Points: Tuning learning rate 
    * 5 Points: Tuning learning rate decay
    * 5 Points: Tuning batch size
    * 5 Points: Tuning activation functions
    * 5 Points: Tuning weight intilaization
    * 10 Points: Trying multiple layers and number of neurons (e.g. playing with network architecture)
    * 5 Points: Tuning l1 and l2 regularization in the weights
    * 5 Points: Tuning l1 and l2 regularization in the activity_kernel
    * 5 Points: Tuning dropout rate
    * 15 Points on having sufficient explanations and overall readability of the notebook
    * 10 Points: In a paragraph, explain why it gets more difficult to train deep neural networks when the number of layers increase (i.e. when the network gets deeper).

* All scores are measured on test set, which should be 20% of the whole dataset
* You can use the [lab sessions's notebook](https://colab.research.google.com/drive/1KQAhxy2oVEvKzyyB7JqbGCKD7x-yso9B?usp=sharing) as a guideline.

**IMPORTANT NOTE**: In your notebook, **per cell**, please explain why you are doing that part (in natural language, Farsi or English). Also, you need to explain what you have gained/understood from that part. If you only provide code without the comments, *you will not get the full mark*.


Please hand in the *compiled* notebook (or the link to your *compiled* notebook on Google Colab/Github/Kaggle) **on the assignment page of Sharif CW**.

**Due date:** Friday, Ordibehesht 19, 23:59


### Assignment 11: Convolutional Neural Networks, Transfer Learning and Data Augmentation
* On the database of your choice, practice image classification using convolutional neural networks
* For each of the following tasks, please **try at least 3 different options**.
* In all cases, please use 3-fold cross validation and use the average validation accuracy as the measure to tune.
    * 15 Points on the notebook running correctly.
    * 10 Points: Creating a convolutional network with Keras (with at least two layers of convolution layer)
    * 20 Points: Tuning the above network for:
        * 5 Points: Tuning the kernel size (i.e. the size of the receptive field) for convolutional layers
        * 5 Points: Tuning the stride for convolutional layers
        * 5 Points: Tuning the pooling size (i.e. the size of the receptive field) for pooling layers
        * 5 Points: Tuning the stride for pooling layers
    
    * 10 Points: Perform data augmentation and train your model above using the ImageGenerator class 
    * 20 Points: Perform transfer learning using **two** of the available models in Keras applications (e.g. VGG19, ResNet, EfficientNet, etc.)    
    * 15 Points on having sufficient explanations and overall readability of the notebook
    * 10 Points: Express you opnion about the effects of the window size (i.e. receptive field) in convolution layers on the performance of neural network. In other words, what happens if we increase or decrease the size of the receptive field? and Why?

* You can use the [lab sessions's notebook](https://colab.research.google.com/drive/17j_1eA9_hPPsDvtLdCqXXcCDBOm-tLWb?usp=sharing) as a guideline.

**IMPORTANT NOTE**: In your notebook, **per cell**, please explain why you are doing that part (in natural language, Farsi or English). Also, you need to explain what you have gained/understood from that part. If you only provide code without the comments, *you will not get the full mark*.


Please hand in the *compiled* notebook (or the link to your *compiled* notebook on Google Colab/Github/Kaggle) **on the assignment page of Sharif CW**.

**Due date:** Friday, Ordibehesht 26, 23:59



### Assignment 12: Autoencoders and Generative AI
* Using the dataset for assignment 11 for image analysis:
    * 5 Points: Create a dense autoencoder
    * 10 Points: Using the convolutional architecture from assignment 11, create and train convolutional autoencoder
    * 10 Points: Create and train a denoising autoencoder

* 25 Points: Using CIFAR-10 dataset, create and train a Generative and adversarial Network (GAN).
* 10 Points: Use OpenAI API to generate an image of your choice an a voice reading a text which was generated by the chat completion API.

* 15 Points on the notebook running correctly.
* 15 Points on having sufficient explanations and overall readability of the notebook
* 10 Explain the process of adversarial learning in a few sentences.

* 20 Bonus Points: create and train a Variational AutoEncoder (VAE) to create images similar to Fashion MNIST dataset.

* You can use the [lab sessions's notebook](https://colab.research.google.com/drive/1s0m2oZEmJpLlzi2z2Ifl7KzwJhH6TAGQ?usp=sharing) as a guideline.

**IMPORTANT NOTE**: In your notebook, **per cell**, please explain why you are doing that part (in natural language, Farsi or English). Also, you need to explain what you have gained/understood from that part. If you only provide code without the comments, *you will not get the full mark*.


Please hand in the *compiled* notebook (or the link to your *compiled* notebook on Google Colab/Github/Kaggle) **on the assignment page of Sharif CW**.

**Due date:** Thursday, Khordad 9, 23:59



### Assignment 13: Imbalanced Data and Explainable AI

* Using the dataset for assignment 7 or 8, where you had some *imbalanced classes*:
    * 5 Points: Create a data loading and cleaning pipeline using Pandas pipe
    * 5 Points: Create a preprocessing pipeline using Scikit-learn pipelines
    * 5 Points: Use an Imputer to deal with null data points to your pipe
    * 5 Points: Attach a classifier to your pipeline to test the performance of your algorithm

* Practice dealing with imblanaced data using:
    * 5 Points: Random undersampling for the majority classes
    * 5 Points: Random oversampling for the miniority classes
    * 5 Points: SMOTE oversampling
    * 5 Points: Class weights to perform cost-senisitve training

* Explainable AI: Using the convolutional architecture from assignment 11, explain *why* the model misclassified an image using:
    * 10 Points: Grad-CAM
    * 10 Points: SHAP
    * 5 Points: LIME
    * 5 Points: Eli5

* 10 Points on the notebook running correctly.
* 20 Points on having sufficient explanations and overall readability of the notebook

* You can use the [lab sessions's notebook](https://colab.research.google.com/drive/1s0m2oZEmJpLlzi2z2Ifl7KzwJhH6TAGQ?usp=sharing) as a guideline.

**IMPORTANT NOTE**: In your notebook, **per cell**, please explain why you are doing that part (in natural language, Farsi or English). Also, you need to explain what you have gained/understood from that part. If you only provide code without the comments, *you will not get the full mark*.


Please hand in the *compiled* notebook (or the link to your *compiled* notebook on Google Colab/Github/Kaggle) **on the assignment page of Sharif CW**.

**Due date:** Thursday, Khordad 16, 23:59



# Homeworks and Due Dates - Revised Structure

---

## Assignment 1: Data Fundamentals - Pandas, EDA, Data Cleaning & Preprocessing

**Total Points: 100 + 80 Bonus Points**

### Part 1: Getting Familiar with Pandas and Jupyter/Colab/Kaggle (30 Points)

* 20 Points: Get familiar with Pandas and Jupyter/Colab/Kaggle Notebooks by completing the exercises on [this mini tutorial](https://www.kaggle.com/learn/pandas) on Kaggle. You can use [Lab Session 1's notebook](https://colab.research.google.com/drive/1BoWlL7S1yZkw3q4tKTGG8ZiXXEcMmJws?usp=sharing) as an additional reference.
    * Once completed, email us the certificate so that we can celebrate together :)
* 10 Points: Pick a dataset to work on for your homeworks (see some suggestions [here](/ADS2025/resources)).

### Part 2: Exploratory Data Analysis (15 Points)

* 5 Points on the variety of topics explored on the dataset (EDA)
* 10 Points on the depth and quality of exploratory analysis (correlations, distributions, outliers, relationships between variables)

### Part 3: Data Cleaning (25 Points)

* 10 Points on handling all missing values
* 10 Points on handling invalid or duplicate values
* 5 Points on documentation and explanation of cleaning decisions

### Part 4: Data Preprocessing (20 Points)

* 10 Points on data preprocessing and converting everything to numerical values
* 5 Points on data normalization/standardization
* 5 Points on proper train-test split preparation

### Part 5: Overall Quality (10 Points)

* 5 Points on the notebook running correctly
* 5 Points on having sufficient explanations and overall readability of the notebook

### Bonus Points (80 Points)

* 40 Points: Complete the exercises on [this mini tutorial](https://www.kaggle.com/learn/data-cleaning) on Kaggle.
    * Once completed, send us the certificate via email or through Sharif Courseware so that we can celebrate together :)
* 40 Points: Use Pandas Profiling or sweetviz library to generate a comprehensive automated EDA report

You can use [Lab Session 1's notebook](https://colab.research.google.com/drive/1BoWlL7S1yZkw3q4tKTGG8ZiXXEcMmJws?usp=sharing) and [Lab Session 2's notebook](https://colab.research.google.com/drive/1tgSjMu0var9LQUNG8VymHV47NI7WfDAH?usp=sharing) as guidelines.

Please hand in the *compiled* notebook (or the link to your *compiled* notebook on Google Colab/Github/Kaggle).

**Note:**  
If you are using platforms such as Google Colab, Github, or Kaggle, please ensure that you also submit the `.ipynb` file.

**Due date:** Thursday, Bahman 15, 23:59

---

## Assignment 2: Data Visualization, Web Scraping & Feature Engineering

**Total Points: 100 + 70 Bonus Points**

### Part 1: Data Visualization (45 Points)

* 5 Points on the notebook running correctly
* 5 points: pie charts
* 5 points: box plots
* 5 Points: line charts
* 5 Points: stacked (multiple) line charts
* 5 Points: bar charts
* 5 Points: multiple bar charts
* 5 Points: stacked bar charts
* 5 points: scatter plots and bubble charts
* 5 Points on showing the uncertainty (error bars) on a chart of your choice
* 10 Points on interactive charts using Plotly and/or Bokeh

**Please make sure that all your charts have proper title, axis range, axis labels and legends.**

### Part 2: Web Scraping (15 Points)

* Web-scraping (15 points): Please write a simple scraper to extract the following data for 50 "Samand" cars, manufactured after 1385 from the site: https://bama.ir
    * Price 
    * Mileage
    * Color
    * Production year
    * Transmission type (manual or automatic)
    * Description
    * Please submit your code as well as results in the form of an excel file.
    * You can use the [web-scraping notebook](https://colab.research.google.com/drive/1hmaWqEw2WIbrLBynaTo7L1Yn4oWgyHqG?usp=sharing#scrollTo=3H2H6IbNVdfR) as a guideline.

### Part 3: Feature Engineering - Creating New Features (25 Points)

* 5 Points on the notebook running correctly
* Create new features based on:
    * 10 points: ratio, binning, function of a column and combining columns
    * 5 points: date/time features
    * 5 points: counts and aggregation features

### Part 4: Feature Selection and Dimensionality Reduction (15 Points)

* 10 points: feature selection based on Mutual Information
* 5 points: dimensionality reduction using PCA

### Bonus Points (70 Points)

* 40 Points: Complete the exercises on [this mini tutorial](https://www.kaggle.com/learn/data-visualization) on Kaggle.
    * Once completed, send us the certificate via Sharif Courseware so that we can celebrate together :)
* 30 Points: Complete the exercises on [this mini tutorial](https://www.kaggle.com/learn/feature-engineering) on Kaggle.
    * Once completed, send us the certificate via **Sharif CW page** so that we can celebrate together :)

### Theoretical Question (10 Points)

* 10 Points for answering this question (in Farsi or English in your notebooks): when is feature engineering a "nice to have option" and in what situations it is a "must to have"?

You can use [Lab Session 3's notebook](https://colab.research.google.com/drive/1didsHn5Hn3QkeLNuneGsS1Tp3jpuT1nI?usp=sharing) and [Lab Session 4's notebook](https://colab.research.google.com/drive/1WLqnFcHbjC8YaHNsSy2Zt9bZWociDWUi?usp=sharing) as guidelines.

Please hand in the *compiled* notebook (or the link to your *compiled* notebook on Google Colab/Github/Kaggle) **on the assignment page of Sharif CW**.

**Due date:** Thursday, Esfand 15, 23:59

---

## Assignment 3: Machine Learning Fundamentals - Metrics, Regression & Classification

**Total Points: 100 + 40 Bonus Points**

### Part 1: Accuracy Measures (35 Points)

* 5 Points on the notebook running correctly
* 15 Points: Regression Accuracy Metrics:
    * 4 Points: measuring Mean Squared Error (MSE)
    * 4 Points: measuring Mean Absolute Error (MAE)
    * 4 Points: measuring Mean Absolute Percentage Error (MAPE)
    * 3 Points: measuring R2 Score
* 10 Points: Binary Classification Accuracy Metrics:
    * 3 Points: Precision
    * 3 Points: Recall
    * 4 Points: F1-Score
* 15 Points: Multi-class Classification Accuracy Metrics:
    * 3 Points: Precision for each class
    * 3 Points: Recall for each class
    * 9 Points: Macro, Weighted, Micro-averaged F1-Score

### Theoretical Question about Metrics (10 Points)

* 10 Points: Suppose we have a multi-label classification problem in the field of football, where each sample (player) can belong to some of 4 classes that we have:
    * Class 1: The player has played for the national team before
    * Class 2: The player had previous history of heart problems
    * Class 3: The player had knee injuries before
    * Class 4: The player has been the captain of the team in the past 

What accuracy metric do you use to best capture the accuracy of classification algorithm which predicts the above classes based on some data from each player and why?

### Part 2: Regression Methods (30 Points)

* 6 Points: Linear Regression (**try** to reach R2-score above 0.8)
* 6 Points: Linear Regression (**try** to reach R2-score above 0.85)
* 6 Points: Kernel Regression (R2-score not important!)
* 6 Points: Logistic Regression (**try** to reach with R2-score above 0.8)
* 3 Points: Ridge Regression (**try** to reach with R2-score above 0.75)
* 3 Points: LASSO Regression (**try** to reach with R2-score above 0.75)

### Theoretical Question about Regression (5 Points)

* 5 Points: explain kernel trick in a few sentences and how it can be helpful in achieving better regression results.

### Part 3: Binary Classification Methods (30 Points)

* 3 Points: Logistic Regression for Classification (with F1-score above 0.75)
* 5 Points: SVM (with F1-score above 0.8)
* 3 Points: Kernel SVM (with F1-score above 0.8)
* 9 Points: KNN (with F1-score above 0.8)
    * 6 Points on the algorithm implementation
    * 3 Points: Tune for best number of neighbors (K)
* 7 Points: Decision Trees (with F1-score above 0.8)
    * 5 Points on the algorithm implementation
    * 2 Points: Tune for the best maximum depth to avoid overfitting
* 3 Points: Random Forests (with F1-score above 0.85)

### Theoretical Question about Classification (5 Points)

* 5 Points: explain 3 techniques to regularize the training process for decision trees.

### Part 4: Multiclass Classification Methods (30 Points)

* 6 Points: Multiclass SVM (with F1-score above 2.5/number_of_classes)
* 9 Points: Multiclass Logistic Regression (with F1-score above 2.5/number_of_classes)
    * 3 Points: using OVR technique
    * 3 Points: using multinomial approach
    * 3 Points: calculate log loss for the output
* 6 Points: Multiclass KNN (with F1-score above 2.5/number_of_classes)
    * 3 Points on the algorithm implementation
    * 3 Points: Tune for best number of neighbors (K)
* 3 Points: Multiclass Decision Trees (with F1-score above 2.5/number_of_classes)
* 9 Points: Boosting Techniques (with F1-score above 2.5/number_of_classes)
    * 3 Points: XGBoost
    * 3 Points: LightGBM
    * 3 Points: Adaboost or Catboost
* 3 Points: Grid search to tune one of the boosting methods above

### Theoretical Question about Multiclass (5 Points)

* 5 Points: Please explain how KNN and decision trees can be extended to multi-label classification problems.

### Bonus Points (40 Points)

* 20 Bonus points: On the dataset we used in the lab session (for detecting diabetes), achieve an F1 score above 0.9 (on the test set) using whatever classification method you like (test set should be 20% of the whole data).
* 20 Bonus points: On the dataset we used in the lab session (for predicting the position of the players being one of possible *12*), achieve an F1 score above 0.6 (on the test set) using whatever classification method you like (test set should be 20% of the whole data).

### IMPORTANT NOTES

* To measure the above metrics, you can either perform simple regression/classification using scikit-learn modules, or simply generate a random vector as the prediction and measure the difference (accuracy) between this "prediction" vector and the actual values (one of the columns of your choice).
* On the database of your choice, perform **multiclass** classification (with at least 4 classes)
* All scores are measured on test set, which should be 20% of the whole dataset
* In your notebook, **per cell**, please explain why you are doing that part (in natural language, Farsi or English). Also, you need to explain what you have gained/understood from that part. If you only provide code without the comments, *you will not get the full mark*.

You can use [Lab Session 5's notebook](https://colab.research.google.com/drive/1PYDfB4pAs0Hv2RY55Jsq4bsw9_ByNjsd?usp=sharing), [Lab Session 6's notebook](https://colab.research.google.com/drive/1HRfgAfs94rtK9WdEUpO9Os43yEt-x7zd?usp=sharing), [Lab Session 7's notebook](https://colab.research.google.com/drive/1uQRWXCp8o9tSJaRR8C31K2ZPMsHtbfOD?usp=sharing), and [Lab Session 8's notebook](https://colab.research.google.com/drive/1hqOlp9ps9Umby2Q3FZKEa6glS-g2Hkw-?usp=sharing) as guidelines.

Please hand in the *compiled* notebook (or the link to your *compiled* notebook on Google Colab/Github/Kaggle) **on the assignment page of Sharif CW**.

**Due date:** Thursday, Farvardin 22, 23:59

---

## Assignment 4: Neural Networks & Deep Learning

**Total Points: 100 + 15 Bonus Points**

### Part 1: Basic Neural Networks (25 Points)

* 5 Points on the notebook running correctly
* 5 Points: Multilayer Perceptron with Scikit-Learn
    * 2.5 Points: binary classification with F1-score above 0.75
    * 2.5 Points: regression with R2-score above 0.8
* 10 Points: 4-layer feedforward network with Keras
    * 5 Points: binary classification with F1-score above 0.75
    * 5 Points: regression with R2-score above 0.8
* 5 Points: 4-layer **non-sequential** feedforward network with Keras
    * 2.5 Points: binary classification with F1-score above 0.75
    * 2.5 Points: regression with R2-score above 0.8

### Theoretical Question (5 Points)

* 5 Points: Explain why neural networks are so powerful and what the difficult part is in designing neural networks.

### Part 2: PyTorch Implementation (20 Points)

* 20 Points: 4-layer feedforward network with PyTorch
    * 10 Points: binary classification with F1-score above 0.75
    * 10 Points: regression with R2-score above 0.8

### Part 3: Deep Neural Network Tuning (45 Points)

* You can use the neural network you developed in Part 1 (using Keras)
* The problem you work on in this exercise can be either regression or classification, whichever you find more suitable (one of them is sufficient)
* For each of the following tasks, please **try at least 5 different options**
* In all cases, please use 4-fold cross validation and use the average validation accuracy as the measure to tune.
    * 5 Points: Tuning for optimization algorithm (e.g. SGD, ADAM, etc.)
    * 3 Points: Tuning learning rate 
    * 3 Points: Tuning learning rate decay
    * 3 Points: Tuning batch size
    * 3 Points: Tuning activation functions
    * 3 Points: Tuning weight initialization
    * 6 Points: Trying multiple layers and number of neurons (e.g. playing with network architecture)
    * 3 Points: Tuning l1 and l2 regularization in the weights
    * 3 Points: Tuning l1 and l2 regularization in the activity_kernel
    * 3 Points: Tuning dropout rate

### Theoretical Question (5 Points)

* 5 Points: In a paragraph, explain why it gets more difficult to train deep neural networks when the number of layers increase (i.e. when the network gets deeper).

### Part 4: Convolutional Neural Networks (20 Points)

* On the database of your choice, practice image classification using convolutional neural networks
* For each of the following tasks, please **try at least 3 different options**.
* In all cases, please use 3-fold cross validation and use the average validation accuracy as the measure to tune.
    * 5 Points: Creating a convolutional network with Keras (with at least two layers of convolution layer)
    * 10 Points: Tuning the above network for:
        * 2.5 Points: Tuning the kernel size (i.e. the size of the receptive field) for convolutional layers
        * 2.5 Points: Tuning the stride for convolutional layers
        * 2.5 Points: Tuning the pooling size (i.e. the size of the receptive field) for pooling layers
        * 2.5 Points: Tuning the stride for pooling layers
    * 5 Points: Perform data augmentation and train your model above using the ImageDataGenerator class

### Theoretical Question (5 Points)

* 5 Points: Express your opinion about the effects of the window size (i.e. receptive field) in convolution layers on the performance of neural network. In other words, what happens if we increase or decrease the size of the receptive field? and Why?

### Part 5: Transfer Learning (10 Points)

* 10 Points: Perform transfer learning using **two** of the available models in Keras applications (e.g. VGG19, ResNet, EfficientNet, etc.)

### Bonus Points (15 Points)

* Bonus 15 Points (if dataset has time-series like features): 3-layer Recurrent Neural Network with Keras
    * 10 Points: binary classification with F1-score above 0.75
    * 5 Points: regression with R2-score above 0.8

### IMPORTANT NOTES

* All scores are measured on test set, which should be 20% of the whole dataset
* In your notebook, **per cell**, please explain why you are doing that part (in natural language, Farsi or English). Also, you need to explain what you have gained/understood from that part. If you only provide code without the comments, *you will not get the full mark*.

You can use [Lab Session 9's notebook](https://colab.research.google.com/drive/1EuCbMozXH19FbIKmFJtyiCqkpLv5-gh9?usp=sharing), [Lab Session 10's notebook](https://colab.research.google.com/drive/1KQAhxy2oVEvKzyyB7JqbGCKD7x-yso9B?usp=sharing), and [Lab Session 11's notebook](https://colab.research.google.com/drive/17j_1eA9_hPPsDvtLdCqXXcCDBOm-tLWb?usp=sharing) as guidelines.

Please hand in the *compiled* notebook (or the link to your *compiled* notebook on Google Colab/Github/Kaggle) **on the assignment page of Sharif CW**.

**Due date:** Thursday, Ordibehesht 20, 23:59

---

## Assignment 5: Generative AI, RAG & AI Agents

**Total Points: 100 + 20 Bonus Points**

### Part 1: Autoencoders (25 Points)

* 5 Points: Create a dense autoencoder
* 10 Points: Using the convolutional architecture from assignment 4, create and train convolutional autoencoder
* 10 Points: Create and train a denoising autoencoder

### Part 2: Generative Adversarial Networks (25 Points)

* 20 Points: Using CIFAR-10 dataset, create and train a Generative Adversarial Network (GAN).
    * 5 Points: Implement the Generator network
    * 5 Points: Implement the Discriminator network
    * 5 Points: Implement the training loop with proper loss functions
    * 5 Points: Generate and display sample images showing the progression of training

### Theoretical Question (5 Points)

* 5 Points: Explain the process of adversarial learning in a few sentences.

### Part 3: OpenAI API Integration (10 Points)

* 10 Points: Use OpenAI API to generate an image of your choice and a voice reading a text which was generated by the chat completion API.
    * 5 Points: Image generation using DALL-E API
    * 3 Points: Text generation using Chat Completion API
    * 2 Points: Text-to-Speech using TTS API

### Part 4: RAG (Retrieval-Augmented Generation) System (25 Points)

#### 4.1 Document Processing and Vector Store (10 Points)

* 3 Points: Collect and prepare a corpus of at least 20 documents (can be text files, PDFs, or web pages on a specific topic)
* 4 Points: Implement document chunking strategy:
    * Split documents into appropriate chunk sizes (e.g., 500-1000 tokens)
    * Handle chunk overlap to maintain context
    * Preserve document metadata (source, title, date, etc.)
* 3 Points: Create embeddings for all chunks using OpenAI's embedding model or an open-source alternative (e.g., sentence-transformers)

#### 4.2 Vector Database Implementation (8 Points)

* 5 Points: Set up a vector database using one of the following:
    * ChromaDB (local storage)
    * Pinecone (cloud-based)
    * FAISS (Facebook AI Similarity Search)
* 3 Points: Implement efficient similarity search with proper indexing

#### 4.3 RAG Pipeline Implementation (12 Points)

* 3 Points: Query processing:
    * Implement query embedding
    * Handle query preprocessing (cleaning, reformulation if needed)
* 4 Points: Document retrieval:
    * Implement Top-K similarity search (retrieve 3-5 most relevant chunks)
    * Experiment with different K values and document the results
    * Implement re-ranking mechanism (optional but recommended)
* 3 Points: Context injection and prompt engineering:
    * Design effective prompts that incorporate retrieved context
    * Handle cases where retrieved documents may be insufficient
* 2 Points: Response generation using LLM (OpenAI GPT-4, GPT-3.5, or Anthropic Claude)

#### 4.4 Evaluation and Comparison (7 Points)

* 4 Points: Create at least 5 test queries and compare responses:
    * Baseline: LLM without RAG
    * With RAG: LLM with retrieved context
    * Document the quality differences
* 3 Points: Implement basic evaluation metrics:
    * Answer relevance scoring
    * Source attribution accuracy
    * Response completeness

### Theoretical Question (3 Points)

* 3 Points: Explain the difference between RAG and Fine-tuning. When would you use each approach and why?

### Part 5: AI Agents (15 Points)

#### 5.1 Simple ReAct Agent (8 Points)

* 5 Points: Implement a ReAct (Reasoning and Acting) agent with the following components:
    * Thought: Agent's reasoning about what to do next
    * Action: Tool selection and execution
    * Observation: Processing tool results
* 3 Points: Integrate at least 3 tools from the following options:
    * Calculator (for mathematical operations)
    * Web Search (using SerpAPI or similar)
    * Wikipedia Search
    * Weather API
    * Custom tool of your choice (e.g., database query, file reader)

#### 5.2 Multi-Step Task Execution (7 Points)

* 4 Points: Implement a planning mechanism:
    * Break down complex queries into sub-tasks
    * Create a task execution plan
    * Handle task dependencies
* 3 Points: Demonstrate Chain of Thought reasoning:
    * Show step-by-step reasoning process
    * Handle errors and retry logic
    * Maintain conversation context across multiple steps

#### 5.3 Agent Examples (Required)

Demonstrate your agent with at least 2 complex tasks such as:
* "Find the population of the capital city of the country with the highest GDP in Europe, then calculate what 15% of that population would be"
* "Search for the current weather in Tokyo, convert the temperature to Fahrenheit if it's in Celsius, and tell me if I should bring an umbrella"
* Design your own multi-step task that requires at least 3 tool calls

### Theoretical Question (3 Points)

* 3 Points: Explain the key differences between a simple RAG system and an AI Agent. What are the advantages and limitations of each?

### Bonus Points (20 Points)

* 20 Bonus points: create and train a Variational AutoEncoder (VAE) to create images similar to Fashion MNIST dataset.
    * 8 Points: Implement the encoder network with proper sampling layer
    * 8 Points: Implement the decoder network
    * 4 Points: Implement and explain the VAE loss function (reconstruction loss + KL divergence)

### IMPORTANT NOTES

* For Part 4 (RAG) and Part 5 (AI Agents), you will need API keys. You can use:
    * OpenAI API (requires payment but offers free credits for new users)
    * Anthropic Claude API
    * Open-source alternatives like Hugging Face models
* Document your API usage and any costs incurred
* Include error handling and edge cases in your implementations
* In your notebook, **per cell**, please explain why you are doing that part (in natural language, Farsi or English). Also, you need to explain what you have gained/understood from that part. If you only provide code without the comments, *you will not get the full mark*.

You can use [Lab Session 12's notebook](https://colab.research.google.com/drive/1s0m2oZEmJpLlzi2z2Ifl7KzwJhH6TAGQ?usp=sharing) as a guideline for the Autoencoders and GAN sections.

Please hand in the *compiled* notebook (or the link to your *compiled* notebook on Google Colab/Github/Kaggle) **on the assignment page of Sharif CW**.

**Due date:** Thursday, Khordad 10, 23:59

---

## Assignment 6: Production ML, Software Engineering & Explainable AI

**Total Points: 100**

### Part 1: ML Pipelines and Imbalanced Data (30 Points)

#### 1.1 Pipeline Construction (15 Points)

* 3 Points: Create a data loading and cleaning pipeline using Pandas pipe
    * Implement custom functions for data validation
    * Chain multiple cleaning operations
* 4 Points: Create a preprocessing pipeline using Scikit-learn pipelines
    * Include proper feature scaling
    * Handle categorical encoding within the pipeline
* 3 Points: Use an Imputer to deal with null data points in your pipeline
    * Compare different imputation strategies (mean, median, KNN, iterative)
* 5 Points: Attach a classifier to your pipeline to test the performance
    * Ensure the entire pipeline is trainable as a single unit
    * Save and load the complete pipeline

#### 1.2 Handling Imbalanced Data (15 Points)

Using the dataset from assignment 3 where you had some *imbalanced classes*:

* 4 Points: Random undersampling for the majority classes
    * Implement and compare different undersampling strategies
    * Analyze the impact on model performance
* 4 Points: Random oversampling for the minority classes
    * Compare with undersampling results
* 4 Points: SMOTE (Synthetic Minority Over-sampling Technique)
    * Explain how SMOTE generates synthetic samples
    * Compare with simple oversampling
* 3 Points: Class weights to perform cost-sensitive training
    * Calculate appropriate class weights
    * Integrate into your classifier

### Part 2: Explainable AI (XAI) (20 Points)

Using the convolutional neural network from assignment 4:

#### 2.1 Visual Explanations (15 Points)

* 5 Points: Implement Grad-CAM (Gradient-weighted Class Activation Mapping)
    * Generate heatmaps for correctly classified images
    * Generate heatmaps for misclassified images
    * Explain what the model is "looking at"
* 5 Points: Implement SHAP (SHapley Additive exPlanations)
    * Generate SHAP values for image predictions
    * Create summary plots and waterfall plots
* 3 Points: Implement LIME (Local Interpretable Model-agnostic Explanations)
    * Explain individual predictions with superpixel-based explanations
* 2 Points: Implement ELI5 (Explain Like I'm 5)
    * Generate feature importance visualizations

#### 2.2 Analysis and Insights (5 Points)

* 5 Points: For at least 3 misclassified images, provide detailed analysis:
    * What features did the model focus on?
    * Why did the misclassification occur?
    * What could be improved in the training process?

### Part 3: Model Deployment with FastAPI (25 Points)

#### 3.1 API Development (15 Points)

* 5 Points: Create REST API endpoints using FastAPI:
    * `/train` - Endpoint to trigger model training
    * `/predict` - Endpoint for single predictions
    * `/batch_predict` - Endpoint for batch predictions
    * `/model_info` - Endpoint to get model metadata (accuracy, version, etc.)
* 4 Points: Implement request/response validation using Pydantic:
    * Define input models with proper type hints
    * Define output models with structured responses
    * Include data validation rules
* 3 Points: Implement proper error handling:
    * Handle invalid inputs gracefully
    * Return meaningful error messages
    * Use appropriate HTTP status codes
* 3 Points: Add API documentation:
    * Write clear docstrings for all endpoints
    * Use FastAPI's automatic OpenAPI documentation
    * Include example requests and responses

#### 3.2 Model Serialization and Loading (5 Points)

* 3 Points: Implement model saving and loading:
    * Save trained models using joblib or pickle
    * For neural networks, save both architecture and weights
    * Version your models (model_v1.pkl, model_v2.pkl)
* 2 Points: Implement model versioning strategy:
    * Track model versions and their performance metrics
    * Allow API to load specific model versions

#### 3.3 API Testing (5 Points)

* 3 Points: Test your API using:
    * curl commands or Postman
    * Document successful requests and responses
* 2 Points: Performance testing:
    * Measure prediction latency
    * Test with concurrent requests

### Part 4: Dockerization (15 Points)

#### 4.1 Dockerfile Creation (8 Points)

* 5 Points: Write a Dockerfile that:
    * Uses an appropriate base image (python:3.9-slim or similar)
    * Installs all necessary dependencies
    * Copies your application code
    * Exposes the correct port
    * Sets appropriate working directory
* 3 Points: Optimize the Docker image:
    * Use multi-stage builds if applicable
    * Minimize image size
    * Use .dockerignore file

#### 4.2 Container Management (7 Points)

* 3 Points: Build the Docker image:
    * Document the build process
    * Tag the image appropriately
* 2 Points: Run the container:
    * Map ports correctly
    * Handle environment variables
* 2 Points: Test the containerized application:
    * Verify API accessibility from outside the container
    * Test all endpoints with the containerized version

### Part 5: Version Control and Code Organization (15 Points)

#### 5.1 Git Workflow (6 Points)

* 2 Points: Initialize a Git repository with proper structure
* 2 Points: Make regular commits with meaningful commit messages:
    * Follow conventional commit format (feat:, fix:, docs:, etc.)
    * Show progression of your work
* 2 Points: Implement branching strategy:
    * Create feature branches for new functionality
    * Show at least one merge from a feature branch

#### 5.2 Project Structure (5 Points)

* 5 Points: Organize your code with the following structure:

```
project/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── preprocessing.py
│   │   └── loading.py
│   ├── models/
│   │   ├── train.py
│   │   └── predict.py
│   ├── api/
│   │   └── main.py
│   └── utils/
│       └── helpers.py
├── tests/
│   ├── test_preprocessing.py
│   └── test_api.py
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── saved_models/
├── notebooks/
│   └── exploration.ipynb
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

#### 5.3 Documentation (4 Points)

* 4 Points: Create a comprehensive README.md including:
    * Project description and objectives
    * Installation instructions
    * Usage examples for API
    * Docker build and run commands
    * Project structure explanation
    * Dependencies and requirements

### Part 6: Testing and Production Best Practices (15 Points)

#### 6.1 Unit and Integration Testing (7 Points)

* 4 Points: Write unit tests using pytest:
    * Test preprocessing functions
    * Test model prediction functions
    * Achieve reasonable test coverage
* 3 Points: Write integration tests for API:
    * Test API endpoints
    * Test request/response format
    * Test error handling

#### 6.2 Logging and Monitoring (4 Points)

* 2 Points: Implement logging:
    * Use Python's logging module
    * Set appropriate log levels (DEBUG, INFO, WARNING, ERROR)
    * Log important events (predictions, errors, API calls)
* 2 Points: Implement basic monitoring:
    * Track prediction latency
    * Count API requests
    * Monitor error rates

#### 6.3 Configuration Management (2 Points)

* 1 Point: Use environment variables for sensitive data:
    * API keys
    * Database credentials
    * Model paths
* 1 Point: Create configuration files:
    * Separate configs for development/production
    * YAML or JSON configuration files

#### 6.4 Error Handling and Robustness (2 Points)

* 1 Point: Implement comprehensive error handling:
    * Catch and handle exceptions gracefully
    * Provide helpful error messages
* 1 Point: Input validation and sanitization:
    * Validate data types and ranges
    * Handle edge cases

### Theoretical Questions (10 Points)

* 5 Points: What are the main challenges when deploying ML models to production? Discuss at least 5 challenges and how to address them.
* 3 Points: Explain the difference between a Model Training Pipeline and an Inference Pipeline. What components are unique to each?
* 2 Points: Why is model versioning important in production ML systems? How would you implement it?

### IMPORTANT NOTES

* You can use the dataset from assignment 3 (where you had imbalanced classes) for Part 1
* All code should be well-documented and follow PEP 8 style guidelines
* Include requirements.txt or environment.yml with all dependencies
* Your Docker container should be fully self-contained and runnable on any machine
* In your notebook and code files, please explain why you are doing each part (in natural language, Farsi or English). Also, you need to explain what you have gained/understood from that part. If you only provide code without the comments, *you will not get the full mark*.

You can use [Lab Session 13's notebook](https://colab.research.google.com/drive/1s0m2oZEmJpLlzi2z2Ifl7KzwJhH6TAGQ?usp=sharing) as a guideline for the Pipeline and Imbalanced Data sections.

### Submission Requirements

* Compiled Jupyter notebook for Parts 1 and 2
* GitHub repository link containing all code, Dockerfile, and documentation
* Video demonstration (3-5 minutes) showing:
    * Your API running in a Docker container
    * Testing API endpoints with sample requests
    * Explaining your XAI results

Please submit:

1. The *compiled* notebook (or the link to your *compiled* notebook on Google Colab/Github/Kaggle) **on the assignment page of Sharif CW**
2. Link to your GitHub repository
3. Link to your demo video (upload to Google Drive or YouTube)

**Due date:** Thursday, Khordad 20, 23:59

---

## General Notes for All Assignments

### Documentation and Explanation

**CRITICAL:** In your notebook, **per cell**, please explain why you are doing that part (in natural language, Farsi or English). Also, you need to explain what you have gained/understood from that part. If you only provide code without the comments, *you will not get the full mark*.

### Code Quality

* Write clean, readable, and well-commented code
* Follow Python best practices (PEP 8)
* Use meaningful variable names

### Submission Format

* Submit the *compiled* notebook (all cells executed)
* If using Colab/GitHub/Kaggle, ensure sharing settings are correct
* Always submit the `.ipynb` file

### Test Set

* All evaluations must be performed on the test set (20% of data)
* Never train on test data

### Cross-Validation

* Use K-Fold Cross Validation where specified
* Document your validation strategy

### Bonus Points

* Bonus points are optional but encouraged
* Send Kaggle certificates via email or Sharif Courseware

### Help and Resources

* Use provided lab notebooks as guidelines
* Refer to official documentation for libraries
* Ask questions on the course forum

**Submission Platform:** All assignments must be submitted on **Sharif Courseware (CW)** unless otherwise specified.

---

Due date: Thursday, Khordad 20, 23:59
{% endcomment %}
