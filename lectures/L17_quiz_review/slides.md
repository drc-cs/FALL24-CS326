---
title: COMP_SCI 326
separator: <!--s-->
verticalSeparator: <!--v-->
theme: serif
revealOptions:
  transition: 'none'
---

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 70%; position: absolute;">

  # Introduction to Data Science Pipelines
  ## L.17 | Quiz Review

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 80%; padding-top: 30%">
  <iframe src="https://lottie.host/embed/bd6c5b65-d724-4f97-882c-40f58367ea38/BIKhZdSeqW.json" height="100%" width = "100%"></iframe>
  </div>
</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 70%; position: absolute;">

  # Welcome to CS 326.
  ## Please check in using PollEverywhere.
  Scan the QR code or go to [pollev.com/nucs](https://pollev.com/nucs)

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 100%; padding-top: 5%">
  <img src="https://storage.googleapis.com/slide_assets/PollEverywhere.png" width="50%">
  </div>
</div>

<!--s-->

## Announcements

- H.04 is due **tonight** 11.19.2024.
- P.04 (Written Report) will be due on 12.03.2024.
- P.03 (Presentations) will take place 12.03.2024 / 12.05.2024.

<!--s-->

<div class="header-slide">

# Quiz Review

</div>

<!--s-->

# Lockdown Browser (Canvas)

1. [Download Lockdown Browser](https://download.respondus.com/lockdown/download7.php?id=171646780).
2. Install the browser on your computer.
3. Ensure that you can log in to your Northwestern University Canvas account using the browser.
4. Take the [practice](https://canvas.northwestern.edu/courses/217294/quizzes/252735) quiz on Canvas to ensure that everything is working correctly.

You will then be able to access the quiz on Canvas and take the quiz on your computer.

**Note**: I have enabled the Monitoring feature for students that are unable to join us in-person, but I ask that everyone use it to ensure a fair testing environment.

<!--s-->

<div class="header-slide">

# Quiz Review

</div>

<!--s-->

# L.02 | Data Sources

[[slides]](https://drc-cs.github.io/FALL24-CS326/lectures/L02_data_sources/#/)

- Be able to identify structured, semi-structured, and unstructured data, as well as the advantages and disadvantages of each.

- Given a scenario with missing data, pick the appropriate method to handle it.

- Be able to describe methods for identifying outliers in a dataset.

<!--s-->

# H.01 | Hello World

[[homework]](https://github.com/drc-cs/FALL24-CS326/tree/main/homeworks/H01)

- Understand the roles of conda, GitHub, vscode, and pytest in your development workflow.

<!--s-->

# L.03 | Exploratory Data Analysis

[[slides]](https://drc-cs.github.io/FALL24-CS326/lectures/L03_eda/#/)

- Identify skew (positive, negative).

- Identify kurtosis (leptokurtic, mesokurtic, platykurtic).

- Know key properties of a normal distribution.

<!--s-->

# L.04 | Correlation

[[slides]](https://drc-cs.github.io/FALL24-CS326/lectures/L04_correlation_association/#/)

- Differentiate when to use Pearson or Spearman correlation, and how to interpret their results (negative / positive / no relationship).

- Identify a scenario as Simpson's Paradox (or not).

<!--s-->

# L.05 & H.02 | Hypothesis Testing

[[slides]](https://drc-cs.github.io/FALL24-CS326/lectures/L05_hypothesis_testing/#/)
[[homework]](https://github.com/drc-cs/FALL24-CS326/tree/main/homeworks/H02)

- Construct an A/B Test to test a hypothesis.

- Define hypothesis testing for a scenario in terms of $H_0$ and $H_1$.

- Provided a scenario, identify the hypothesis test to use (t-test, paired t-test, chi-squared test, anova).

- Know the non-parametric analogs to the tests we covered in lecture.

<!--s-->

# L.06 | Data Preprocessing

[[slides]](https://drc-cs.github.io/FALL24-CS326/lectures/L06_data_preprocessing/#/)

- Define feature engineering in the context of machine learning applications.

- Define and be able to identify data that has been scaled (min-max, standard).

- Describe the curse of dimensionality and how it affects machine learning models.

- Choose a method for dimensionality reduction (Feature Selection, Feature Sampling, Random Projection, or PCA) and detail how it works.

<!--s-->

# L.07 | Machine Learning I

[[slides]](https://drc-cs.github.io/FALL24-CS326/lectures/L07_supervised_machine_learning_i/#/)

- Define the terms: training set, validation set, and test set and their primary uses.

- Identify a scenario as a classification or regression problem.

- Explain the KNN algorithm and how it works.

- Explain where the normal equation for linear regression comes from.

- Be able to identify L1 and L2 regularization and explain at a high level how they work.

- Understand the intuition behind the cross-entropy loss function.

<!--s-->

# H.03 | Machine Learning

[[homework]](https://github.com/drc-cs/FALL24-CS326/tree/main/homeworks/H03)

- Be able to look at code for logistic regression gradient descent and identify missing or incorrect components.

- Provided with a **simple** numpy operation, identify the shape of the output. This may include an axis argument. [[🔗]](https://numpy.org/doc/stable/user/basics.broadcasting.html)

<!--s-->

# L.08 | Machine Learning II

[[slides]](https://drc-cs.github.io/FALL24-CS326/lectures/L08_supervised_machine_learning_ii/#/)

- Explain ROC curves (axes) and what the AUC represents.

- Explain the value of k-fold cross-validation.

- Explain the value of a softmax function in the context of a multi-class classification problem.

<!--s-->

# L.09 | Machine Learning III

[[slides]](https://drc-cs.github.io/FALL24-CS326/lectures/L09_supervised_machine_learning_iii/#/)

- Explain the ID3 algorithm and how it works (understand entropy & information gain).

- Be able to identify a decision tree model as overfitting or underfitting.

- Differentiate between and be able to explain different ensemble modeling methods (bagging, boosting, stacking).

<!--s-->

# L.10 & H.04 | Clustering

[[slides]](https://drc-cs.github.io/FALL24-CS326/lectures/L10_unsupervised_machine_learning/#/)
[[homework]](https://github.com/drc-cs/FALL24-CS326/tree/main/homeworks/H03)

- Explain the intuition behind partitional clustering / K-means algorithm (and know where it fails!).

- Explain the intuition behind density-based clustering / DBScan algorithm (and know where it fails!).

<!--s-->


# L.11 | Recommendation Systems

[[slides]](https://drc-cs.github.io/FALL24-CS326/lectures/L11_recommendation_modeling/#/)

- Differentiate between content and collaborative filtering.

- Explain the intuition behind matrix factorization in collaborative filtering.

<!--s-->

# L.14 | Time Series

[[slides]](https://drc-cs.github.io/FALL24-CS326/lectures/L14_time_series_analysis/#/)

- Identify additive vs multiplicative decomposition in time series data.

- Know the value of differencing in time series data (i.e. what does it do, and why is that important?).

- Look at ACF / PACF plots and determine what order of AR or MA to use in an ARIMA model.

- Explain walk-forward validation in the context of time series data.

<!--s-->

# L.15 | Natural Language Processing

[[slides]](https://drc-cs.github.io/FALL24-CS326/lectures/L15_natural_language_processing_i/#/)

- Explain TF-IDF and how it is used in text analysis.

- Explain perplexity, bleu, and rouge in the context of text analysis.

<!--s-->

# L.16 | Natural Language Processing

[[slides]](https://drc-cs.github.io/FALL24-CS326/lectures/L16_natural_language_processing_ii/#/)

- Explain the benefits of word embeddings over one-hot encoding.

- Differentiate between a CBOW and Skip-Gram model.

- Explain what a vector database is and how it can be used in a RAG application.

- Explain the benefits of RAG over just-LLM chatbots.

<!--s-->

<div class="header-slide">

# Project Time

<iframe src="https://lottie.host/embed/bd6c5b65-d724-4f97-882c-40f58367ea38/BIKhZdSeqW.json" height="100%" width = "100%"></iframe>

</div>