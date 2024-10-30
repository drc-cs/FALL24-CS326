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
  ## L.11 Recommendation Systems

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

### H.04 Release

- H.04 will be released Tuesday, November 5.

### P.02

- P.02 will take place in-person on October 31 and November 5.

### Office Hours

- As updated in the [syllabus](https://github.com/drc-cs/FALL24-CS326), office hours are on Tuesdays from 2:30 - 3:15PM in Mudd 3510.

<!--s-->

<div class="header-slide">

# L.11 Recommendation Systems

</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Intro Poll
  ## On a scale of 1-5, how confident are you with the following recommendation modeling methods?

  1. Content-Based Filtering
  2. Collaborative Filtering

  Scan the QR code or go to [pollev.com/nucs](https://pollev.com/nucs)

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 100%; padding-top: 5%">
  <img src="https://storage.googleapis.com/slide_assets/PollEverywhere.png" width="50%">
  </div>
</div>

<!--s-->

## L.10 | Recommendation Systems

In today’s lecture, we will explore two common methods for building recommendation systems:

1. Content-Based Filtering
    - User Profile
    - Item Profile
    - Similarity Metrics
    - Evaluation
    - Pros and Cons
    
2. Collaborative Filtering
    - User-User
    - Item-Item
    - Filling in the Sparse Matrix
    - Pros and Cons

<!--s-->

## Recommendation Systems

Why build a recommendation system? 

- **Personalization**: Users are more likely to engage with a platform that provides personalized recommendations. Spotify absolutely excels at this.

- **Increased Engagement**: Users are more likely to spend more time on a platform that provides relevant recommendations. Instagram & TikTok are great examples of this.

- **Increased Revenue**: Users are more likely to purchase items that are recommended to them. According to a report by Salesforce, you can increase conversion rates for web products by 22.66% by using an intelligent recommendation system [[source]](https://brandcdn.exacttarget.com/sites/exacttarget/files/deliverables/etmc-predictiveintelligencebenchmarkreport.pdf).

<!--s-->

<div class="header-slide">

# Content-Based Filtering

</div>

<!--s-->

## Content-Based Filtering

Content-based filtering methods are based on a description of the **item** and a profile of the **user**’s preferences. A few quick definitions:

### User Matrix

- A set of data points that represents all user’s preferences. When a user interacts with an item, the user matrix is updated.

### Item Matrix

- A set of data points that represents every item’s content.
- Items can be represented as a set of features.
  - For books, these features could be genre, author, and publication date. More sophisticated models can use text analysis to extract features from the book's content.

### User-Item Similarity Matrix

- A set of data points that represents the similarity between user preferences and items.
- These similarities can be weighted by previous reviews or ratings from the user.

<!--s-->

## Content-Based Filtering | Basic Idea

The basic idea behind content-based filtering is to recommend items that are similar to those that a user liked in the past.

<img src="https://cdn.sanity.io/images/oaglaatp/production/a2fc251dcb1ad9ce9b8a82b182c6186d5caba036-1200x800.png?w=1200&h=800&auto=format" height="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Rosidi</p>

<!--s-->

## Content-Based Filtering | Intuitive Example & Question 

Let's say we have a user who has rated the following movies:

- The Shawshank Redemption: <span class="code-span">5</span>
- The Godfather: <span class="code-span">4</span>

And we want to know if they will enjoy:

- The Dark Knight: <span class="code-span">?</span>
- The Notebook: <span class="code-span">?</span>

We have the following information about the movies. They are encoded as an ordered, binary array of genres:

<span class="code-span">[ Drama, Crime, Thriller, Action, Romance]</span>

- The Shawshank Redemption: <span class="code-span">Drama, Crime, Thriller</span> | <span class="code-span"> [1, 1, 1, 0, 0]</span>
- The Godfather: <span class="code-span">Drama, Crime</span> | <span class="code-span"> [1, 1, 0, 0, 0]</span>
- The Dark Knight: <span class="code-span">Drama, Crime, Action</span> | <span class="code-span"> [1, 1, 0, 1, 0]</span>
- The Notebook: <span class="code-span">Drama, Romance</span> | <span class="code-span"> [1, 0, 0, 0, 1]</span>

Using content-based filtering, should we recommend (A) The Dark Knight or (B) The Notebook to the user?

<!--s-->

## Content-Based Filtering | Similarity Metrics

<div style="font-size: 0.8em;">

Commonly, similarity is determined simply as the dot product between two vectors:

$$ \textbf{Similarity} = A \cdot B $$

But exactly as represented by our KNN lecture, we can use different similarity metrics to calculate the similarity between two vectors.

</div>

<!--s-->

## Content-Based Filtering | Similarity / Distance Metrics

<div class = "col-wrapper">

<div class="c1" style = "width: 50%;">

### Euclidean Distance

`$ d(x, x') =\sqrt{\sum_{i=1}^n (x_i - x'_i)^2} $`

### Manhattan Distance

`$ d(x, x') = \sum_{i=1}^n |x_i - x'_i| $`

### **Cosine Distance**

`$ d(x, x') = 1 - \frac{x \cdot x'}{||x|| \cdot ||x'||} $`

</div>

<div class="c2" style = "width: 50%;">

### Jaccard Distance

`$ d(x, x') = 1 - \frac{|x \cap x'|}{|x \cup x'|} $`

### Hamming Distance

`$ d(x, x') = \frac{1}{n} \sum_{i=1}^n x_i \neq x'_i $`

</div>
</div>

<!--s-->

## Content-Based Filtering | Evaluation

Evaluation of content-based filtering depends on the application. Ideally, you would use a prospective study (A/B testing) to see if your recommendations are leading to increased user engagement!

<div class="col-centered">
<img src = "https://blog.christianposta.com/images/abtesting.png" style="border-radius:15px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Optimizely, 2024</p>
</div>

<!--s-->

## Content-Based Filtering | Pros and Cons

| Pros | Cons |
| --- | --- |
| Easy to implement. | Limited to the user’s past preferences. |
| No need for data on other users. | Limited to the item’s features. |
| Can recommend niche items. | Can overfit to the user’s preferences. |
| Can provide explanations for recommendations. | Cold-start problem. |

<!--s-->

## Question | Content-Based Filtering

Which of the following is an advantage of content-based filtering?


<div class='col-wrapper' style = 'display: flex; align-items: top; margin-top: 2em; margin-left: -1em;'>
<div class='c1' style = 'width: 60%; display: flex; align-items: center; flex-direction: column; margin-top: 2em'>
<div style = 'line-height: 2em;'>
&emsp;A. Easy to implement. <br>
&emsp;B. No need for data on other users. <br>
&emsp;C. Can recommend niche items. <br>
&emsp;D. All of the above. <br>
</div>
</div>
<div class='c2' style = 'width: 40%; display: flex; align-items: center; flex-direction: column;'>
<img src='https://storage.googleapis.com/slide_assets/PollEverywhere.png' width='100%'>
<a>poll.ev.com/nucs</a>
</div>
</div>

<!--s-->

<div class="header-slide">

# Collaborative Filtering

</div>

<!--s-->

## Collaborative Filtering

### User-User Collaborative Filtering

Based on the idea that users who have "agreed in the past will agree in the future".

### Item-Item Collaborative Filtering

Based on the idea that item reviews are often grouped together.

<img src="https://www.nvidia.com/content/dam/en-zz/Solutions/glossary/data-science/recommendation-system/img-2.png" height="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">NVIDIA, 2024</p>

<!--s-->

## Collaborative Filtering | User-Item Matrix

The User-Item Matrix is a matrix that represents the relationship between users and items.

You can find nearest neighbors from two different perspectives: user-user (rows) or item-item (cols).

<div class="col-centered">
<img src="https://storage.googleapis.com/slide_assets/user-item-matrix.png" height="50%" style="margin: 0 auto; display: block; border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Hashemi, 2020</p>
</div>

<!--s-->

## Collaborative Filtering | User-User

User-User Collaborative Filtering is based on the idea that users who have agreed in the past will agree in the future.

1. Create a User-Item Matrix that contains the user’s reviews of items.
2. Find users who are similar to the target user (User-User)
3. Recommend items that similar users have liked.

----------------

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

<img src="https://storage.googleapis.com/slide_assets/user-item-matrix.png" height="50%" style="margin: 0 auto; display: block; border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Hashemi, 2020</p>

</div>
<div class="c2" style = "width: 50%">

<img src="https://www.nvidia.com/content/dam/en-zz/Solutions/glossary/data-science/recommendation-system/img-2.png" height="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">NVIDIA, 2024</p>

</div>
</div>

<!--s-->

## Collaborative Filtering | Item-Item

Item-Item Collaborative Filtering is based on the idea that users who each liked an item will like similar items. It is **different** from content-based filtering because it does not have anything to do with the item characteristics.

1. Create a User-Item Matrix that represents the user’s reviews of items.
2. Find items that are often "grouped" together (Item-Item)
3. Recommend these similar items to the target user.

-------

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

<img src="https://storage.googleapis.com/slide_assets/user-item-matrix.png" height="50%" style="margin: 0 auto; display: block; border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Hashemi, 2020</p>

</div>
<div class="c2" style = "width: 50%">

<img src="https://miro.medium.com/v2/resize:fit:801/1*skK2fqWiBF7weHU8SjuCzw.png" height="50%" style="margin: 0 auto; display: block; border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Tomar, 2017</p>

</div>
</div>

<!--s-->
## Collaborative Filtering
### Filling in the User-Item Matrix

The user-item matrix is often missing many values. This is because users typically only interact with a small subset of items. This presents a problem with finding nearest neighbors. 

So, how do we fill in the gaps?

<!--s-->

## Collaborative Filtering
### Filling in the User-Item Matrix with the Scale's Mean

A simple way to fill in the gaps in the user-item matrix is to use the scale's mean. Typically you will center the data on the mean of the scale, and fill in the gaps with zero.

In the example below, the scale is from 1-5. The mean of the scale is 3. So, we subtract 3 from every existing score and replace the rest with 0.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### Sparse

|  | I.1 | I.2 | I.3 |
| --- | --- | --- | --- |
| U.1 | 5 | ... | 3 |
| U.2 | 0 | 4 | ... |
| U.3 | 3 | 2 | 0 |

</div>
<div class="c2" style = "width: 50%">

### Dense

|   | I.1 | I.2 | I.3 |
| --- | --- | --- | --- |
| U.1 | 2 | 0 | 0 |
| U.2 | -3 | 1 | 0 |
| U.3 | 0 | -1 | -3 |


</div>
</div>

<!--s-->

## Collaborative Filtering

### Filling in the User-Item Matrix with Matrix Factorization

Matrix Factorization is a technique to break down a matrix into the product of multiple matrices. It is used in collaborative filtering to estimate the missing values in the user-item matrix, and is often performed with alternating least squares (ALS).

$$ R \approx P \cdot Q^T $$

Where: 

- $R$ is the user-item matrix.
- $P$ is the user matrix.
- $Q$ is the item matrix.
- $K$ is the number of latent features.

<img src="https://www.nvidia.com/content/dam/en-zz/Solutions/glossary/data-science/recommendation-system/img-6.png" height="30%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">NVIDIA, 2024</p>

<!--s-->

## Collaborative Filtering

**Alternating Least Squares** is an optimization algorithm that is used to minimize the error between the predicted and actual ratings in the user-item matrix. There are many ways to solve for $P$ and $Q$! The example shown below is using the closed-form solution.

<div class = "col-wrapper">

<div class="c1" style = "width: 50%">

### Steps

1. Initialize the user matrix $P$ and the item matrix $Q$ randomly.

2. Fix $Q$ and solve for $P$.

3. Fix $P$ and solve for $Q$.

4. Repeat steps 2 and 3 until the error converges.

</div>

<div class="c2" style = "width: 50%">

### Example

$$ P = (Q^T \cdot Q + \lambda \cdot I)^{-1} \cdot Q^T \cdot R $$
$$ Q = (P^T \cdot P + \lambda \cdot I)^{-1} \cdot P^T \cdot R $$

Where:

- $R$ is the user-item matrix.
- $P$ is the user matrix.
- $Q$ is the item matrix.
- $\lambda$ is the regularization term.
- $I$ is the identity matrix.

<div style = "font-size: 0.6em;">

\**Note: You need to include the regularization term to prevent overfitting.*

</div>

<!--s-->

## Collaborative Filtering | Pros and Cons

| Pros | Cons |
| --- | --- |
| Can recommend items that the user has not seen before | Cold-start problem for both users and items |
| Can recommend items that are popular among similar users | Missing values in the user-item matrix |

<!--s-->

## Question | Collaborative Filtering

Which of the following is an advantage of **user-user collaborative** filtering?


<div class='col-wrapper' style = 'display: flex; align-items: top; margin-top: 2em; margin-left: -1em;'>
<div class='c1' style = 'width: 70%; display: flex; align-items: center; flex-direction: column; margin-top: 2em'>
<div style = 'line-height: 2em; font-size: 0.8em;'>
&emsp;A. Can recommend very different items that the user has not seen before. <br>
&emsp;B. Tends to have a full matrix. <br>
&emsp;C. No need for data on other users. <br>
&emsp;D. All of the above. <br>
</div>
</div>
<div class='c2' style = 'width: 30%; display: flex; align-items: center; flex-direction: column;'>
<img src='https://storage.googleapis.com/slide_assets/PollEverywhere.png' width='100%'>
<a>poll.ev.com/nucs</a>
</div>
</div>

<div class="header-slide">

# Conclusion

</div>

<!--s-->

## Conclusion

In today’s lecture, we explored:

1. Content-Based Filtering
2. Collaborative Filtering

Recommendation systems are a powerful tool for personalizing user experiences and increasing user engagement.

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Exit Poll
  ## On a scale of 1-5, how confident are you with the following recommendation modeling methods?

  1. Content-Based Filtering
  2. Collaborative Filtering


  Scan the QR code or go to [pollev.com/nucs](https://pollev.com/nucs)

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 100%; padding-top: 5%">
  <img src="https://storage.googleapis.com/slide_assets/PollEverywhere.png" width="50%">
  </div>
</div>

<!--s-->

<div class="header-slide">

# Project Time

<iframe src="https://lottie.host/embed/bd6c5b65-d724-4f97-882c-40f58367ea38/BIKhZdSeqW.json" height="100%" width = "100%"></iframe>

</div>