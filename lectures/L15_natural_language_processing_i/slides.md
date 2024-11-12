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
  ## L.15 Natural Language Processing I

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 80%; padding-top: 30%">
  <iframe src="https://lottie.host/embed/bd6c5b65-d724-4f97-882c-40f58367ea38/BIKhZdSeqW.json" height="100%" width = "100%"></iframe>
  </div>
</div>

<!--s-->

<div class="header-slide">

# L.15 Natural Language Processing I

</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Intro Poll
  ## On a scale of 1-5, how confident are you with the following methods:

- Regular Expressions
- Tokenization
- Bag of Words (BoW)
- Term Frequency-Inverse Document Frequency (TF-IDF)
- Markov Chains
- Evaluation of Text Generation Models

Scan the QR code or go to [pollev.com/nucs](https://pollev.com/nucs)

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 100%; padding-top: 5%">
  <img src="https://storage.googleapis.com/slide_assets/PollEverywhere.png" width="50%">
  </div>
</div>

<!--s-->

## Introduction to NLP | Definition and Importance

Natural Language Processing (NLP) enables computers to understand, interpret, and generate human language.

NLP combines computational linguistics with machine learning to process and analyze large amounts of natural language data.

<!--s-->

## NLP Applications

<div class="col-wrapper">

<div class="c1" style="width: 50%; margin: 0; padding: 0;">

### Applications of NLP

- **Chatbots**: Automate customer service interactions.

- **Sentiment Analysis**: Determine sentiment from text data.

- **Machine Translation**: Translate text from one language to another.

</div>

<div class="c2 col-centered" style="width: 50%">

<div>

<img src="https://amazinum.com/wp-content/uploads/2022/10/blog-NLP-pic1.png" width="400" style="margin: 0; padding: 0; display: block;">

<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">Amazinum 2024</span>

</div>

</div>

</div>

<!--s-->

## NLP Impact on Society

NLP is transforming industries by automating tasks, improving data analysis, and enhancing communication. It impacts our daily lives through digital assistants, real-time translation, and personalized content delivery.

- Initial attempts involved rule-based systems (if-then statements).

- Classical methods use statistics (Naive Bayes).

- Deep learning techniques (combined with statistics) have emerged in recent years (LSTM/transformers).

<!--s-->

<div class="header-slide">

# Regular Expressions

</div>

<!--s-->

## Regular Expressions

Regular expressions (regex) are a powerful tool for working with text data. They allow us to search, match, and manipulate text using a concise and expressive syntax.

You may feel compelled to do basic sting manipulation with Python's built-in string methods. However, regular expressions are much more powerful and flexible. Consider the following example:

> "My phone number is (810) 555-1234."

<!--s-->

## Regular Expressions | Example

> "My phone number is (810)555-1234"

### String Methods

<span class="code-span">phone_number = text.split(" ")[-1]</span>

This method would work for the given example but **not** for "My phone number is (810) 555-1234. Call me!"
or "My phone number is (810) 555-1234. Call me! It's urgent!"

### Regular Expression

<span class="code-span">phone_number = re.search(r'\(\d{3}\)\d{3}-\d{4}', text).group()</span>

This regular expression will match any phone number in the format (810)555-1234, including the additional text above.

<!--s-->

## Regular Expressions | Syntax

Regular expressions are a sequence of characters that define a search pattern. They are used to search, match, and manipulate text strings.

<div class="col-wrapper" style="font-size: 0.7em;">

<div class="c1" style="width: 50%; margin: 0; padding: 0;">

- **Literals**: Characters that match themselves.
  - `a` matches the character "a".
  - `123` matches the sequence "123".

- **Metacharacters**: Special characters that represent a class of characters.
  - `.` matches any character except a newline.
  - `^` matches the start of a string.
  - `$` matches the end of a string.

- **Quantifiers**: Specify the number of occurrences of a character.
  - `*` matches zero or more occurrences of the preceding character
  - `+` matches one or more occurrences.
  - `?` matches zero or one occurrence.

</div>

<div class="c2" style="width: 50%">

- **Character Classes**: Define a set of characters to match.
  - `[abc]` matches any character in the set.
  - `[^abc]` matches any character not in the set.
  - `[a-z]` matches any character in the range.
  - `\d` matches any digit.
  - `\w` matches any word character.
  - `\s` matches any whitespace character.

- **Groups**: Define a group of characters to match.
  - `(abc)` matches the sequence "abc".
  - `(a|b)` matches "a" or "b".
  - `(?:abc)` matches "abc" but does not capture it.

</div>

<!--s-->

## Regular Expressions

Want to practice or make sure your expression works? 

Live regular expression practice: https://regex101.com/

<!--s-->

<div class="header-slide">

# Tokenization

</div>

<!--s-->

## Tokenization

Tokenization is the process of breaking text into smaller units, such as sub-words or words. It is a fundamental step in NLP tasks.

<div class="col-wrapper">

<div class="c1" style="width: 50%; margin: 0; padding: 0;">

- **Character Tokenization**: Splitting text into individual characters.

- **Word Tokenization**: Splitting text into individual words or n-grams.

- **Tokenization Libraries**: NLTK, spaCy, scikit-learn.

</div>

<div class="c2" style="width: 50%">

<div>

<img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/0*VymfbxfPtuH3MpJl.png" width="400" style="margin: 0; padding: 0; display: block;">

<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">Khandelwal 2023</span>

</div>

</div>

</div>

🔥 *NLTK is awesome. It's a great library for beginners!*

<!--s-->

## W.S.I.C.A.X.I.T.E.O.D.L.O.H.D.I.R.T.D.L.?

Modern LLMs use a special type of tokenization called byte pair encoding (BPE). We'll talk more about byte pair encoding specifically on Thursday when we build our R.A.G. system. 

<!--s-->

<div class="header-slide">

# Stemming and Lemmatization

</div>

<!--s-->

## Stemming and Lemmatization

Stemming and lemmatization are text normalization techniques that reduce words to their base or root form. They are used to improve text processing and analysis by reducing the vocabulary size.

<div class="col-wrapper">

<div class="c1" style="width: 50%; margin: 0; padding: 0;">

### Stemming

*Reduces words to their base or root form.*

Stemming examples include "running" to "run" and "requested" to "request."

Some disadvantages include over-stemming and under-stemming -- for example, "better" to "bett" instead of "good."

</div>

<div class="c2" style="width: 50%">

### Lemmatization

*Reduces words to their base form using a dictionary.*

Lemmatization examples include "better" to "good" and "requested" to "request."

Some disadvantages include increased computational complexity and slower processing times, and often requires a dictionary or corpus.

</div>

<!--s-->

## W.S.I.C.A.X.I.T.E.O.D.L.O.H.D.I.R.T.D.L.?

In the era of deep learning, stemming and lemmatization are less common due to the use of word embeddings and subword tokenization. Consider the power of word embeddings and the massive problem they solved here!

We'll talk more about word embeddings in the next lecture, but it's important to have an appreciation for old-school so you know what you have. :p

<!--s-->

<div class="header-slide">

# Stop Words

</div>

<!--s-->

## Stop Words

Stop words are common words that are filtered out during text processing to reduce noise and improve performance. They are typically removed before tokenization.

**Examples** include <span class="code-span">and</span> / <span class="code-span">the</span> / <span class="code-span">or</span>

<!--s-->

## W.S.I.C.A.X.I.T.E.O.D.L.O.H.D.I.R.T.D.L.?

Attention mechanisms in transformers (presumably) learn to ignore stop words.

<!--s-->

## Example Sentence Preprocessing

<div style = "font-size: 0.8em;">

Consider the following sentence:

> "The quick brown fox jumps over the lazy dog."

Tokenized: 

> ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]

Lemmatized:

> ["The", "quick", "brown", "fox", "jump", "over", "the", "lazy", "dog"]

Stop Words Removed:

> ["quick", "brown", "fox", "jump", "lazy", "dog"]

</div>

<!--s-->

<div class="header-slide">

# Classical NLP Methods

</div>

<!--s-->

## Bag of Words (BoW)

BoW represents text by counting the occurrences of each word, ignoring grammar and order. It's simple but lacks context and semantic understanding.

<img src="https://storage.googleapis.com/slide_assets/bow.png" width="100%" style="margin: 0; padding: 0; display: block; border-radius: 10px;">

<!--s-->

## Term Frequency-Inverse Document Frequency (TF-IDF)

TF-IDF is a statistical measure that evaluates the importance of a word in a document relative to a corpus. It is used to represent text data as a vector.

<div class="col-wrapper">

<div class="c1" style="width: 50%; margin: 0; padding: 0; font-size: 0.8em;">

### TF-IDF

- **Term Frequency (TF)**: Frequency of a term in a document.

- **Inverse Document Frequency (IDF)**: Importance of a term across a corpus.

</div>

<div class="c2" style="width: 40%; font-size: 0.8em;">

### Calculation

$$ \text{TF-IDF} = \text{TF} \times \text{IDF} $$
$$ \text{IDF} = \log \left( \frac{N}{n} \right) $$
$$ \text{TF} = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}} $$

Where:
  - $f_{t,d}$ is the frequency of term $t$ in document $d$.
  - $N$ is the total number of documents.
  - $n$ is the number of documents containing term $t$.
  - $t'$ is a term in document $d$.
  - $d$ is a document.

</div>

</div>

<!--s-->

## W.S.I.C.A.X.I.T.E.O.D.L.O.H.D.I.R.T.D.L.?

While the field of NLP has seen significant advancements with deep learning, classical models remain relevant for many tasks. These models are interpretable, computationally efficient, and require less data.

In fields where data is scarce or interpretability is crucial, classical models can even be preferred. Most text tasks can be framed into two categories: **text classification** and **text generation**.

<!--s-->

## Text Classification | K-Nearest Neighbors

Once you have a fixed-length vector representation of text, you can use K-Nearest Neighbors (KNN) to classify text by comparing document vectors.

<div class="col-wrapper">

<div class="c1" style="width: 50%; margin: 0; padding: 0;">

### Concept

Classifies text by comparing document vectors.

### Text Representation

Uses BoW or TF-IDF vectors (or more recently, word embeddings).

</div>

<div class="c2" style="width: 50%">

<div>

<img src="https://storage.googleapis.com/slide_assets/bow.png" width="400" style="margin: 0; padding: 0; display: block;">

<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">CITATION</span>

</div>

</div>

</div>

<!--s-->

## Text Generation | Markov Chains

Markov Chains are probabilistic models used for generating text. By modeling the context of words with historical patterns, Markov chains can simulate text generation processes.

<div class="col-centered">
<img src="https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2F2.bp.blogspot.com%2F-U2fyhOJ7bN8%2FUJsL23oh3zI%2FAAAAAAAADRs%2FwZNWvVR-Jco%2Fs1600%2Ftext-markov.png" width="500" style="margin: 0; padding: 0; display: block; border-radius: 10px;">
<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">awalsh128.blogspot.com</span>
</div>

<!--s-->

## Text Generation | Markov Chains

In the context of text generation, a Markov chain uses a finite set of states (words) and transitions between these states based on probabilities.

### Key Elements

- **States**: Words or tokens in a text.

- **Transition Probabilities**: The probability of moving from one word to another. 

- **Order**: Refers to how many previous states (words) influence the next state.

<!--s-->

## Text Generation | Markov Chains

Consider a simple first-order Markov Chain, which uses the current word to predict the next word.


### Transition Matrix

A transition matrix represents the probabilities of transitioning from each word to possible subsequent words.

### Markov Process (First Order)

$$ P(w_{t+1} | w_t) = P(w_{t+1} | w_t, w_{t-1}, \ldots, w_1) $$

tldr; the probability of the next word depends only on the current word.


<!--s-->

## Text Generation | Markov Chains

Let's say we have the following text:

> "The quick brown fox jumps over the lazy dog."

|  | The | quick | brown | fox | jumps | over | lazy | dog |
|--------------|-------|---------|---------|-------|---------|--------|--------|-------|
| The        | 0.0   | 0.5     | 0.0     | 0.0   | 0.0     | 0.0    | 0.5    | 0.0   |
| quick      | 0.0   | 0.0     | 1.0     | 0.0   | 0.0     | 0.0    | 0.0    | 0.0   |
| brown      | 0.0   | 0.0     | 0.0     | 1.0   | 0.0     | 0.0    | 0.0    | 0.0   |
| fox        | 0.0   | 0.0     | 0.0     | 0.0   | 1.0     | 0.0    | 0.0    | 0.0   |
| jumps      | 0.0   | 0.0     | 0.0     | 0.0   | 0.0     | 1.0    | 0.0    | 0.0   |
| over       | 0.0   | 0.0     | 0.0     | 0.0   | 0.0     | 0.0    | 1.0    | 0.0   |
| lazy       | 0.0   | 0.0     | 0.0     | 0.0   | 0.0     | 0.0    | 0.0    | 1.0   |
| dog        | 0.0   | 0.0     | 0.0     | 0.0   | 0.0     | 0.0    | 0.0    | 0.0   |

Using these probabilities, you can generate new text by predicting each subsequent word based on the current word.

<!--s-->

## Text Generation | Markov Chains

Increasing the order allows the model to depend on more than one preceding word, creating more coherent and meaningful sentences.

### Second-Order Markov Chain Example

Given bi-grams:

> "The quick", "quick brown", "brown fox", "fox jumps", "jumps over", "over the", "the lazy", "lazy dog"

The transition probability now depends on pairs of words:

$$ P(w_3 | w_1, w_2) $$

This provides better context for the generated text, but can also reduce flexibility and introduction of new combinations.

<!--s-->

<div class="header-slide">

# Evaluation of Text Generation Models

</div>

<!--s-->

## Evaluating Text Generation Models

Text generation models are evaluated based on their ability to predict the next word in a sentence. With natural language processing, especially with more advanced models, there are several metrics used to evaluate the quality of generated text:

- **Perplexity**: Measures how well a probability model predicts a sample.
- **BLEU Score**: Measures the quality of machine-translated text.
- **ROUGE Score**: Measures the quality of text summarization.

<!--s-->

## Evaluation | Perplexity

Perplexity measures how well a probability model predicts a sample. A lower perplexity indicates better predictive performance. Perplexity is defined as: 

$$ \text{Perplexity} = 2^{-\frac{1}{N} \sum_{i=1}^{N} \log_2 P(w_i | w_{i-1})} $$

Where:
- $ N $ is the number of words in the sample.
- $ P(w_i | w_{i-1}) $ is the probability of word $w_i $ given the previous word $w_{i-1}$.

Intuitively, perplexity measures how surprised a model is by the next word in a sentence.

<!--s-->

## Evaluation | BLEU Score

BLEU is a metric used to evaluate the quality of machine-translated text. It compares the machine-translated text to a reference translation. The BLEU score ranges from 0 to 1.

$$ \text{BLEU} = \text{BP} \times \exp \left( \sum_{n=1}^{N} w_n \log p_n \right) $$

Where:
- $ \text{BP} $ is the brevity penalty.
- $ w_n $ is the weight for n-grams.
- $ p_n $ is the precision for n-grams.

BLEU is more precise and conservative, focusing on word accuracy in translation with a focus on precision.

<!--s-->

## Evaluation | ROUGE Score

ROUGE is a metric used to evaluate the quality of text summarization. It compares the generated summary to a reference summary. The ROUGE score ranges from 0 to 1.

$$ \text{ROUGE} = \frac{\text{Number of Overlapping Words}}{\text{Total Number of Words in Reference Summary}} $$

ROUGE focuses on recall (or the number of overlapping words) in summarization.

<!--s-->


<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Exit Poll
  ## On a scale of 1-5, how confident are you with the following methods:

- Regular Expressions
- Tokenization
- Bag of Words (BoW)
- Term Frequency-Inverse Document Frequency (TF-IDF)
- Markov Chains
- Evaluation of Text Generation Models

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

