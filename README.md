# Amazon E-Commerce Analysis

This repository contains our final project for the **Data Mining (2024–2025)** course. The objective of the project is to analyze Amazon product data across multiple categories through a pipeline that integrates preprocessing, exploratory analysis, and a series of machine learning tasks.

## Project Structure

The project is divided into two main parts:

### Part 1: Data Exploration and Feature Engineering

We performed standard preprocessing tasks including text cleaning, metadata selection, and basic feature extraction. We visualized trends such as rating distributions, top products by review count, and time-based shifts in user feedback. Word clouds were used to highlight frequently occurring terms in low-rated popular products. Although we implemented a sentiment analysis class for combining textual sentiment with numerical ratings, it was not used in later tasks as it was only part of the assignment's requirements.

### Part 2: Learning Tasks

This section includes our applied machine learning work, using the processed datasets for modeling and evaluation.

#### Clustering  
We applied k-means clustering to group similar products within each category. The feature space included numerical metadata and text-based features. To determine the number of clusters, we used the elbow method and silhouette scores as evaluation tools. These helped us balance between under- and over-segmentation of product groups.

#### Recommendation Systems  
We built and evaluated several types of recommendation models:
- **Collaborative Filtering** using both user-based and item-based approaches  
- **Content-Based Filtering**, using Word2Vec embeddings of product descriptions  
- **Hybrid Models**, integrating collaborative and content-based predictions  
- **Random Baseline**, used for performance comparison

All methods were evaluated per user using recall at fixed top-k recommendation lists.

#### Sentiment Classification  
We constructed models to classify the sentiment of reviews into discrete categories. Both traditional classifiers and a BERT-based deep learning model were evaluated. Due to class imbalance, we used the macro F1-score as the main metric. Additional visual analyses included learning curves, confusion matrices, and ROC curves.

#### Frequent Pattern Mining  
#### Frequent Pattern Mining  
We applied the Apriori algorithm separately for each year and category, extracting frequent itemsets based on co-reviewed products. This revealed how product associations evolve over time within each domain.


## Dataset

We used selected subsets of the [Amazon Product Dataset](http://deepyeti.ucsd.edu/jianmo/amazon/index.html), focusing on:
- Books  
- All Beauty  
- Electronics  
- Movies and TV  
- Sports and Outdoors  

Each dataset includes user reviews and product metadata, enabling a range of exploration and modeling approaches.

## Team

- **[Vaggelis Argyropoulos](https://github.com/Vaggelis-Arg)**
- **[Christos Vladikas](https://github.com/chrisvl11)**
- **[Antonis Kalatzis](https://github.com/tonykalantzis)**

---

This project provides a full data mining workflow over large-scale e-commerce data, combining exploratory techniques, unsupervised and supervised learning, recommendation modeling, and pattern mining, with attention to evaluation and interpretability throughout.
