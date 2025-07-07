# **Amazon-Ecommerce-Analysis**

A data mining project analyzing Amazon product reviews, covering sentiment analysis, recommendation systems, and clustering using Python and machine learning. Processes JSON data, extracts insights, and builds ML models for e-commerce trends.


## **Team Information**


### Team member #1: **Vaggelis Argyropoulos**
* Github: [**Vaggelis-Arg**](https://github.com/Vaggelis-Arg)
* ID: sdi2200010
* Email: sdi2200010@di.uoa.gr
---
### Team member #2: **Christos Vladikas**
* Github: [**chrisvl11**](https://github.com/chrisvl11)
* ID: sdi2200020
* Email: sdi2200020@di.uoa.gr
---
### Team member #3: **Antonis Kalatzis**
* Github: [**tonykalantzis**](https://github.com/tonykalantzis) 
* ID: sdi2100046
* Email: sdi2100046@di.uoa.gr
---


### **Task 1: Data Exploration and Feature Engineering**


#### **Dataset Preparation**

---
**Question**: 

*Extract data for any five categories that you like (e.g., Electronics, Books, Home & Kitchen). Visit the dataset link and download only the JSON files for the categories that you plan to use. Parse the JSON files and create the csv file(s) that you are going to use for the rest of the Tasks (For the 5 product categories selected by each student, create 5 different CSV files.). The dataset is quite large, even for the 5 categories. Start with a smaller subset by limiting the numbers of rows downloaded per category, making sure your code works, before getting more rows for the analysis part of this project. Clean the data by handling missing values, normalizing prices, and preprocessing text (more on the text preprocessing techniques in Part2).*

---

**Answer:**

We used the `datasets` library from Hugging Face with `streaming=True` to load and process review data efficiently for five selected Amazon product categories:

- **Books**
- **All_Beauty**
- **Electronics**
- **Movies_and_TV**
- **Sports_and_Outdoors**

For each category:
- Up to **120,000 rows** were streamed and converted to Pandas DataFrames.
- Reviews and metadata were saved separately into two CSV files:
  - `{category}_reviews.csv`
  - `{category}_metadata.csv`

This design ensured modular preprocessing and low memory usage.

**Selected Fields:**

- **Review Columns:**  
    | rating | title | text | parent_asin | asin | user_id | timestamp | helpful_vote | verified_purchase |
    |--------|-------|------|-------------|------|---------|-----------|---------------|--------------------|
    | ⭐ 1–5 | Review title | Review body text | Product group ID | Product ID | Reviewer ID | Date (UNIX) | Upvotes | True/False |

<br>

- **Metadata Columns:**  
    | parent_asin | main_category | title | average_rating | rating_number | description | price | categories | bought_together | store |
    |-------------|----------------|-------|-----------------|----------------|-------------|-------|------------|------------------|-------|
    | Product group ID | Top-level category | Product name | Avg. rating across reviews | Total number of ratings | Product description | Price (float/string) | Category hierarchy | Commonly co-purchased items | Seller/store name |

**Notes:**

This pipeline minimizes manual intervention and download overhead, making it **scalable** to other categories. Streaming also enabled fast iteration and ensured that only relevant fields were retained for downstream tasks like clustering, classification, and recommendation.

---


#### Text and Metadata Preprocessing

To prepare the dataset for downstream tasks like sentiment analysis and recommendation, we implemented a comprehensive text and metadata cleaning pipeline.

The key steps include:

- **Downloading required NLTK resources**, such as tokenizers, stopwords, and sentiment lexicons.
- **Expanding contractions** (e.g., "don't" → "do not").
- **Replacing special tokens**:
  - URLs → `link`
  - Mentions/hashtags → `tag`
  - Emails → `mail`
  - Numbers → `number`
- **Handling emoticons** using a predefined dictionary that maps symbols (e.g., `:)`, `:(`) to words like `"happy"` or `"sad"`.
- **Text normalization**:
  - Lowercasing
  - Removing punctuation
  - Reducing excessive character repetitions
  - Lemmatizing tokens and removing stopwords

Metadata titles and descriptions were also cleaned using the same function. For price data, numeric values were coerced and rows with missing prices were removed.

After merging the cleaned reviews and metadata using `parent_asin`, the final output was saved in `{category}_cleaned.csv` for each product category.

This preprocessing ensures that text input is consistent and structured for all subsequent analysis and machine learning steps.

---

#### **Ratings and Reviews**

---

**Question**:

*Visualize using Matplotlib, Seaborn, or Plotly. You can use histograms, box plots, scatter plots, bar charts, word clouds, etc.:*

- *What is the distribution of product ratings within each of the 5 selected categories? Are there any categories with significantly higher or lower average ratings?*
- *Identify products with a high number of reviews but low ratings. What are some common keywords or phrases in the reviews for these products?*
- *For each of the 5 selected categories, identify the top 5 best-selling (highest review count) products. What are their key attributes (features)?*
- *How have average product ratings evolved over time within each category? Create line plots to show the average rating “trend” for each category over months or years. Are there any noticeable patterns or seasonal variations?*

---

**Answer**:

We performed exploratory data analysis (EDA) on the cleaned review data for all selected categories. Each visualization offers insight into customer behavior and product reception.

##### 1. Rating Distribution

A histogram was plotted to show the distribution of star ratings (1–5) for each product category. This helps identify skewness toward high or low ratings.

##### 2. Low-Rated Popular Products $\xrightarrow{}{}$ Word Cloud

We isolated products with average ratings below 3 and a sufficient number of reviews, then generated a word cloud from their review texts. This visualization reveals the most frequent terms associated with customer dissatisfaction.

##### 3. Rating Trends Over Time

By computing the average monthly rating, we visualized rating trends as line plots over time. These help highlight seasonality, shifts in perception, or long-term product performance changes.

##### 4. Top 5 Best-Selling Products

We identified and listed the top 5 most-reviewed products in each category. The analysis includes:
- Product Title
- ASIN
- Store (if available)
- Average Rating
- Total Number of Reviews

Each category visualization includes these components in a single subplot layout, making it easy to interpret multiple data views simultaneously.

---

### **Task 2: Feature Engineering with Sentiment Scores and Ratings**



---

**Question**:

*You can choose from three alternative methods for combining text sentiment, VADER, Hugging Face models, and review ratings to create a final sentiment score. These approaches will help create more nuanced features for machine learning tasks like sentiment classification, recommendation, or customer segmentation.*

##### ***Alternative 1: Weighted Combination of Text Sentiment and Rating***
*In this approach, you will combine sentiment extracted from the review text using VADER or a Hugging Face model with the numerical rating provided by the user. This approach emphasizes blending both the subjective opinion from the text and the explicit satisfaction level indicated by the rating.*

***Steps:***
1.  ***Sentiment Extraction from Review Text:***
    - ***VADER Sentiment:*** *VADER Sentiment: Use the VADER sentiment analyzer to derive a sentiment score from the review text. This score typically ranges from -1 (negative) to +1 (positive).*\

    - ***Hugging Face Sentiment Model**: Alternatively, you can use a pre-trained sentiment model from Hugging Face (e.g., **DistilBERT, RoBERTa**) for a more context-sensitive sentiment classification. The model will classify the review as **positive, negative, or neutral**.*


2. ***Normalize the Rating:***
    - *Convert the numerical **rating** (1 to 5 stars) into a normalized scale from 0 to 1:*
    $$Normalized\ Rating = \frac{Rating - 1}{4}$$
    *Note: This makes the rating comparable with the sentiment scores.*

3. ***Calculate the Final Sentiment Score:***
    -  *Combine the sentiment score from the text with the normalized rating using a weighted average:*
    $$Final\ Sentiment\ Score = w_1\ \times \ Text\ Sentiment\ +\ w_2\ \times\ Normalized\ Rating $$
    *Where w1 and w2 are weights that reflect the importance of text sentiment and rating, respectively. You can experiment with different weight values.* 

##### ***Alternative 2: Rating-Adjusted Sentiment***
*This approach involves adjusting the sentiment score based on the rating to ensure that the numerical rating reflects the intensity of sentiment more strongly.*
1. ***Sentiment Extraction from Review Text:***
    - *Use **VADER** or **Hugging Face Sentiment Models** to extract sentiment scores from the review text, which will be later classified as positive, neutral, or negative.*

2.  ***Adjust Sentiment Based on Rating:***
    -  *For reviews with higher ratings (e.g., 4 or 5 stars), amplify the sentiment score to reflect stronger positive sentiment. For lower ratings (e.g., 1 or 2 stars), adjust the sentiment score downward, making it more negative, even if the text is neutral or mildly positive.*
    - ***Adjusting Method:***
        - *For ratings of **4 or 5**, increase the sentiment score by a factor (e.g., adding 0.2).* 
        - *For ratings of **1 or 2**, decrease the sentiment score by a factor (e.g., subtracting 0.2).*

3. ***Final Sentiment Score:***
    - *Take the adjusted sentiment score and produce the final sentiment label.*

---

**Answer:**


We implemented **both Alternative 1 and Alternative 2** for generating enriched sentiment scores by combining the review text sentiment and user rating.


- In **Alternative 1**, we used a weighted average of:
    - the sentiment score extracted from the review text, and
    - the normalized user rating using the formula:
    $$
    \text{Normalized Rating} = \frac{Rating - 1}{4}
    $$
    The weights `w_text` and `w_rating` were configurable to allow tuning. We supported both **VADER** and transformer-based models (**DistilBERT**, **RoBERTa**, **BERT**) as sentiment extractors. The final score was computed using the method `sentiment_score_weighted_text_rating()`.

- In **Alternative 2**, we adjusted the sentiment score **after extraction**, based on the star rating:
    - Ratings ≥ 4 increased the sentiment score (e.g., +0.2)
    - Ratings ≤ 2 decreased it (e.g., −0.2)

    This adjustment was implemented in the method `sentiment_score_rating_adjusted()`, preserving score boundaries between −1 and +1.

Both methods were encapsulated in a reusable class called `SentimentScoreWithRating`, supporting both **VADER** and **transformer-based models** from Hugging Face (DistilBERT, RoBERTa, BERT). This flexible design allows experimentation with different combinations for better sentiment feature engineering.

---


### **Task 3: Feature Engineering with Price Metrics (Optional)**


---

**Question:**

1. ***Price-per-Feature Metrics:***
    - ***Objective**:  Create features that allow price comparison across products by normalizing price values against their unique attributes.*
    - ***Methods**: For products with various features (e.g., size, color, material), calculate the price per unit of feature. For example, a product like a "leather sofa" may be priced differently depending on its size (2-seater vs. 5-seater), and this can be normalized by dividing the price by the size to compare value for money.*
    - ***Tools**: Simple arithmetic operations or feature extraction tools like pandas can be used to compute these metrics.*

2. ***Normalized Ratings:***
    - ***Objective**: Address the potential bias of high-rating counts by creating a normalized score that adjusts ratings based on the volume of reviews.*
    - ***Methods**: Calculate the normalized rating by adjusting the product's average rating with respect to the number of reviews it has. A product with a high rating but few reviews may be more volatile, whereas one with many reviews will provide more stable, reliable data. Calculate a weighted rating using the formula:*
    $$WightedRating\ =\ Average\ Rating * \log{(ReviewCount\ +\ 1)}$$

*By creating these additional features, the dataset becomes more meaningful and allows subsequent machine learning models to learn better patterns from the data. **Feature importance** can later be assessed using techniques like Random Forest or Gradient Boosting Machines to determine the most relevant features for a particular model.*

---

**Answer**

We implemented **both** feature engineering objectives: **price-per-feature metrics** and **normalized (weighted) ratings**, across all product categories.

1. **Weighted Rating Calculation**  
   To reduce bias from products with very few ratings, we using the given formula.
   This transformation balances raw rating quality with the review volume, making it more reliable for use in modeling tasks like recommendation and ranking.

2. **Price-per-Unit Feature Extraction**  
   To allow price comparisons across products with varying units (e.g., 500ml, 20 tablets, 3 packs), we built a regex-based extraction pipeline. It searches both the product title and description for quantifiable terms such as:

   - Volume: `500ml`, `1 liter`, `12 fl oz`
   - Weight: `2kg`, `300g`
   - Quantity: `12 count`, `30 tablets`, `4 packs`
   - Size/Dimensions: `12 inches`, `10 wipes`, `5 pieces`

   The matched value was stored in a `quantity_extracted` column and used to calculate:
   $$
   \text{Price per Unit} = \frac{\text{Price}}{\text{Quantity Extracted}}
   $$

   If no quantity could be reliably extracted, we defaulted to `1` as a fallback to avoid missing values.

These engineered features were added to the cleaned dataset and inspected across all categories to ensure they were meaningful, numeric, and usable in future tasks such as regression, clustering, and product ranking.

---


## **Part 2 - Learning Tasks**

***Objective**:* 

*Apply machine learning techniques for clustering, recommendation systems, and sentiment analysis.*


---

**Question:**

*Group similar products within categories based on features like price, description, and ratings.*

1. ***Preprocessing*** :

    * *Clean and prepare the text data for model training by standardizing the input (Convert text to lowercase, remove punctuation, perform stemming…) . **Tools**: you can use `nltk` or `spaCy` for preprocessing tasks.*

    * *Scale numerical features (price, ratings) to prevent them from dominating the clustering.*

2. ***Vectorization*** :
    *  *Use **TF-IDF** for vectorization of product descriptions (don’t forget to experiment with the max_features parameter to limit the vocabulary size)*.

    * *Combine numerical features and TF-IDF vectors into a single feature matrix*.

3.  ***Clustering**: Perform clustering using **one clustering** method (e.g., K-Means with elbow method, DBSCAN). Visualize clusters using dimensionality reduction techniques (PCA, t-SNE)*.

***Evaluation Metric:***
Silhouette Score: Measures how similar items within a cluster are compared to items in
other clusters. A higher score indicates better-defined clusters. Analyze the
characteristics of the products within each cluster to gain insights. If you are using
K-means, use the elbow method to help determine the optimal number of clusters.

---

**Answer**

We performed clustering on each product category using a combination of **textual and numerical features**, following the required three-stage process.

1. **Preprocessing**  
   We combined the cleaned product title and description into a single field (`combined_text`). Missing values were replaced with empty strings.  
   Numerical features (`price` and `rating`) were scaled using `StandardScaler` after filling in missing values with the column means.

2. **Vectorization**  
   We applied **TF-IDF vectorization** to the combined text using a vocabulary limited to 10,000 terms (with `min_df=10`, `max_df=0.75` to remove rare and overly common terms).  
   The resulting TF-IDF vectors were horizontally stacked with the scaled numerical features to form the final feature matrix.

3. **Clustering and Evaluation**  
   We used **K-Means clustering**, experimenting with `k` from 2 to 10. Both the **Elbow Method** (based on distortion) and the **Silhouette Score** were used to determine the optimal number of clusters.  
   After identifying the best `k`, we re-ran K-Means with the optimal value and stored the cluster assignments.

   For visualization, we reduced the feature space to 2D using **PCA** and plotted a scatterplot of the clustered products. This helped in identifying how well-separated the clusters were.

4. **Cluster Analysis**  
   For each cluster, we computed average **rating**, **price**, and the number of products it contains.  
   We also printed a small sample of representative products per cluster to qualitatively inspect their attributes.

The final clustered datasets were saved as `{category}_clustered.csv` for all five categories.


---

### **Task 2: Recommendation System**

---

**Question**:

*A recommendation system suggests products to users based on past behavior and preferences. This task involves two major techniques: **Collaborative Filtering** and **Content-Based Filtering**. You can use products from only one category for the recommendation system. Optional - Use products from multiple categories for the recommendation system.*

***Data Prepatation***:
- *For CF, the data needs to be in a suitable format (user-item interactions, ratings). You will need the following fields from the dataset: user_id, asin, rating*

- *For CBF, the preprocessed product descriptions and other relevant attributes are needed (asin, text, price, category, title). You should preprocess the text, and title. You should also normalize the price. Finally, one hot encoding, or another method to convert the category into numerical features, if products from multiple categories are being used.*


***Collaborative Filtering** using scikit-learn, implement a user-based and item-based collaborative filtering (CF) system that generates personalized book recommendations for a given user. Based on the user’s past purchases and ratings, the CF system should recommend the products with their predicted scores. Evaluate the recommendations by
showing the top-K recommended products and their predicted ratings.*

- ***User-Based Collaborative Filtering***
    - *Create a user-item matrix from the training data, filling missing values with 0.*
    - *Calculate the cosine similarity between users using sklearn.metrics.pairwise.cosine_similarity*
    - *For each user-item pair in the test set:*
        - *Find the K most similar users from the training set*
        - *Calculate a predicted rating based on the weighted average of ratings from the similar users.*

- ***Item-Based Collaborative Filtering***
    - *Create a user-item matrix from the training data, filling missing values with 0.*
    - *Transpose the matrix, and then calculate the cosine similarity between items using cosine similarity.*
    - *For each user-item pair in the test set:*
        - *Find the K most similar items from the training set (Try starting with K=5).*
        - *Calculate a predicted rating based on the weighted average of ratings from the similar items.*


***Content-Based Filtering:***

- *Convert product descriptions into **Word2Vec** vectors (it is better to download and use pre-trained embeddings).*
- *Calculate cosine similarity between product vectors.*
- *Recommend top-K products with high similarity scores.*
- *Optional: Evaluate the recommendations using Recall at K (which measures how many of the relevant items appear in the top-K recommended products).*

***Hybrid Approach: Weighted Average:***
- *Combine the recommendations from CF and CBF (For this approach, combine the results from collaborative and content-based filtering, using the asin field to connect the data)*
- *Method: Assign weights to the CF and CBF scores and combine them.* $$ Hybrid\ Score\ =\  (CF\ Score\ *\ CF\ Weight)\ +\ (CBF\ Score\ *\ CBF\ Weight)$$ Example Weights: CF weight: 0.7, CBF weight: 0.3

- *Evaluate the performance of the hybrid system. (experimentation is crucial to find he optimal weights for the hybrid system).*

***Evaluation**: Analyze the strengths and weaknesses of each approach.*



---

**Answer**

We implemented a full recommendation pipeline using three techniques: **Collaborative Filtering (CF)**, **Content-Based Filtering (CBF)**, and a **Hybrid Approach**, evaluated on a per-user basis using Recall@K.

##### Evaluation

We evaluated all recommendation methods using **Recall@100**, on a user-wise test split (80/20) with a minimum of 5 ratings per user. The system iterated over test users and compared predicted top-K ASINs against held-out items.

**Key Observations**:
- **Item-based CF** and **Hybrid** methods consistently outperformed others in recall.
- **CBF** performed well for users with limited history, thanks to rich product text.
- **Random** baselines showed significantly lower recall, validating the effectiveness of our models.

All methods were modular, allowing consistent benchmarking and flexible weighting.

    
---


##### 2. Content-Based Filtering (CBF)

For CBF, we used **Word2Vec embeddings** to encode each product’s combined textual features (title + description):

- We loaded a pre-trained **Google News Word2Vec** model.
- Each product was represented by averaging its word embeddings.
- A user profile vector was generated by averaging the vectors of products the user rated ≥ 4.
- Cosine similarity was used to compute similarity between the user profile and all product vectors.
- The top-K most similar products were recommended.

This method worked well even with sparse user histories, as long as enough textual information was present.


##### 3. Hybrid Recommendation System

The hybrid system combined both CF and CBF scores using a weighted average:

$$
\text{Hybrid Score} = (w_{cf} \cdot \text{CF Score}) + (w_{cbf} \cdot \text{CBF Similarity})
$$

- CF Score was computed as the average of user-based and item-based predictions.
- CBF Similarity was the average similarity between the candidate product and products the user rated highly.
- Default weights used were **CF: 0.6** and **CBF: 0.4**.

This allowed us to balance behavioral patterns with content similarity, leading to more robust and explainable recommendations.


##### 4. Baseline Recommendations

To evaluate the effectiveness of our recommendation models, we implemented a **random recommendation baseline**.

This method selects `k` random products from the entire dataset, regardless of user preferences or item similarity:

##### 5. Training and Running

Once the models for **Collaborative Filtering**, **Content-Based Filtering**, and the **Hybrid approach** were implemented, we proceeded with dataset preparation and evaluation.

- For each category:
  - We cleaned and filtered the data to retain only necessary fields (`user_id`, `asin`, `rating`, `text`, `price`).
  - We split the dataset **per user** into training and test sets using an 80/20 ratio, skipping users with fewer than 5 ratings to ensure evaluation reliability.

- We then constructed:
  - A **user-item matrix** for CF methods
  - A **Word2Vec-based vector matrix** for CBF, using pre-trained embeddings
  - **Cosine similarity matrices** for users, items, and content vectors

- Using the helper method `get_recommendations()`, we generated the top-100 recommendations for each test user from:
  - `user_cf`
  - `item_cf`
  - `cbf`
  - `hybrid`
  - `random` (baseline)

- For evaluation, we used **Recall@100**, which measures how many of the test user's true positive items appeared in the top-100 predicted items.

This setup allowed for a unified and comparative benchmarking of all recommendation methods across different categories.


### **Task3: Classification task - sentiment analysis**

---

**Question**:

*Sentiment analysis aims to understand the emotional tone of **customer reviews**, **classifying them** as **positive**, **negative**, or **neutral**.*

1. ***Preprocessing**: Preprocessing steps will now be performed on the reviews*

2. ***Feature Extraction (vectorization)***

    1.  *Create **TF-IDF** features like you did for the descriptions in the previous task.*
    2.  *Use **Word2Vec**, or **FastText** for feature extraction on the reviews (it is better to use pre-trained embeddings).*
    3.  *If you have completed Task 3 you can also append the numerical features you created to the word vectors and experiment to see if they improve performance on the sentiment analysis task. Try thesentiment analysis task without and with the numerical features.*
    
3. ***Best model***

    - ***Objective**: Train different classifiers to predict sentiment labels from the features extracted from review texts.*
    - ***Approach**: Use the following classification models:*
        - ***Naive Bayes***
        - ***KNN***
        - ***Random forests***
        - ***Optional: Deep Learning Models**: More advanced models like **LSTM** or **BERT** can be used for more accurate sentiment predictions, especially when working with large datasets. Use the available functions from Hugging face library and perform the sentiment classification task with deep learning models*
    - ***Metrics for Evaluation**: **F1-Score**, which balances precision and recall. Create a table to showcase **results for all models and all feature sets**. (this is very important because it showcases the performance of your methods and gives you a way to compare different approaches) . All models will be trained exclusively on the training dataset. Model performance will be evaluated on the held-out test dataset.*

4. ***10-Fold Cross-Validation**: This will provide a more robust estimate of modelgeneralization (prevent overfitting). Evaluate and record the performance of eachmodel using 10-fold cross-validation on the **training data**.*

    ***Evaluation Metrics***:
    * *Calculate and record the following metrics:*
      * *Precision (Macro-average)*
      * *Recall (Macro-average)*
      * *F1-Score (Macro-average)*
      * *Accuracy*
       
    ***Results Presentation:***

    * *Create a clear and well-organized table to showcase the results for all models and feature sets. This table should include the evaluation metrics from the 10-fold cross-validation on the train set.*

        | Feature Set | Model         | Precision | Recall | F1-Score | Accuracy |
        |-------------|---------------|-----------|--------|----------|----------|
        | TF-IDF      | KNN           |           |        |          |          |
        | TF-IDF      | Naive Bayes   |           |        |          |          |
        | TF-IDF      | Random Forest |           |        |          |          |
        | Embeddings  | KNN           |           |        |          |          |
        | Embeddings  | Naive Bayes   |           |        |          |          |
        | Embeddings  | Random Forest |           |        |          |          |

    * *You then perform a **single, final evaluation on the test set**. For the test set, you should create a separate table (or section in your report) that shows the performance metrics (precision, recall, F1-score, accuracy) of each trained model.*
  
        | Model         | Precision | Recall | F1-Score | Accuracy |
        |---------------|-----------|--------|----------|----------|
        | KNN           |           |        |          |          |
        | Naive Bayes   |           |        |          |          |
        | Random Forest |           |        |          |          |


---

**Answer**: 

We performed a comprehensive exploratory analysis across all product categories after sentiment labeling. All categories were processed together in a single pipeline, and insights were extracted per category to assess text content and sentiment distribution.

The datasets are **highly unbalanced**, with many more positive reviews than neutral or negative ones — a common characteristic in e-commerce platforms where satisfied customers are more likely to leave reviews.

For each category, we extracted and visualized the following:

#####  **Word Cloud**
A word cloud was generated from the combined cleaned title and text fields (`cleaned_review`) to visualize the most frequent terms across all reviews.

#####  **Top 20 Tokens**
We computed the top 20 most frequent tokens using NLTK's tokenizer and visualized them using a horizontal bar chart. This helps identify repetitive or dominant vocabulary within a category.

#####  **Sentiment Distribution**
Each review was labeled as:
- **positive** if `rating ≥ 4`
- **neutral** if `2 ≤ rating < 4`
- **negative** if `rating < 2`

A count plot was displayed to show the distribution of these sentiment labels. Most categories had a significantly higher proportion of **positive** reviews.

#####  **BERT Token Count Analysis**
We calculated token lengths using NLTK's `word_tokenize` to estimate appropriate input lengths for transformer-based models:
- Mean, standard deviation, and variance of token counts were computed.
- The **suggested max token length** per category was estimated as:
  $$
  \text{max\_len} = \min(\text{mean} + 2 \cdot \text{std}, 512)
  $$
- A histogram of review lengths (in tokens) was also plotted, capped at 300 tokens for clarity.

The calculated `bert_max_len` values for each category were stored in a dictionary for use in downstream modeling.


---

**Answer**

We trained a BERT-based classifier to predict sentiment labels (positive, neutral, negative) from customer reviews.

- We **mapped sentiment labels to integers** and created a custom `SentimentDataset` to tokenize and encode reviews using a Hugging Face tokenizer.

- During training, we used **early stopping based on F1-score** to handle class imbalance and avoid overfitting. The training loop included:
  - Linear learning rate scheduling
  - Validation loss tracking
  - Best model checkpointing based on macro F1

- After training, we evaluated the model on a test set using **macro-averaged precision, recall, F1-score, and accuracy**.

This setup allowed us to build a strong deep learning sentiment model aligned with the project’s classification goals.



---

**Answer**

We evaluated multiple classification models for sentiment analysis, comparing both traditional machine learning methods and deep learning (BERT) across all product categories. The goal was to classify customer reviews into **positive**, **neutral**, or **negative** classes.

#### Data Preparation

- Each review was labeled based on its rating:
  - `positive` if rating ≥ 4
  - `neutral` if rating is 2 or 3
  - `negative` if rating < 2
- Reviews were preprocessed and tokenized.
- The dataset was **split using stratified sampling** into training (80%) and test (20%) sets to maintain class balance.


#### Feature Extraction

- **TF-IDF**: Top 10,000 terms, filtered by `min_df=10`, `max_df=0.75`
- **Word2Vec**: Pretrained Google News vectors were averaged per review


#### Traditional Models Evaluated

Each of the following models was trained and evaluated using both **TF-IDF** and **Word2Vec** features:
- **Naive Bayes** (`MultinomialNB` for TF-IDF, `GaussianNB` for Word2Vec)
- **KNN** (`KNeighborsClassifier`)
- **Random Forest** (`RandomForestClassifier`)

Two evaluations were performed:
1. **10-Fold Cross-Validation** on the training set (macro-averaged precision, recall, F1-score, accuracy)
2. **Final evaluation on the test set** using the same metrics


#### BERT (Deep Learning Model)

- A pre-trained **BERT (bert-base-uncased)** model was fine-tuned on the same dataset using PyTorch and Hugging Face.
- Early stopping was applied based on **macro F1-score**, with a validation set (10% of train split).
- Reviews were tokenized using the BERT tokenizer and padded to a category-specific `max_len`.
- BERT was **not cross-validated** to avoid overfitting and reduce computational cost.


#### Baseline Comparison

We included a **DummyClassifier** as a naive baseline using the **most frequent class** strategy, helping assess whether learned models outperform random or trivial predictions.

---

To evaluate the performance of our BERT sentiment classifier, we visualized key metrics using three diagnostic plots for each category:

#### 1. Learning Curves

We plotted **training and validation loss** over epochs to monitor the model’s learning behavior. This helps identify issues such as overfitting or underfitting and guides decisions on early stopping.


#### 2. ROC Curve (Multiclass)

Using the model's output logits, we calculated **one-vs-rest ROC curves** for each sentiment class (negative, neutral, positive).  
Each curve includes its **AUC score**, offering insight into the model’s confidence and separability across classes.


#### 3. Confusion Matrix

We visualized the **confusion matrix** to better understand how the model performed on each class. It clearly shows:
- Correct vs. incorrect predictions
- Which classes were most confused
- Class-specific weaknesses (e.g., confusion between neutral and positive)

These plots provide a comprehensive view of the model's effectiveness, both globally (via ROC) and per-class (via confusion matrix).


