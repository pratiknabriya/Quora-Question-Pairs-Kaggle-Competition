## About Quora Question Pairs Kaggle competition

Where else but Quora can a physicist help a chef with a math problem and get cooking tips in return? Quora is a place to gain and share knowledge—about anything. It’s a platform to ask questions and connect with people who contribute unique insights and quality answers. This empowers people to learn from each other and to better understand the world.

Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term.

Currently, Quora uses a Random Forest model to identify duplicate questions. In this competition, Kagglers are challenged to tackle this natural language processing problem by applying advanced techniques to classify whether question pairs are duplicates or not. Doing so will make it easier to find high quality answers to questions resulting in an improved experience for Quora writers, seekers, and readers.

For dataset and more info refer https://www.kaggle.com/c/quora-question-pairs

## Summary & Results

| Model no. |        Model        | Text Vectorizer | Hyperparameter tuning | Train loss | Test loss | Accuracy % |
|-----------|---------------------|------------|-----------------------|-----------|------------|------------|
|    1A.    | Logistic Regression | TFIDF-W2V  |          YES          |   0.391   |   0.393    | 80.28
|    1B.    | Logistic Regression |   TFIDF    |          YES          |   0.352   |   0.367    | 82.09
|    2A.    |      Linear SVM     | TFIDF-W2V  |          YES          |   0.394   |   0.393    | 80.41
|    2B.    |      Linear SVM     |   TFIDF    |          YES          |   0.357   |   0.372    | 81.82
|     3.    |       XGBoost       | TFIDF-W2V  |          YES          |   0.267   |   0.321    | 85.18

## My Approach 

1. The original dataset comprises of 404290 rows and 6 columns. We checked for the null values and found couple of null entries in the questions which were replaced with empty string.

2. Class label distribution:
   * is_duplicate = 0  ->  255027 datapoints (~ 63%)

   * is_duplicate = 1  ->  149263 datapoints (~ 37%)

3. Wordcloud observation:
   * Most common words in duplicate question pairs  ->  "Donald Trump", "Hillary Clinton", "difference", "best way", etc.

   * Most common words in non_duplicate question pairs  ->  "India", "one", "difference", "not", "like", "best", "best way" etc. 

4. Then we found the total number of unique question in the entire dataframe and plotted a histogram of their frequencies. Most questions occur only once or twice, while there are relatively very few questions that are asked more than 10 times. The highest number of times any question is asked is 157. 

5. Next we extracted some basic features like frequency of the questions, string length of the questions, number of words in the questions and other features that are combination of these features. We also created a new feature word share which is equal to number of common words divided by total unique words. 

6. Then we performed the correlation analysis of the newly extracted features above using seaborn heatmap and found that some pairs of feature are highly correlated.

7. Furthur we performed Univariate analysis on some of the extracted features and concluded that "word_share" feature does moderately good job in seperating the class labels while "word_common" feature is terrible due to high overlap. This also suggests that having high number of words common among the question pairs doesn't mean that they are likely to be duplicates. 

8. Before proceeding to advanced feature engineering, we investigated some random question texts to come-up with a text-preprocessing pipe-line. The text preprocessing pipe-line includes the usual steps - Expanding contractions, Removing html tags, Removing special characters like '%','&', '@', removing stopwords, etc.

9. In the Advanced feature engineering section we further constructed 15 new specialized features in context of the given task. This includes features like the ratio of common token counts divided by the min and max token lengths among question 1 and question 2. Similary, we did same for stop word counts and meaningful word (i.e.token words - stop words) counts. We also added some fuzzy features like fuzz_ratio, fuzz_partial ratio, etc. and some other features that check if first word of q1 and q2 is same, last word is same, mean length and absolute length difference. 
Reference -> https://github.com/seatgeek/fuzzywuzzy
http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/

10. Just like the previous sections, the feature extraction is followed by the correlation analysis, bi-variate analysis using pair plot and univariate analysis. This is followed by t-SNE wherein we visualize the 16 and 29 dimension data respectively by embedding it in 2 dimension such that the neigbourhood is preserved. 

11. Before proceeding to building a classification model, we vectorize our text data of question 1 and question 2 using TF-IDF weighted Word2Vec vectorizer(question text -> 300 features each) as well as only TF-IDF vectorized (question vector -> 5000 features each). Note that here the vocabulary for the text vectorization should be built on the combined question1 and question2 texts.

12. Now we have 3 sets of features - 
   * Basic features (word_common, word_share, question frequencies etc.)
   * Advanced features (cwc_min, csc_max, fuzz_ratio, etc.)
   * TFIDF-W2V / TFIDF Vectorized text features. 

  We stack all the features together and proceed to train our Machine Learning models. 

13. The business metric we are using for checking the model performance is log-loss. 

  The models we train are - 
   * Logistic regression (with TFIDF-W2V vectorized text data).
   * Logistic regression (with TFIDF vectorized text data).
   * Linear SVM (with TFIDF-W2V vectorized text data).
   * Linear SVM (with TFIDF vectorized text data).
   * XGBoost (with TFIDF vectorized text data).

  All the above models we build by using the best set of hyperparameters found using GridSearchCV and RandomSearchCV.

14. In order to compute the log-loss metric, the predicted output should be actual probability scores instead of class labels. So after featching the probability scores from the above classification models, we perform an additional step of Calibration in order to obtain the accurate probability. 

15. Finally the class-labels are predicted from the corrected probability scores and the results of each model are displayed in form of Confusion matrix, Precision matrix and Recall matrix. The consolidated results from all the models are already summerized above in the Conclusion section. 

16. It is observed that tree-based ensemble model XGBoost tends to perform better as compared to the Linear models models like Logistic regression and Linear SVM. In case of linear models, models with TF-IDF vectorized text features outperforms the model with TFIDF-W2V vectorized text features. 
