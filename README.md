# Text classification based on sentiment analysis

This project aims to perform sentiment analysis on Arabic reviews using the BERT model, a pre-trained deep learning model that has been proven effective for this task. Sentiment analysis is a technique in natural language processing that identifies and extracts subjective information from text data to determine the prevailing attitude of a particular text. The ultimate objective of this project is to develop a sentiment analysis model that can accurately classify Arabic reviews as positive, or negative.

# Implementation

<img src="https://i.ibb.co/nQH0KcM/Picture1.png" alt="Picture1" border="0">

The process of sentiment analysis involves several steps, including preprocessing, tokenization, lemmatization, and stop-word removal. Preprocessing involves determining relevant keywords that highlight the central message of the text. Tokenization divides the text into sentences and words for analysis called tokens. Lemmatization transforms words to their root form, and stop-word removal filters out insignificant words. Natural Language Processing (NLP) technologies assign sentiment scores to identified keywords, providing a quantitative measure of the sentiment expressed in the text.

Once the text is preprocessed, it undergoes sentiment analysis using the BERT model, a widely used approach. BERT learns word relationships within sentences through masked language modeling, where some words are masked, and the model predicts the missing ones. With this knowledge, BERT can be applied to various NLP tasks, including sentiment analysis. By examining word relationships in sentences, BERT accurately predicts the text's sentiment. The model's accuracy is assessed for both negative and positive reviews and compared to other machine learning models.

------------------------------------------------------------------------------------

After conducting sentiment analysis on an Arabic text using BERT, it can be concluded that the model performed well in accurately identifying and classifying the sentiment of the text. The results showed that the trained model achieved an accuracy of 87.67% on the test data, indicating that the model learned reasonably well from the training data. The figure below provides a graph of the values predicted by the model compared to the actual values in the testing dataset.

<img src="https://i.ibb.co/g6dcSfV/Picture2.png" alt="Picture2" border="0">


Additionally, we compared the test model with other ML models such as Naive Bayes MultinomialNB and BernoulliNB, to evaluate accuracy in the sentiment prediction model. After applying NB models on the same Arabic dataset, we concluded that BERT had the best accuracy as it outperformed both MultinomialNB and BernoulliNB by approximately 5%.

<img src="https://i.ibb.co/JkR8q8K/Picture3.png" alt="Picture3" border="0">

In the figure below, the True and predicted values of MNB and BNB are shown, where the confusion matrix explains the ratios of correctly predicted values to falsely predicted values.
<br/> 
<br/>
<img src="https://i.ibb.co/KWhpXWZ/Picture4.png" alt="Picture4" border="0"><img src="https://i.ibb.co/zRRcL4d/Picture5.png" alt="Picture5" border="0">
<br/>

Overall, BERT proves to be a promising tool for sentiment analysis in Arabic texts, and we can expect even more advanced techniques and models to further enhance the accuracy and effectiveness of sentiment analysis in the field of natural language processing.

------------------------------------------------------------------------------------
# Libraries used 
Throughout the project, various libraries were utilized, including Numpy, Pandas, Seaborn, Transformers, Time, Unicode data, Torch, Train_test_split, MultinomialNB, Classification_report, CountVectorizer, and matplotlib.pyplot. These libraries facilitated tasks like data manipulation, visualization, machine learning, and text processing.

# The programming language used
we used Python language to implement and design the interface.



