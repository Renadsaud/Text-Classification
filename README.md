# Text_classification
This project aims to perform sentiment analysis on Arabic reviews using the BERT model, a pre-trained deep learning model that has been proven effective for this task. Sentiment analysis is a technique in natural language processing that identifies and extracts subjective information from text data to determine the prevailing attitude of a particular text. The ultimate objective of this project is to develop a sentiment analysis model that can accurately classify Arabic reviews as positive, or negative.

[![Picture1.png](https://i.postimg.cc/wMxSMVfn/Picture1.png)](https://postimg.cc/DJDCYqKc)

The process of sentiment analysis involves several steps, including preprocessing, tokenization, lemmatization, and stop-word removal. Preprocessing involves determining relevant keywords that highlight the central message of the text. Tokenization divides the text into sentences and words for analysis called tokens. Lemmatization transforms words to their root form, and stop-word removal filters out insignificant words. Natural Language Processing (NLP) technologies assign sentiment scores to identified keywords, providing a quantitative measure of the sentiment expressed in the text.

Once the text is preprocessed, it undergoes sentiment analysis using the BERT model, a widely used approach. BERT learns word relationships within sentences through masked language modeling, where some words are masked, and the model predicts the missing ones. With this knowledge, BERT can be applied to various NLP tasks, including sentiment analysis. By examining word relationships in sentences, BERT accurately predicts the text's sentiment. The model's accuracy is assessed for both negative and positive reviews and compared to other machine learning models.

After conducting sentiment analysis on an Arabic text using BERT, it can be concluded that the model performed well in accurately identifying and classifying the sentiment of the text. The results showed that the trained model achieved an accuracy of 87.67% on the test data, indicating that the model learned reasonably well from the training data. 

Additionally, we compared the test model with other ML models such as Naive Bayes MultinomialNB and BernoulliNB, to evaluate accuracy in the sentiment prediction model. After applying NB models on the same Arabic dataset, we concluded that BERT had the best accuracy as it outperformed both MultinomialNB and BernoulliNB by approximately 5%. 

Overall, BERT proves to be a promising tool for sentiment analysis in Arabic texts, and we can expect even more advanced techniques and models to further enhance the accuracy and effectiveness of sentiment analysis in the field of natural language processing.


Throughout the project, various libraries were utilized, including Numpy, Pandas, Seaborn, Transformers, Time, Unicode data, Torch, Train_test_split, MultinomialNB, Classification_report, CountVectorizer, and matplotlib.pyplot. These libraries facilitated tasks like data manipulation, visualization, machine learning, and text processing.



![image](https://github.com/Renadsaud/Text_classification/assets/95434316/4d9bd6e2-ce08-4c3c-b135-fb0ccfd15a4d)


![image](https://github.com/Renadsaud/Text_classification/assets/95434316/0a8a4e16-c1fc-453d-9d6b-5b4e326328d6)

![image](https://github.com/Renadsaud/Text_classification/assets/95434316/5df05ebe-82bf-473e-af17-2be02b8ed11a)

