import pandas as pd
import nltk
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report


# Load your dataset into a DataFrame
df = pd.read_csv('C:/Users/jesmi/OneDrive/Desktop/Ap/Restaurantreviews.csv')
# Check the columns in your DataFrame
print(df.columns)

# Display the first few rows of your DataFrame to see the structure and content
print(df.head())

# Display the first few rows of the DataFrame to understand its structure
print(df.head())

# Function to preprocess text
def preprocess_text(text):
    # Check if text is not NaN or None
    if pd.isnull(text):
        return ""  # Return an empty string for missing values
    else:
        # Convert text to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Tokenize text
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        
        # Lemmatize words
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        
        # Join tokens back into text
        processed_text = ' '.join(tokens)
        
        return processed_text

# Check if 'Review' column exists in the DataFrame
if 'Review' in df.columns:
    # Apply preprocessing to each review in the 'Review' column
    df['cleaned_review'] = df['Review'].apply(preprocess_text)
    # Display the cleaned data
    print(df[['Review', 'cleaned_review']].head())
else:
    print("Error: 'Review' column not found in the DataFrame.")
#Step3############################################################################ 
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

# Distribution of review lengths
review_lengths = df['cleaned_review'].apply(lambda x: len(x.split()))
plt.hist(review_lengths, bins=50, color='skyblue')
plt.xlabel('Review Length')
plt.ylabel('Frequency')
plt.title('Distribution of Review Lengths')
plt.show()

# Word frequency analysis
all_words = ' '.join(df['cleaned_review'])
word_freq = Counter(all_words.split())
word_freq_df = pd.DataFrame(word_freq.most_common(20), columns=['Word', 'Frequency'])
print("Top 20 Most Common Words:")
print(word_freq_df)

# Word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Most Common Words')
plt.show()
#Step4############Feature engineering########################################################################
from sklearn.feature_extraction.text import CountVectorizer

# Initialize CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform the preprocessed text data
X_bow = vectorizer.fit_transform(df['cleaned_review'])

# Display the shape of the BoW matrix
print("Shape of BoW matrix:", X_bow.shape)
#Step5##############Model Selection and Training###############################################################
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#import nltk
#nltk.download('vader_lexicon')
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#Initialize the sentiment Analyzer
sid = SentimentIntensityAnalyzer()

# Define a function to assign sentiment scores to each review
def analyze_sentiment(text):
    scores = sid.polarity_scores(text)
    # The compound score ranges from -1 (most negative) to 1 (most positive)
    if scores['compound'] >= 0.05:
        return 'positive'
    elif scores['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Apply sentiment analysis to the cleaned reviews
df['predicted_sentiment'] = df['cleaned_review'].apply(analyze_sentiment)

# Optionally, you can save the cleaned and analyzed dataset to a new CSV file
df.to_csv('cleaned_and_analyzed_dataset.csv', index=False)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_bow, df['predicted_sentiment'], test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict the sentiment labels for the testing data
y_pred = model.predict(X_test)

# Visualize the result
import matplotlib.pyplot as plt

# Count the occurrences of each predicted sentiment label
sentiment_counts = df['predicted_sentiment'].value_counts()

# Plot a bar plot
sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'])
plt.title('Distribution of Predicted Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate a classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))
#Step6####################CLASSIFICATION###################################################################################
# Check if 'Review' column exists in the DataFrame
if 'Review' in df.columns:
    df['cleaned_review'] = df['Review'].apply(preprocess_text)
else:
    print("Error: 'Review' column not found in the DataFrame.")

# Create a function to assign aspect labels based on review content
def assign_aspect_label(cleaned_review):
    if "ambience" in cleaned_review.lower() or "ambiance" in cleaned_review.lower():
        return "Ambience"
    elif "service" in cleaned_review.lower() or "waiter" in cleaned_review.lower() or "staff" in cleaned_review.lower():
        return "Service"
    elif "food" in cleaned_review.lower() or "taste" in cleaned_review.lower() or "starrers" in cleaned_review.lower():
        return "Food Taste"
    elif "price" in cleaned_review.lower() or "value" in cleaned_review.lower():
        return "Price/Value"
    elif "hygiene" in cleaned_review.lower() or "cleanliness" in cleaned_review.lower() or "hospitalized" in cleaned_review.lower() or "infection" in cleaned_review.lower():
        return "Hygiene/Cleanliness"
    elif "experience" in cleaned_review.lower() or "overall" in cleaned_review.lower() or "beyond" in cleaned_review.lower():
        return "Overall Experience"
    else:
        return "Other"

# Assign aspect labels to each review
df['aspect_label'] = df['cleaned_review'].apply(assign_aspect_label)

# Print the distribution of aspect labels
print("Aspect Distribution:")
print(df['aspect_label'].value_counts())

# Print example reviews for each aspect
print("\nExample Reviews for Each Aspect:")
for aspect in df['aspect_label'].unique():
    print(f"\nAspect: {aspect}")
    aspect_reviews = df[df['aspect_label'] == aspect]['Review'].head(3).tolist()
    for cleaned_review in aspect_reviews:
        print(cleaned_review)




