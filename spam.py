#spam and not spam
import numpy as np#no
from sklearn.feature_extraction.text import CountVectorizer#no
from sklearn.model_selection import train_test_split#no
from collections import defaultdict#fine
import math#fine

# Step 1: Read data from a text file
# Assuming each line is formatted as: "message,label"
messages = []
labels = []

with open("messages.txt", "r") as file:
    for line in file:
        parts = line.strip().split(',')
        if len(parts) == 2:  # Ensure there are exactly 2 parts (message and label)
            message, label = parts
            messages.append(message)
            labels.append(label)
        else:
            print(f"Skipping invalid line: {line.strip()}")

# Convert lists to numpy arrays for consistency
x = np.array(messages)  # The input text (features)
y = np.array(labels)    # The labels (spam or not_spam)

# Step 2: Convert text data to numeric features using CountVectorizer
cv = CountVectorizer()#Naive Bayes, require numerical input to work with values
X = cv.fit_transform(x)  # Create a document-term matrix (word counts)
vocab = cv.get_feature_names_out()

# Split the data into training and testing sets (67% train, 33% test) strikes a balance between providing enough data for the model to learn patterns and enough data to test the model's ability to generalize
X_train, X_test, y_train, y_test = train_test_split(X.toarray(), y, test_size=0.33, random_state=42)

# Step 3: Calculate Prior Probabilities (P(class))
def calculate_priors(y_train):
    class_counts = defaultdict(int)
    total_count = len(y_train)
    
    for label in y_train:
        class_counts[label] += 1
    
    priors = {label: count / total_count for label, count in class_counts.items()}
    return priors

# Step 4: Calculate Likelihoods (P(word | class))
def calculate_likelihoods(X_train, y_train, vocab):
    word_counts = {label: np.zeros(len(vocab)) for label in np.unique(y_train)}
    class_counts = defaultdict(int)
    
    for i, label in enumerate(y_train):
        word_counts[label] += X_train[i]
        class_counts[label] += np.sum(X_train[i])
    
    # Smoothing to avoid zero probabilities (Laplace smoothing)
    for label in word_counts:
        word_counts[label] = (word_counts[label] + 1) / (class_counts[label] + len(vocab))  # Smoothed likelihood
    
    return word_counts

# Step 5: Predict the class for a new message
def predict(message, priors, likelihoods, vocab):
    # Convert the message to a vector (using the same vectorizer)
    message_vector = cv.transform([message]).toarray().flatten()
    
    # Initialize dictionary to store the log-probabilities for each class
    log_probs = {label: math.log(prior) for label, prior in priors.items()}
    
    # Calculate log-probabilities for each class
    for label, likelihood in likelihoods.items():
        for i, word_count in enumerate(message_vector):
            if word_count > 0:  # Only consider words that appear in the message
                log_probs[label] += word_count * math.log(likelihood[i])
    
    # Return the class with the highest log-probability
    return max(log_probs, key=log_probs.get)

# Train the model
priors = calculate_priors(y_train)
likelihoods = calculate_likelihoods(X_train, y_train, vocab)

# Step 6: Test the model on test data
def evaluate(X_test, y_test, priors, likelihoods, vocab):
    correct = 0
    for i, message_vector in enumerate(X_test):
        message = ' '.join([vocab[idx] for idx in np.nonzero(message_vector)[0]])
        predicted = predict(message, priors, likelihoods, vocab)
        if predicted == y_test[i]:
            correct += 1
    return correct / len(y_test)

accuracy = evaluate(X_test, y_test, priors, likelihoods, vocab)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Optional: Test with a new message
new_message = "Money money money Im Mr Krabs and i want your MONEYYY!!!!!!!!"
prediction = predict(new_message, priors, likelihoods, vocab)
print(f"Prediction for '{new_message}': {prediction}")

