import re
from collections import Counter, defaultdict

# Sample dataset (label, SMS text)
data = [
    ("spam", "Win $1000 now!!! Click here: http://spam.com"),
    ("ham", "Hey, are we still on for dinner tonight?"),
    ("spam", "Congratulations! You've won a free gift card."),
    ("ham", "Don't forget the meeting tomorrow at 10 AM."),
    ("spam", "Claim your prize now!!! Call 1800-SPAM"),
    ("ham", "Can you pick up some groceries on your way home?"),
]


# Preprocess text: Lowercase, remove special characters, and split into words
def preprocess(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove non-alphanumeric characters
    return text.lower().split()

# Split data into ham and spam
ham_texts = [sms for label, sms in data if label == "ham"]
spam_texts = [sms for label, sms in data if label == "spam"]

# Count word frequencies for ham and spam
ham_words = Counter(word for sms in ham_texts for word in preprocess(sms))
spam_words = Counter(word for sms in spam_texts for word in preprocess(sms))

# Total number of ham and spam messages
num_ham = len(ham_texts)
num_spam = len(spam_texts)
total_messages = num_ham + num_spam

# Prior probabilities
p_ham = num_ham / total_messages
p_spam = num_spam / total_messages

# Calculate word probabilities with Laplace smoothing
def calculate_word_probabilities(word_counts, total_words, vocab_size):
    probabilities = defaultdict(lambda: 1 / (total_words + vocab_size))  # Laplace smoothing
    for word, count in word_counts.items():
        probabilities[word] = (count + 1) / (total_words + vocab_size)
    return probabilities

# Vocabulary size
vocab = set(ham_words.keys()).union(spam_words.keys())
vocab_size = len(vocab)

# Word probabilities
ham_probabilities = calculate_word_probabilities(ham_words, sum(ham_words.values()), vocab_size)
spam_probabilities = calculate_word_probabilities(spam_words, sum(spam_words.values()), vocab_size)

# Classify a new message
def classify_message(message):
    words = preprocess(message)
    ham_score = p_ham
    spam_score = p_spam
    for word in words:
        ham_score *= ham_probabilities[word]
        spam_score *= spam_probabilities[word]
    return "spam" if spam_score > ham_score else "ham"

# Test the classifier
test_messages = [
    "You have won a free ticket! Call now!",
    "Can we reschedule our meeting to next week?",
    "Claim your reward by visiting spammy-site.com",
    "Are you coming to the party tonight?",
]

for msg in test_messages:
    print(f"Message: \"{msg}\" => Classified as: {classify_message(msg)}")
