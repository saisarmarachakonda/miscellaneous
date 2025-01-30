import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Sample DataFrame
data = {'text': ["Infosys123, IBM", "Worked at Google and Accenture", "Wipro and IBM"]}
df = pd.DataFrame(data)

# Employer names to match
employer_set = {"Infosys", "IBM", "Google", "Wipro", "Tata Consultancy", "Accenture"}

# Similarity threshold
threshold = 80

# Create a TfidfVectorizer with character-based n-grams of size 2
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 2))

# Fit the vectorizer on the employer names
employer_ngrams = vectorizer.fit_transform(employer_set)

# Function to compute matches for each text in the DataFrame
def extract_matches(text):
    # Transform the input text into the same n-gram space
    text_ngrams = vectorizer.transform([text])

    # Compute cosine similarities between the input text and the employer names
    cosine_similarities = cosine_similarity(text_ngrams, employer_ngrams).flatten()

    # Extract matches with thresholding
    matches = [
        (list(employer_set)[idx], score * 100)
        for idx, score in enumerate(cosine_similarities)
        if score * 100 >= 0
    ]
    
    # Extract matched substrings (including numbers attached to the employer names)
    matched_substrings = [
        re.search(re.escape(match) + r'\w*', text).group()
        for match, score in matches if re.search(re.escape(match) + r'\w*', text)
    ]
    
    return matches, matched_substrings

# Apply the function to each row in the DataFrame
df['matches'], df['matched_substrings'] = zip(*df['text'].apply(extract_matches))


df







import pandas as pd
import re
from rapidfuzz import fuzz

# Sample data
data = {'company_name': ['MC DONALDS1234 ABC 123', 'ABC 123 SOLUTIONS', 'XYZ TECH99', 'HELLO-WORLD! XYZ TECH', '$123abc!']}
df = pd.DataFrame(data)

# List of employer names
employer_names = ["MCDONALDS", "ABC 123", "XYZ TECHNOLOGIES", "HELLO WORLD", "$123abc123"]

# Preprocessing function to remove special characters, extra spaces, and unicode characters
def preprocess(text):
    # Remove non-alphanumeric characters and spaces except for underscores and digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation and special characters
    text = re.sub(r'\s+', ' ', text)     # Collapse multiple spaces into one
    text = text.strip()                  # Remove leading/trailing spaces
    return text

# Function to find matched substrings and return them as a list of lists using RapidFuzz
def find_matched_substrings_as_lists_rapidfuzz(df, employer_names, threshold=75):
    return [
        [
            token
            for token in preprocess(company_name).split()
            if any(fuzz.ratio(token, preprocess(employer)) >= threshold for employer in employer_names)
        ] + [
            ' '.join(preprocess(company_name).split()[i:i+2])
            for i in range(len(preprocess(company_name).split()) - 1)
            if any(fuzz.ratio(' '.join(preprocess(company_name).split()[i:i+2]), preprocess(employer)) >= threshold for employer in employer_names)
        ]
        for company_name in df['company_name']
        if any(fuzz.ratio(token, preprocess(employer)) >= threshold for token in preprocess(company_name).split() for employer in employer_names) or
           any(fuzz.ratio(' '.join(preprocess(company_name).split()[i:i+2]), preprocess(employer)) >= threshold for i in range(len(preprocess(company_name).split()) - 1) for employer in employer_names)
    ]

# Get the list of matched substrings as lists using RapidFuzz
final_matched_lists = find_matched_substrings_as_lists_rapidfuzz(df, employer_names)

# Display the list of lists
print(final_matched_lists)




import pandas as pd
from rapidfuzz import fuzz
import re

def preprocess_for_matching(text):
    """Preprocess text for matching by normalizing case and keeping meaningful characters."""
    text = text.upper().strip()
    text = re.sub(r'[^A-Z0-9\-\$]', ' ', text)  # Preserve hyphens and dollar signs
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text

def split_company_names(text):
    """Splits company names based on common delimiters like commas, ampersands, and slashes."""
    return [name.strip() for name in re.split(r'[,&/]+', text) if name.strip()]

def find_best_substring_match(original_text, sub_name, employer_names):
    """Finds the best-matching substring from the original company name."""
    sub_name_clean = preprocess_for_matching(sub_name)
    
    best_match = max(
        [(emp, fuzz.partial_ratio(sub_name_clean, preprocess_for_matching(emp))) for emp in employer_names],
        key=lambda x: x[1], default=(None, 0)
    )

    if best_match[1] > 75:  # Threshold for a strong match
        match_start = original_text.upper().find(sub_name.upper())
        if match_start != -1:
            return original_text[match_start:match_start + len(sub_name)]

    return None

# Sample DataFrame
data = {'company_name': ['MCDONALDS, ABC 123', 'ABC 123 SOLUTIONS', 'XYZ TECH99', 'HELLO-WORLD! XYZ TECH', '$123abc!']}
df = pd.DataFrame(data)

# List of employer names
employer_names = ["MCDONALDS", "ABC-123 SERVICES", "XYZ TECHNOLOGIES", "HELLO WORLD", "$123abc123"]

# Vectorized function to extract matching substrings for each company name
def extract_matching_substrings(company_name):
    # List comprehension to process each company and find matches
    return [
        match for sub_name in split_company_names(company_name)
        for match in [find_best_substring_match(company_name, sub_name, employer_names)] if match
    ]

df['matched_substrings'] = df['company_name'].apply(extract_matching_substrings)

# Print the DataFrame
print(df)
