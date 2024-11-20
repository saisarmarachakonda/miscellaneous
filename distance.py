from pyspark.ml.feature import Tokenizer, HashingTF, MinHashLSH
from pyspark.sql.functions import col

# Tokenize the employer names
tokenizer = Tokenizer(inputCol="employer_name", outputCol="tokens")
tokenized_df = tokenizer.transform(spark_df)

# Apply HashingTF to generate hashed features
hashing_tf = HashingTF(inputCol="tokens", outputCol="features", numFeatures=10000)
tf_df = hashing_tf.transform(tokenized_df)

# Apply MinHashLSH for approximate similarity
minhash_lsh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=3)
lsh_model = minhash_lsh.fit(tf_df)

# Find duplicate employer names (self-join using LSH)
similar_pairs = lsh_model.approxSimilarityJoin(tf_df, tf_df, 0.8, distCol="similarity")

# Filter out self-matches
similar_pairs_filtered = similar_pairs.filter(
    col("datasetA.employer_name") != col("datasetB.employer_name")
)

# Display results
similar_pairs_filtered.select(
    col("datasetA.employer_name").alias("Employer1"),
    col("datasetB.employer_name").alias("Employer2"),
    col("similarity")
).show(truncate=False)


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Sample data (for example, company names)
data = ['WAL MART', 'WALMART', 'AMAZON', 'AMAZON PVT LTD', 'Walmart', 'Amazon INC']

# Convert data to a DataFrame
df = pd.DataFrame(data, columns=['Company Name'])

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Transform company names into TF-IDF vectors
tfidf_matrix = vectorizer.fit_transform(df['Company Name'])

# Compute pairwise cosine similarity
cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Display similarity matrix
print(cosine_similarities)

# You can then set a threshold (e.g., similarity > 0.8) to identify similar names.
threshold = 0.8
similar_pairs = np.where(cosine_similarities > threshold)
for i, j in zip(*similar_pairs):
    if i < j:  # Avoid redundant pairs
        print(f"Similar Names: {df['Company Name'][i]} and {df['Company Name'][j]}")


import pandas as pd
import Levenshtein

# Sample data (company names)
data = ['WAL MART', 'WALMART', 'AMAZON', 'AMAZON PVT LTD', 'Walmart', 'Amazon INC']

# Convert data to a DataFrame
df = pd.DataFrame(data, columns=['Company Name'])

# Function to calculate Levenshtein distance
def calculate_levenshtein(name1, name2):
    return Levenshtein.distance(name1, name2)

# Compare each pair of names
for i in range(len(df)):
    for j in range(i + 1, len(df)):
        lev_distance = calculate_levenshtein(df['Company Name'][i], df['Company Name'][j])
        if lev_distance <= 2:  # Set a threshold (e.g., distance <= 2)
            print(f"Similar Names: {df['Company Name'][i]} and {df['Company Name'][j]} (Levenshtein Distance: {lev_distance})")




from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# Sample data (company names)
data = ['WAL MART', 'WALMART', 'AMAZON', 'AMAZON PVT LTD', 'Walmart', 'Amazon INC']

# Function to apply fuzzy matching
def fuzzy_match(name, name_list, threshold=80):
    matches = []
    for other_name in name_list:
        similarity = fuzz.ratio(name, other_name)
        if similarity > threshold:  # Adjust threshold as needed
            matches.append((name, other_name, similarity))
    return matches

# Compare each name with the others
matches = []
for name in data:
    matches.extend(fuzzy_match(name, data))

# Display the matches
for match in matches:
    print(f"Similar Names: {match[0]} and {match[1]} (Similarity: {match[2]})")


from datasketch import MinHash, MinHashLSH
import pandas as pd

# Sample data (company names)
data = ['WAL MART', 'WALMART', 'AMAZON', 'AMAZON PVT LTD', 'Walmart', 'Amazon INC']

# Function to create MinHash for a string
def get_minhash(name):
    m = MinHash()
    for word in name.split():
        m.update(word.encode('utf8'))
    return m

# Create LSH index
lsh = MinHashLSH(threshold=0.8, num_perm=128)

# Insert MinHashes into the LSH index
minhashes = {}
for i, name in enumerate(data):
    minhash = get_minhash(name)
    lsh.insert(f"company_{i}", minhash)
    minhashes[f"company_{i}"] = minhash

# Query similar items using LSH
for i, name in enumerate(data):
    minhash = get_minhash(name)
    result = lsh.query(minhash)
    for r in result:
        print(f"Similar Names: {name} and {data[int(r.split('_')[1])]} (MinHash Similarity)")
