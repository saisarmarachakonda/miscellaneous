
# Ensure the column is of string type
df = df.withColumn("employer_name", df["employer_name"].cast("string"))

# Tokenize employer names into words
tokenizer = Tokenizer(inputCol="employer_name", outputCol="words")
df_tokenized = tokenizer.transform(df)  # Transforming the DataFrame

# Define a function to calculate Jaccard similarity between two lists
def jaccard_similarity(list1, list2):
    # Convert the lists to sets
    set1 = set(list1)
    set2 = set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0.0

# Register the UDF
jaccard_udf = udf(jaccard_similarity, DoubleType())

# Create a self-join to compare all employer names with each other
df_with_similarity = df_tokenized.alias("df1").crossJoin(df_tokenized.alias("df2"))

# Apply Jaccard similarity UDF on pairs of employer names
similarity_df = df_with_similarity.withColumn(
    "similarity",
    jaccard_udf(col("df1.words"), col("df2.words"))
)

# Filter out pairs that are very similar (choose your threshold, e.g., 0.8)
threshold = 0.8
similar_pairs = similarity_df.filter(col("similarity") > threshold)

# Show similar employer name pairs
similar_pairs.select("df1.employer_name", "df2.employer_name", "similarity").show()


############33

from pyspark.sql.functions import col, levenshtein

# Filter based on Levenshtein distance (e.g., less than 3 edit operations)
filtered_similar_pairs = similar_pairs_filtered.filter(
    (col("similarity") > 0.8) & 
    (levenshtein(col("datasetA.employer_name"), col("datasetB.employer_name")) < 3)
)

filtered_similar_pairs.select(
    col("datasetA.employer_name").alias("Name1"),
    col("datasetB.employer_name").alias("Name2"),
    col("similarity")
).show(truncate=False)



#########
from pyspark.sql.functions import soundex

# Add Soundex column for employer names
df_with_soundex = spark_df.withColumn("soundex_code", soundex(col("employer_name")))

# Self-join on Soundex codes
similar_pairs = df_with_soundex.alias("df1").join(
    df_with_soundex.alias("df2"),
    col("df1.soundex_code") == col("df2.soundex_code")
)

similar_pairs.select(
    col("df1.employer_name").alias("Name1"),
    col("df2.employer_name").alias("Name2"),
    col("df1.soundex_code").alias("SoundexCode")
).show(truncate=False)



##########
from pyspark.ml.feature import Tokenizer, HashingTF, MinHashLSH
from pyspark.sql.functions import col

# Tokenize the employer names
tokenizer = Tokenizer(inputCol="employer_name", outputCol="tokens")
tokenized_df = tokenizer.transform(spark_df)

# Apply HashingTF
hashing_tf = HashingTF(inputCol="tokens", outputCol="features", numFeatures=10000)
tf_df = hashing_tf.transform(tokenized_df)

# Apply MinHashLSH for approximate similarity
minhash_lsh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=3)
lsh_model = minhash_lsh.fit(tf_df)

# Find similar pairs
similar_pairs = lsh_model.approxSimilarityJoin(tf_df, tf_df, 0.8, distCol="similarity")

# Filter out self-matches
similar_pairs_filtered = similar_pairs.filter(
    col("datasetA.employer_name") != col("datasetB.employer_name")
)

similar_pairs_filtered.select(
    col("datasetA.employer_name").alias("Name1"),
    col("datasetB.employer_name").alias("Name2"),
    col("similarity")
).show(truncate=False)


###############

from pyspark.ml.feature import HashingTF, IDF
from pyspark.sql.functions import col

# Tokenize and hash the employer names
tokenizer = Tokenizer(inputCol="employer_name", outputCol="tokens")
tokenized_df = tokenizer.transform(spark_df)

hashing_tf = HashingTF(inputCol="tokens", outputCol="raw_features", numFeatures=10000)
tf_df = hashing_tf.transform(tokenized_df)

idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(tf_df)
tfidf_df = idf_model.transform(tf_df)

# Self-join for cosine similarity
similar_pairs = tfidf_df.alias("df1").crossJoin(tfidf_df.alias("df2"))
similar_pairs = similar_pairs.withColumn(
    "cosine_similarity",
    (col("df1.features").dot(col("df2.features"))) / 
    (col("df1.features").norm(2) * col("df2.features").norm(2))
)

# Filter based on similarity > 0.8
similar_pairs_filtered = similar_pairs.filter(
    (col("cosine_similarity") > 0.8) & 
    (col("df1.employer_name") != col("df2.employer_name"))
)

similar_pairs_filtered.select(
    col("df1.employer_name").alias("Name1"),
    col("df2.employer_name").alias("Name2"),
    col("cosine_similarity")
).show(truncate=False)


############
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import HashingTF

# Tokenize and hash employer names
tokenizer = Tokenizer(inputCol="employer_name", outputCol="tokens")
tokenized_df = tokenizer.transform(spark_df)

hashing_tf = HashingTF(inputCol="tokens", outputCol="features", numFeatures=10000)
tf_df = hashing_tf.transform(tokenized_df)

# Apply KMeans clustering
kmeans = KMeans(featuresCol="features", predictionCol="cluster", k=5)
model = kmeans.fit(tf_df)
clustered_df = model.transform(tf_df)

# Show clustered data
clustered_df.select("employer_name", "cluster").show(truncate=False)






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
