import pandas as pd
from rapidfuzz import fuzz
from itertools import islice

# Initializing the existing data
data = {
    "x": [
        "amazon flix", "amazon flrx", "amazon lt", "amazon flex",
        "amazon flex", "amazon flex", "amazon flex", "amazon dld",
        "amazon ftw", "amazon lft",
        "Amazon", "Amazon flex", "Amzn", "Amazonl", "Amazones",
        "Amazon flexz", "Amazonflez", "Amzonc", "Amazonj", "Amazon",
        "Amazonw", "Amazon flrx", "Amazones", "Amazonva", "Amazon c", "Amazon y",
        "amazon flex", "Amazon lflez", "Amazon flex", "Amazon g",
        "Amazon fulfilment cen", "Amzon full fillment xenter", "Amazon fulfilment center"
    ],
    "y": [
        "amazon xlx", "amazons flex", "amazon y", "amazon flix",
        "amazon flix", "amazon lex", "amazon xlx", "amazon dls",
        "amazon tw", "amazon litaasad",
        "Amazon", "Amazon flex", "Amzn", "Amazoan l", "Amazone a",
        "Amazon flexz", "Amazonf lez", "Amzon r", "Amazon rfd", "Amazon ada",
        "Amazon wt", "Amazon flrex", "Amazon es", "Amazon dava", "Amazon c", "Amazon y",
        "amazon flexes", "Amazon llez", "Amazon llex", "Amazon bf",
        "Amazon fulfilment center", "Amazon ful fulmetn cent", "Amzon full fillment xenter"
    ]
}
df = pd.DataFrame(data)

# Sort the DataFrame by both 'x' and 'y' columns before processing
df = df.sort_values(by=['x', 'y'], ascending=[True, True]).reset_index(drop=True)

# Define similarity threshold
THRESHOLD = 80

# Initialize groups as a list of sets
groups = []

# Function to add an element one at a time
def add_to_groups(x_val, y_val):
    matched_group = None

    # Convert to lowercase for case-insensitive matching
    x_val = x_val.lower()
    y_val = y_val.lower()

    # Sort groups before matching for consistency
    for group in islice(groups, len(groups)):  
        sorted_group = sorted(group, key=lambda item: item.lower())  # Sort in lowercase for consistency
        
        matches_x = [item for item in sorted_group if fuzz.ratio(x_val, item) > THRESHOLD]
        matches_y = [item for item in sorted_group if fuzz.ratio(y_val, item) > THRESHOLD]

        if len(matches_x) == len(group) or len(matches_y) == len(group):
          if len(matches_x) == len(group):
            group.update([x_val])
          if len(matches_y) == len(group):
            group.update([y_val]) 
          matched_group =True
          break

    if not matched_group:
        groups.append(set([x_val, y_val]))  # Create a new group with both x_val and y_val

# Loop over the rows of the sorted dataframe and add each x, y pair to the groups
for x_val, y_val in zip(df['x'], df['y']):
    add_to_groups(x_val, y_val)

# Print the grouped results
for idx, group in enumerate(groups, 1):
    print(f"Group {idx}: {sorted(group)}")  # Print sorted groups for consistency


++++++++

df = df.sort_values(by=['x', 'y'], ascending=[True, True]).reset_index(drop=True)

# Define similarity threshold
THRESHOLD = 80

# Initialize groups as a list of sets
groups = []

# Function to add an element one at a time
def add_to_groups(x_val, y_val):
    matched_group = None

    # Convert to lowercase for case-insensitive matching
    x_val = x_val.lower()
    y_val = y_val.lower()

    # Create a list of all elements in existing groups for comparison
    group_items = [item for group in groups for item in group]

    # Use fuzzywuzzy process.extract to get the closest match for x_val and y_val from the group items
    best_match_x = process.extractOne(x_val, group_items, scorer=fuzz.ratio)
    best_match_y = process.extractOne(y_val, group_items, scorer=fuzz.ratio)

    # Filter groups based on the best match
    filtered_groups = []
    if best_match_x and best_match_x[1] > THRESHOLD:
        # Filter groups where a match is found for x_val
        filtered_groups = [group for group in groups if best_match_x[0] in group]
    
    elif best_match_y and best_match_y[1] > THRESHOLD:
        # Filter groups where a match is found for y_val
        filtered_groups = [group for group in groups if best_match_y[0] in group]

    # Sort groups before matching for consistency
    for group in islice(filtered_groups, len(filtered_groups)):  
        sorted_group = sorted(group, key=lambda item: item.lower())  # Sort in lowercase for consistency
        
        matches_x = [item for item in sorted_group if fuzz.ratio(x_val, item) > THRESHOLD]
        matches_y = [item for item in sorted_group if fuzz.ratio(y_val, item) > THRESHOLD]

        if len(matches_x) == len(group) or len(matches_y) == len(group):
          if len(matches_x) == len(group):
            group.update([x_val])
          if len(matches_y) == len(group):
            group.update([y_val]) 
          matched_group =True
          break

    if not matched_group:
        groups.append(set([x_val, y_val]))  # Create a new group with both x_val and y_val

# Loop over the rows of the sorted dataframe and add each x, y pair to the groups
for x_val, y_val in zip(df['x'], df['y']):
    add_to_groups(x_val, y_val)

# Print the grouped results
for idx, group in enumerate(groups, 1):
    print(f"Group {idx}: {sorted(group)}") 



+++++++

def generate_char_ngrams(text):
    text = text.replace(" ", "").replace(",", "")  # Remove spaces and commas
    ngrams = set()  # Use a set to avoid duplicate n-grams

    for n in range(1, len(text) + 1):  # From 1-gram to full length
        for i in range(len(text) - n + 1):
            ngrams.add(text[i:i + n])
    
    return list(ngrams)



from thefuzz import fuzz
import pandas as pd

# Sample data
data = {'company_name': [
    "AMAZON13 UNITED PARCEL AND WALMART", 
    "AMAZON UBER", 
    "AMAZON,UBER EATS", 
    "DOORDASH", 
    "INSTACARAMAZON", 
    "AMAZONFLEX UBER AND ALSO INSTACART"
]}
df = pd.DataFrame(data)

# List of employer names
employer_names = ["AMAZON", "UBER", "UBER EATS", "DOORDASH", "INSTACART", "AMAZON FLEX", "WALMART"]

# Function to generate one-token and two-token combinations
def generate_tokens(text):
    words = text.replace(',', ' ').split()  # Split on spaces and commas
    one_tokens = words  # Single words
    two_tokens = [' '.join(pair) for pair in zip(words, words[1:])]  # Consecutive word pairs
    return one_tokens, two_tokens

# Function to find best matches
def get_best_matches(text, employer_names, threshold=50):
    one_tokens, two_tokens = generate_tokens(text)
    
    best_matches = {}  # Dictionary to track the max score for each employer

    # Check one-token matches
    for token in one_tokens:
        for employer in employer_names:
            score = fuzz.partial_ratio(employer, token)
            if score > threshold:
                if employer not in best_matches or score > best_matches[employer]["partial_ratio"]:
                    best_matches[employer] = {"matched": employer, "partial_ratio": score, "original_token": token}

    # Check two-token matches
    for token in two_tokens:
        for employer in employer_names:
            score = fuzz.partial_ratio(employer, token)
            if score > threshold:
                if employer not in best_matches or score > best_matches[employer]["partial_ratio"]:
                    best_matches[employer] = {"matched": employer, "partial_ratio": score, "original_token": token}

    return list(best_matches.values())  # Convert to list of dictionaries

# Apply function and store results in a new column
df['match_dict'] = df['company_name'].apply(lambda x: get_best_matches(x, employer_names))

# Display the DataFrame
print(df[['company_name', 'match_dict']])



import pandas as pd
import re
from rapidfuzz import fuzz

# Sample data
data = {'company_name': [
    "AMAZON13 UNITED PARCEL AND WALMART", 
    "AMAZON UBER", 
    "AMAZON,UBER EATS", 
    "DOORDASH", 
    "INSTACARAMAZON", 
    "AMAZONFLEX UBER AND ALSO INSTACART"
]}
df = pd.DataFrame(data)

# List of employer names
employer_names = ["AMAZON", "UBER", "UBER EATS", "DOORDASH", "INSTACART", "AMAZON FLEX", "WALMART"]

# Preprocessing function for fuzzy matching
def preprocess(text):
    return re.sub(r'[^\w\s]', '', text).strip().lower()

# Function to find matched substrings and split company_name based on matches
def split_company_names(df, employer_names, threshold=80):
    matched_lists = []
    unmatched_texts = []

    for company_name in df['company_name']:
        matched_substrings = set()  # Store unique matched substrings
        words = re.split(r'[,\s/&]+', company_name)  # Split on spaces, commas, slashes, &, etc.
        
        for employer in employer_names:
            clean_employer = preprocess(employer)
            
            # Search for an exact match in company_name
            for word in words:
                if fuzz.partial_ratio(preprocess(word), clean_employer) >= threshold:
                    matched_substrings.add(word)  # Store the exact matched substring

            # Check entire company_name with fuzzy matching
            match_ratio = fuzz.ratio(preprocess(company_name), clean_employer)
            partial_match = fuzz.partial_ratio(preprocess(company_name), clean_employer)
            token_set_match = fuzz.token_set_ratio(preprocess(company_name), clean_employer)

            if max(match_ratio, partial_match, token_set_match) >= threshold:
                if employer in company_name:
                    matched_substrings.add(employer)  # Store exact match if found
        
        matched_substrings = list(matched_substrings)  # Convert set to list
        
        # Remove matched substrings from the original text
        unmatched_text = company_name
        for match in matched_substrings:
            unmatched_text = unmatched_text.replace(match, "").strip()

        # Append results
        matched_lists.append(matched_substrings)
        unmatched_texts.append(unmatched_text)

    # Add results to DataFrame
    df['matched_employers'] = matched_lists
    df['remaining_text'] = unmatched_texts

    return df

# Apply function and store results in DataFrame
df = split_company_names(df, employer_names)

# Display the result
df


import pandas as pd
import recordlinkage

# Sample data
data = {import pandas as pd
from collections import defaultdict

# Sample data
data = {
    'id': [1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 8, 8, 7],
    'employer_name': [
        "COSTEX TRACTOR PART", "ABC CORP", 
        "C&J TECH", "C & J TECH", "C & J", 
        "BLA", "BLABLA", 
        "ABC CORP", "COSTEX TRACTOR PART", "BLABLA",
        "COSTEX TRACTOR PART", "XYZ INC", "ABC CORP",
        "COSTEX", "DEF", 
        "C&J TECH", "C & J TECH",
        "C J TECH", "C & J TECH", "C & J TECH"
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Function to count pairs of employer names and individual counts
def count_pairs_and_individual_counts(df):
    pair_counts = defaultdict(int)
    individual_counts = defaultdict(int)
    
    # Iterate through each 'id' and get all unique pairs of employer names
    for _, group in df.groupby('id'):
        employers = group['employer_name'].tolist()
        
        # Count individual occurrences
        for employer in employers:
            individual_counts[employer] += 1

        # Count pairs
        for i in range(len(employers)):
            for j in range(i + 1, len(employers)):
                pair = tuple(sorted([employers[i], employers[j]]))  # Sort to avoid order-sensitive pairs
                pair_counts[pair] += 1
    
    return pair_counts, individual_counts

# Count pairs and individual counts
pair_counts, individual_counts = count_pairs_and_individual_counts(df)

# Filter employers with total occurrences > 3
filtered_employers = {employer for employer, count in individual_counts.items() if count > 3}

# Display the filtered individual counts
print("Filtered Employers (Total Individual Counts > 3):")
filtered_individual_counts = {k: v for k, v in individual_counts.items() if v > 3}
print(filtered_individual_counts)

# Union-Find (Disjoint Set Union - DSU) to group connected employers
class UnionFind:
    def __init__(self, elements):
        self.parent = {e: e for e in elements}
        self.rank = {e: 0 for e in elements}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)

        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1

# Initialize UnionFind for the filtered employer names
uf = UnionFind(filtered_employers)

# Union employers that share a pair
for pair, count in pair_counts.items():
    employer1, employer2 = pair
    if employer1 in filtered_employers and employer2 in filtered_employers:
        uf.union(employer1, employer2)

# Group filtered employers based on their connected components
groups = defaultdict(list)
for employer in filtered_employers:
    root = uf.find(employer)
    groups[root].append(employer)

# Final result: List of connected groups of employer names
final_groups = list(groups.values())

# Show the final grouped result
print("\nFinal Employer Groups (Filtered by Count > 3):")
print(final_groups)







    "ID": [1, 2, 3, 4, 5, 6],
    "identity_id": ["A1", "A2", "A3", "B1", "B2", "C1"],
    "Employer Name": ["Google LLC", "Google Inc.", "Google", "Amazon", "Amazon.com", "Microsoft"],
    "State": ["CA", "CA", "CA", "NY", "NY", "WA"],
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Indexing: Generate potential pairs with blocking on 'State'
indexer = recordlinkage.Index()
indexer.block("State")  # Compare only pairs within the same state
candidate_links = indexer.index(df, df)

# Comparison: Define similarity rules
compare = recordlinkage.Compare()
compare.string("Employer Name", "Employer Name", method="levenshtein", threshold=0.8, label="name_similarity")
compare.exact("State", "State", label="state_match")  # Ensure the state matches
compare.exact("identity_id", "identity_id", label="identity_id_match")  # Exact match for identity_id

# Compute similarity scores
features = compare.compute(candidate_links, df)

# Filter: Retain pairs with high similarity
matches = features[(features["name_similarity"] == 1) & (features["state_match"] == 1)].reset_index()

# Map IDs and additional columns from original data
matches = matches.merge(df[["ID", "identity_id"]].rename(columns={"ID": "ID_1", "identity_id": "identity_id_1"}), left_on="level_0", right_index=True)
matches = matches.merge(df[["ID", "identity_id"]].rename(columns={"ID": "ID_2", "identity_id": "identity_id_2"}), left_on="level_1", right_index=True)

# Add Cluster IDs
matches["Cluster ID"] = matches.index + 1  # Assign unique cluster IDs

# Final output
print(matches[["ID_1", "identity_id_1", "ID_2", "identity_id_2", "Cluster ID"]])





import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# Sample data
data = {
    "Employer Name 1": ["Employer A", "Employer B", "Employer C", "Employer D"],
    "Employer Name 2": ["Employer B", "Employer C", "Employer D", "Employer A"],
    "Similarity Score": [0.9, 0.85, 0.7, 0.95],
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Create a similarity matrix
names = list(set(df["Employer Name 1"]).union(set(df["Employer Name 2"])))
similarity_matrix = pd.DataFrame(0, index=names, columns=names)

for _, row in df.iterrows():
    similarity_matrix.loc[row["Employer Name 1"], row["Employer Name 2"]] = row["Similarity Score"]
    similarity_matrix.loc[row["Employer Name 2"], row["Employer Name 1"]] = row["Similarity Score"]

# Convert similarity to distance (1 - similarity)
distance_matrix = 1 - similarity_matrix.values

# Apply Agglomerative Clustering
n_clusters = 2  # Set the number of clusters (can be adjusted)
agg_cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity="precomputed", linkage="average")
cluster_labels = agg_cluster.fit_predict(distance_matrix)

# Map clusters back to employer names
cluster_mapping = dict(zip(names, cluster_labels))

# Add Cluster IDs to the DataFrame
df["Cluster ID 1"] = df["Employer Name 1"].map(cluster_mapping)
df["Cluster ID 2"] = df["Employer Name 2"].map(cluster_mapping)

# Visualize Dendrogram
linked = linkage(distance_matrix, method="average")
plt.figure(figsize=(8, 4))
dendrogram(linked, labels=names, orientation="top", distance_sort="descending", show_leaf_counts=True)
plt.title("Dendrogram")
plt.show()

print(df)








import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

# Sample data
data = {
    "Employer Name 1": ["Employer A", "Employer B", "Employer C", "Employer D"],
    "Employer Name 2": ["Employer B", "Employer C", "Employer D", "Employer A"],
    "Similarity Score": [0.9, 0.85, 0.7, 0.95],
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Prepare a similarity matrix
names = list(set(df["Employer Name 1"]).union(set(df["Employer Name 2"])))
similarity_matrix = pd.DataFrame(0, index=names, columns=names)

for _, row in df.iterrows():
    similarity_matrix.loc[row["Employer Name 1"], row["Employer Name 2"]] = row["Similarity Score"]
    similarity_matrix.loc[row["Employer Name 2"], row["Employer Name 1"]] = row["Similarity Score"]

# Convert similarity to distance (1 - similarity)
distance_matrix = 1 - similarity_matrix.values

# Apply DBSCAN
epsilon = 0.3  # Maximum distance for clusters (equivalent to similarity of 0.7)
min_samples = 1  # Minimum points to form a cluster
dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, metric="precomputed")
clusters = dbscan.fit_predict(distance_matrix)

# Map clusters back to employer names
cluster_mapping = dict(zip(names, clusters))

# Assign Cluster IDs to the DataFrame
df["Cluster ID 1"] = df["Employer Name 1"].map(cluster_mapping)
df["Cluster ID 2"] = df["Employer Name 2"].map(cluster_mapping)

print(df)






import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# Sample data
data = {
    "Employer Name 1": ["Employer A", "Employer B", "Employer C", "Employer D"],
    "Employer Name 2": ["Employer B", "Employer C", "Employer D", "Employer A"],
    "Similarity Score": [0.9, 0.85, 0.7, 0.95],
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Create a similarity matrix
names = list(set(df["Employer Name 1"]).union(set(df["Employer Name 2"])))
similarity_matrix = pd.DataFrame(0, index=names, columns=names)

for _, row in df.iterrows():
    similarity_matrix.loc[row["Employer Name 1"], row["Employer Name 2"]] = row["Similarity Score"]
    similarity_matrix.loc[row["Employer Name 2"], row["Employer Name 1"]] = row["Similarity Score"]

# Convert similarity to distance (1 - similarity)
distance_matrix = 1 - similarity_matrix

# Perform hierarchical clustering
linkage_matrix = linkage(squareform(distance_matrix), method="average")
threshold = 0.3  # Distance threshold (equivalent to a similarity of 0.7)
clusters = fcluster(linkage_matrix, threshold, criterion="distance")

# Map clusters back to employers
cluster_mapping = dict(zip(names, clusters))

# Add Cluster IDs to the DataFrame
df["Cluster ID 1"] = df["Employer Name 1"].map(cluster_mapping)
df["Cluster ID 2"] = df["Employer Name 2"].map(cluster_mapping)

print(df)
















import pandas as pd
import networkx as nx

# Sample data
data = {
    "Employer Name 1": ["Employer A", "Employer B", "Employer C", "Employer D"],
    "Employer Name 2": ["Employer B", "Employer C", "Employer D", "Employer A"],
    "Similarity Score": [0.9, 0.85, 0.7, 0.95],
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Create a graph
threshold = 0.8  # Define the similarity threshold
G = nx.Graph()

# Add edges for pairs above the threshold
for _, row in df.iterrows():
    if row["Similarity Score"] >= threshold:
        G.add_edge(row["Employer Name 1"], row["Employer Name 2"])

# Find connected components (clusters)
clusters = list(nx.connected_components(G))

# Assign Cluster IDs
cluster_mapping = {}
for cluster_id, cluster in enumerate(clusters):
    for employer in cluster:
        cluster_mapping[employer] = cluster_id

# Add Cluster ID to the DataFrame
df["Cluster ID 1"] = df["Employer Name 1"].map(cluster_mapping)
df["Cluster ID 2"] = df["Employer Name 2"].map(cluster_mapping)

print(df)





















import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz
import re

# Function to find matches using token_set_ratio
def find_matches(input_string, comparison_list, threshold=80):
    """
    Finds matches from the comparison list for a given input string using token_set_ratio.
    """
    delimiters = r",|/|AND"
    tokens = {name.strip() for name in re.split(delimiters, input_string) if name.strip()}

    matches = set()
    for token in tokens:
        matches.update(
            match[0]  # Keep only names from comparison list
            for match in process.extract(token, comparison_list, scorer=fuzz.token_set_ratio, score_cutoff=threshold)
        )
    
    return list(matches)

# Example DataFrame with large data
data = {"employer_name": ["Google,Amazon/Microsoft", "goog", "Amazo AND Google", "Facebook/Apple", "Amazon/Google"]}
df = pd.DataFrame(data)

# List to compare against
comparison_list = ["Google", "Amazon", "Facebook"]

# Apply the find_matches function directly to the 'employer_name' column
df['matches_from_list'] = np.array([find_matches(name, comparison_list) for name in df['employer_name'].values])

# Return the 'matches_from_list' column
print(df['matches_from_list'])



-------------------------------------------------

import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz
import re

# Example DataFrame with large data
data = {"employer_name": ["Google,Amazon/Microsoft", "goog", "Amazo AND Google", "Facebook/Apple", "Amazon/Google"]}
df = pd.DataFrame(data)

# List to compare against
comparison_list = ["Google", "Amazon", "Facebook"]

# Step 1: Split the 'employer_name' column into tokens based on delimiters using vectorized Pandas string operations
delimiters = r",|/|AND"

# Vectorized operation to split employer names into tokens
tokens_list = df['employer_name'].str.split(delimiters).apply(lambda x: [name.strip() for name in x if name.strip()])

# Step 2: Fuzzy matching using list comprehension (vectorized approach)
matches_list = [
    list(set(
        match[0] for token in tokens 
        for match in process.extract(token, comparison_list, scorer=fuzz.token_set_ratio, score_cutoff=80)
    )) 
    for tokens in tokens_list
]

# Step 3: Add the matches as a new column in the DataFrame
df['matches_from_list'] = matches_list

# Step 4: Return only the 'matches_from_list' column
print(df['matches_from_list'])




-------------------------------------------------















# Use a set to store unique pairs regardless of order
unique_pairs = set(
    tuple(sorted(pair)) for pair in zip(df['employer_name_x'], df['employer_name_y'])
)

# Convert the set back to a DataFrame
result_df = pd.DataFrame(list(unique_pairs), columns=['employer_name_x', 'employer_name_y'])

def compute_token_features(df: pd.DataFrame, col1: str, col2: str, common_words: Optional[Set[str]] = None) -> pd.DataFrame:
    """
    Compute token-based features for comparison between two text columns using list comprehensions.
    
    Args:
        df: Input DataFrame containing text columns.
        col1: Name of the first text column to compare.
        col2: Name of the second text column to compare.
        common_words: Set of precomputed common words (optional).
    
    Returns:
        A DataFrame containing token-based features.
    """
    # Default to an empty set if common_words is not provided
    common_words = common_words or set()

    # Tokenize text columns
    tokens_col1 = [set(str(text).lower().split()) for text in df[col1]]
    tokens_col2 = [set(str(text).lower().split()) for text in df[col2]]

    # Compute features using list comprehensions
    common_hit = [len(t1 & t2 & common_words) for t1, t2 in zip(tokens_col1, tokens_col2)]
    rare_hit = [len(t1 & t2 - common_words) for t1, t2 in zip(tokens_col1, tokens_col2)]
    common_miss = [len((t1 ^ t2) & common_words) for t1, t2 in zip(tokens_col1, tokens_col2)]
    rare_miss = [len((t1 ^ t2) - common_words) for t1, t2 in zip(tokens_col1, tokens_col2)]
    n_overlap_words = [len(t1 & t2) for t1, t2 in zip(tokens_col1, tokens_col2)]
    ratio_overlap_words = [
        len(t1 & t2) / len(t1 | t2) if len(t1 | t2) > 0 else 0
        for t1, t2 in zip(tokens_col1, tokens_col2)
    ]
    num_word_difference = [abs(len(t1) - len(t2)) for t1, t2 in zip(tokens_col1, tokens_col2)]

    # Create a DataFrame with the computed features
    feature_df = pd.DataFrame({
        'common_hit': common_hit,
        'rare_hit': rare_hit,
        'common_miss': common_miss,
        'rare_miss': rare_miss,
        'n_overlap_words': n_overlap_words,
        'ratio_overlap_words': ratio_overlap_words,
        'num_word_difference': num_word_difference,
    })

    return pd.concat([df, feature_df], axis=1)
# Example usage
if __name__ == "__main__":
    import pandas as pd

    data = {
        "name1": ["apple banana cherry", "dog cat", "elephant giraffe lion"],
        "name2": ["banana cherry date", "cat mouse", "lion tiger zebra"],
    }

    df = pd.DataFrame(data)
    common_words = {"banana", "cherry", "cat", "apple", "dog", "lion", "tiger"}

    result = compute_vocabulary_features(
        df, "name1", "name2", common_words=common_words
    )

    print(result)
