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
