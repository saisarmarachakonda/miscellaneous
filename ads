import pandas as pd
from rapidfuzz import fuzz, process
import re

# Example DataFrame
data = {
    "employer_name": ["Google,Amazon/Microsoft", "goog", "Amazo AND Google"]
}
df = pd.DataFrame(data)

# List to compare against
comparison_list = ["Google", "Amazon", "Facebook"]

# Function to compute all matches with multiple delimiters
def find_all_matches(value, comparison_list, threshold=80):
    # Use regex to split by multiple delimiters
    names = re.split(r",|/|AND", value)
    return {
        name.strip(): [
            (match[0], match[1])  # Extract matched name and score
            for match in process.extract(name.strip(), comparison_list, scorer=fuzz.partial_ratio, score_cutoff=threshold)
        ]
        for name in names if name.strip()  # Skip empty names
    }

# Apply the function to the column
df["all_matches"] = df["employer_name"].apply(lambda x: find_all_matches(x, comparison_list))

# Flatten the results into a simple list (optional)
df["flat_matches"] = df["all_matches"].apply(
    lambda matches: [(key, match[0], match[1]) for key, values in matches.items() for match in values]
)

# Display the results
print(df[["employer_name", "flat_matches"]])



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
