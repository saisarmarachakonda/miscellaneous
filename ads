def compute_vocabulary_features(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    common_words: set[str] | None = None,
) -> pd.DataFrame:
    """Compute token-based features for comparison between two text columns.

    Args:
        df: Input DataFrame containing text columns.
        col1: Name of the first text column to compare.
        col2: Name of the second text column to compare.
        common_words: Set of precomputed common words (optional).

    Returns:
        A DataFrame containing token-based features, including:
        - common_hit: Count of common words present in both columns.
        - rare_hit: Count of rare words present in both columns.
        - common_miss: Count of common words present in only one column.
        - rare_miss: Count of rare words present in only one column.
        - n_overlap_words: Number of overlapping words between the two columns.
        - ratio_overlap_words: Ratio of overlapping words to total unique words.
        - num_word_difference: Absolute difference in the number of words between the two columns.
    """
    assert common_words is None or isinstance(common_words, set)

    name1 = df[col1]
    name2 = df[col2]

    # Tokenize words and convert to sets
    word_set1 = name1.str.findall(r"\w\w+").map(set)
    word_set2 = name2.str.findall(r"\w\w+").map(set)

    # Compute hits and misses
    hits = pd.Series(word_set1.values & word_set2.values, index=word_set1.index)
    total_words = pd.Series(word_set1.values | word_set2.values, index=word_set1.index)
    misses = total_words - hits

    # Initialize word categories
    common_words = common_words or set()

    # Calculate feature counts using list comprehensions
    common_hits = [sum(1 for word in hit if word in common_words) for hit in hits]
    rare_hits = [sum(1 for word in hit if word not in common_words) for hit in hits]

    common_misses = [sum(1 for word in miss if word in common_words) for miss in misses]
    rare_misses = [sum(1 for word in miss if word not in common_words) for miss in misses]

    # Compute summary statistics
    n_hits = hits.map(len)
    n_total = total_words.map(len)
    n_set1 = word_set1.map(len)
    n_set2 = word_set2.map(len)
    ratio_overlap = (n_hits / n_total).replace(np.inf, 0)

    # Create and return feature DataFrame
    return pd.DataFrame(
        {
            "common_hit": common_hits,
            "rare_hit": rare_hits,
            "common_miss": common_misses,
            "rare_miss": rare_misses,
            "n_overlap_words": n_hits,
            "ratio_overlap_words": ratio_overlap,
            "num_word_difference": (n_set1 - n_set2).abs(),
        },
        dtype="float32",
    )

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
