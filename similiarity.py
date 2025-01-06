import re
from abydos import jaro_winkler, lev, cosine, damerau_levenshtein
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords if not already downloaded
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load stop words
STOP_WORDS = set(stopwords.words('english'))

def remove_suffix(name):
    """Remove common suffixes from a company name."""
    suffixes = ["Inc", "Corp", "LLC", "Ltd", "Co", "Company"]
    pattern = r"\b(?:" + "|".join(suffixes) + r")\b"
    return re.sub(pattern, "", name, flags=re.IGNORECASE).strip()

def remove_stop_words(text):
    """Remove stop words from a given text."""
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in STOP_WORDS]
    return " ".join(filtered_words)

def preprocess_name(name, remove_stop_words_flag=False):
    """Combine preprocessing steps (suffix removal and stopword removal)."""
    name = remove_suffix(name)
    if remove_stop_words_flag:
        name = remove_stop_words(name)
    return name.strip()

def compute_first_word_ratio(name1, name2):
    """Compute similarity ratio of the first words using Fuzzy Ratio."""
    words1 = name1.split()
    words2 = name2.split()
    if words1 and words2:
        return lev.levenshtein(words1[0], words2[0])  # Use Levenshtein distance for the first words
    return 0

def compute_cosine_similarity(name1, name2):
    """Compute cosine similarity using TF-IDF Vectorizer (character bigrams)."""
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 2))
    tfidf_matrix = vectorizer.fit_transform([name1, name2])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return round(cosine_sim[0][0] * 100, 2)

def token_sort_ratio(name1, name2):
    """Token sort ratio calculation."""
    return lev.levenshtein(name1, name2)  # Levenshtein for sorted token comparison

def token_set_ratio(name1, name2):
    """Token set ratio calculation."""
    return jaro_winkler(name1, name2)  # Jaro-Winkler for set-based token comparison

def partial_ratio(name1, name2):
    """Partial ratio using substrings."""
    return damerau_levenshtein(name1, name2)  # Damerau-Levenshtein for partial similarity

def ratio(name1, name2):
    """Standard Abydos Levenshtein ratio."""
    return lev.levenshtein(name1, name2)

def word_level_similarity(name1, name2):
    """Calculate word-level similarity using Levenshtein distance for each word pair."""
    words1 = name1.split()
    words2 = name2.split()
    
    word_matches = []
    for word1 in words1:
        for word2 in words2:
            similarity = lev.levenshtein(word1, word2)
            if similarity < 5:  # Small Levenshtein distance means more similar
                word_matches.append(similarity)
    
    if word_matches:
        return sum(word_matches) / len(word_matches)
    return 0

def character_level_similarity(name1, name2):
    """Character-level similarity using Levenshtein distance."""
    return lev.levenshtein(name1, name2)

def common_substrings(name1, name2):
    """Count common substrings between two names."""
    min_len = min(len(name1), len(name2))
    common_substrs = set()
    
    # Generate all substrings of the first name
    for i in range(min_len):
        for j in range(i + 1, min_len + 1):
            substring = name1[i:j]
            if substring in name2:
                common_substrs.add(substring)
    
    return len(common_substrs)

def compute_adjacent_pair_similarity(name1, name2):
    """Compute similarity for adjacent word pairs in both names."""
    words1 = name1.split()
    words2 = name2.split()

    # Find adjacent word pairs for both names
    adjacent_pairs1 = [(words1[i], words1[i+1]) for i in range(len(words1)-1)]
    adjacent_pairs2 = [(words2[i], words2[i+1]) for i in range(len(words2)-1)]

    similarity_scores = []
    
    # Compare adjacent pairs
    for pair1 in adjacent_pairs1:
        for pair2 in adjacent_pairs2:
            similarity = lev.levenshtein(" ".join(pair1), " ".join(pair2))
            similarity_scores.append(similarity)

    if similarity_scores:
        return sum(similarity_scores) / len(similarity_scores)
    return 0

def compute_similarity(name1, name2, state1=None, state2=None):
    """
    Compute similarity score between two company names and factor in their state.

    :param name1: First company name
    :param name2: Second company name
    :param state1: State of the first company
    :param state2: State of the second company
    :return: A dictionary of similarity scores
    """
    
    # If states are provided, apply region-based weighting
    state_bonus = 1.2 if state1 == state2 else 1.0

    # Calculate partial ratio first
    fuzzy_partial_score = lev.levenshtein(name1, name2)

    # Determine if we should remove stop words and acronyms
    if fuzzy_partial_score > 0.80:
        remove_stop_words_flag = True
    else:
        remove_stop_words_flag = False

    # Preprocessing for both "with suffix" and "without suffix"
    name1_with_suffix = preprocess_name(name1, remove_stop_words_flag)
    name2_with_suffix = preprocess_name(name2, remove_stop_words_flag)
    
    name1_without_suffix = remove_suffix(name1)
    name2_without_suffix = remove_suffix(name2)
    name1_without_suffix = preprocess_name(name1_without_suffix, remove_stop_words_flag)
    name2_without_suffix = preprocess_name(name2_without_suffix, remove_stop_words_flag)

    # Similarity Scores for "with suffix"
    first_word_ratio_with = compute_first_word_ratio(name1_with_suffix, name2_with_suffix)
    cosine_sim_with = compute_cosine_similarity(name1_with_suffix, name2_with_suffix)
    token_sort_with = token_sort_ratio(name1_with_suffix, name2_with_suffix)
    token_set_with = token_set_ratio(name1_with_suffix, name2_with_suffix)
    partial_with = partial_ratio(name1_with_suffix, name2_with_suffix)
    standard_ratio_with = ratio(name1_with_suffix, name2_with_suffix)
    word_sim_with = word_level_similarity(name1_with_suffix, name2_with_suffix)
    char_sim_with = character_level_similarity(name1_with_suffix, name2_with_suffix)
    common_substr_count_with = common_substrings(name1_with_suffix, name2_with_suffix)
    adj_pair_sim_with = compute_adjacent_pair_similarity(name1_with_suffix, name2_with_suffix)

    # Similarity Scores for "without suffix"
    first_word_ratio_without = compute_first_word_ratio(name1_without_suffix, name2_without_suffix)
    cosine_sim_without = compute_cosine_similarity(name1_without_suffix, name2_without_suffix)
    token_sort_without = token_sort_ratio(name1_without_suffix, name2_without_suffix)
    token_set_without = token_set_ratio(name1_without_suffix, name2_without_suffix)
    partial_without = partial_ratio(name1_without_suffix, name2_without_suffix)
    standard_ratio_without = ratio(name1_without_suffix, name2_without_suffix)
    word_sim_without = word_level_similarity(name1_without_suffix, name2_without_suffix)
    char_sim_without = character_level_similarity(name1_without_suffix, name2_without_suffix)
    common_substr_count_without = common_substrings(name1_without_suffix, name2_without_suffix)
    adj_pair_sim_without = compute_adjacent_pair_similarity(name1_without_suffix, name2_without_suffix)

    # Final similarity calculation considering common substrings and adjacent pairs
    final_similarity_with_suffix = (
        (first_word_ratio_with + cosine_sim_with + token_sort_with + token_set_with + partial_with +
         standard_ratio_with + word_sim_with + char_sim_with) / 8) * state_bonus + common_substr_count_with + adj_pair_sim_with

    final_similarity_without_suffix = (
        (first_word_ratio_without + cosine_sim_without + token_sort_without + token_set_without + partial_without +
         standard_ratio_without + word_sim_without + char_sim_without) / 8) * state_bonus + common_substr_count_without + adj_pair_sim_without

    # Return the results for both cases
    return {
        "With Suffix - Final Similarity": final_similarity_with_suffix,
        "Without Suffix - Final Similarity": final_similarity_without_suffix,
    }

# Example Usage
name1 = "The MatchIt Inc"
name2 = "MatchIt Company in City"
state1 = "California"
state2 = "California"  # Same state
similarity_scores = compute_similarity(name1, name2, state1, state2)

# Print Results
if isinstance(similarity_scores, dict):  # If the result is not a string (i.e., similarity calculation is not skipped)
    for metric, score in similarity_scores.items():
        print(f"{metric}: {score}")
else:
    print(similarity_scores)
