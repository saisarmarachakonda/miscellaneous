import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from rapidfuzz import fuzz, process
from abydos.distance import Editex, MRA
from spellchecker import SpellChecker
import itertools
from collections import defaultdict

from spellchecker import SpellChecker
import nltk
from nltk.stem import WordNetLemmatizer
from collections import Counter

from functools import partial


# nltk.download("wordnet")

# Initialize SpellChecker and Lemmatizer
spell = SpellChecker()
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
editex = Editex()
mra = MRA()

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

known_words = spell.known(spell.word_frequency.keys())
# known_words = {lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in known_words}


specific_words = {'lex'}

known_words = known_words | specific_words


def preprocess_company_name(name):
    name = name.lower()
    tokens = nltk.word_tokenize(name)
    if 'lex' in tokens:
        print()

    processed_tokens = []
    for word in tokens:
        if word in spell.known([word]):  # Check if it's a known word
            processed_tokens.append(lemmatizer.lemmatize(word, get_wordnet_pos(word)))
        else:
            processed_tokens.append(word)  # Keep unknown words as they are

    return " ".join(processed_tokens)

def check_wordnet_presence(word):
    if len(word) < 3:
        return True
    wordnet_presence =  word in known_words
    return wordnet_presence

from nltk.corpus import wordnet

# Function to calculate synonym similarity percentage
def check_synonym_match(word1, word2, threshold=90):
    synsets1 = wordnet.synsets(word1)
    synsets2 = wordnet.synsets(word2)

    if not synsets1 or not synsets2:
        return False

    # Compute the max Wu-Palmer similarity
    max_similarity = max(
        (s1.wup_similarity(s2) or 0) for s1 in synsets1 for s2 in synsets2
    ) * 100  # Convert to percentage

    print(max_similarity)
    return max_similarity >= threshold  

# def lists_have_same_known_words(list1, list2):
#     """
#     Step 1: Filter both lists to only include words present in raw_known_words and ignore words < 3 chars.
#     Step 2: Compare the filtered sets. Return True if they are the same, otherwise False.
#     """
#     filtered_list1 = {word for word in list1 if len(word) >= 3 and word in known_words}
#     filtered_list2 = {word for word in list2 if len(word) >= 3 and word in known_words}
    
#     return filtered_list1 == filtered_list2


def is_phonetically_similar(list1, list2):
    def phonetic_similarity(word1, word2):
        length = min(len(word1), len(word2))
        
        # Adjust thresholds dynamically based on length
        if length <= 3:
            edtx_thresh = 0.6  
        elif length <= 4:
            edtx_thresh = 0.7  # Stricter for short words
        elif length <= 8:
            edtx_thresh = 0.6  # Moderate
        else:
            edtx_thresh = 0.5  # More lenient for longer words

        edtx_sim = editex.sim(word1, word2)

        return edtx_sim >= edtx_thresh

    def find_best_match(word, candidates):
        """Find the best phonetically similar match for a word in a list of candidates."""
        for candidate in candidates:
            if phonetic_similarity(word, candidate):
                return True
        return False

    # Filter out two-letter words
    filtered_list1 = [word for word in list1 if len(word) > 2]
    filtered_list2 = [word for word in list2 if len(word) > 2]

    # Ensure all words in list1 have a similar match in list2
    for word in filtered_list1:
        if not find_best_match(word, filtered_list2):
            return False

    return True 


def find_best_matching_groups(new_company, company_groups):
    new_company_norm = preprocess_company_name(new_company)
    results = process.extract(new_company_norm, company_groups.keys(), scorer=fuzz.ratio, limit=None)
    return [match[0] for match in results if match[1] >= 75]



def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return "a"  # Adjective
    elif tag.startswith("V"):
        return "v"  # Verb
    elif tag.startswith("N"):
        return "n"  # Noun
    else:
        return "n"  # Default to noun if not recognized


def lemmatize_phrase(phrase):
    words = phrase.split()
    pos_tags = nltk.pos_tag(words)
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    return " ".join(lemmatized_words)

def process_text(original_word):
    tokens = original_word.split()
    processed_tokens = tokens[:1]  # Always keep the first token and lemmatize it
    for token in tokens[1:]:
        lemma = lemmatizer.lemmatize(token)
        if lemma in known_words and len(lemma) > 2:
            processed_tokens.append(lemma)
    return " ".join(processed_tokens)


def get_name(phrases):
    # Lemmatize phrases
    lemmatized_phrases = {
        phrase: " ".join(lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in phrase.split()) 
        for phrase in phrases
    }

    # Extract words and count valid ones
    words = list(itertools.chain(*[phrase.split() for phrase in lemmatized_phrases.values()]))
    valid_word_counts = Counter(word for word in words if word in known_words)

    # If no valid words are found, return the longest phrase
    if not valid_word_counts:
        return max(phrases, key=lambda phrase: len(phrase.split()))

    # Score each phrase by the sum of valid word occurrences
    best_phrase = max(
        lemmatized_phrases.values(),
        key=lambda phrase: sum(valid_word_counts.get(word, 0) for word in phrase.split()),
        default=None
    )

    # Retrieve the original phrase before lemmatization
    if best_phrase:
        new_name = next(original for original, lemmatized in lemmatized_phrases.items() if lemmatized == best_phrase)

        # Filter out short and irrelevant words, prioritizing known words
        filtered_words = [
            word for word in new_name.split() 
            if len(word) >= 3 and ( word in known_words or len(word) >= 4)
        ]

        new_name = " ".join(filtered_words)

    else:
        new_name = max(phrases, key=lambda phrase: len(phrase.split()))

    return new_name


def update_and_rename_key(data, old_key, sublist_index, new_value, new_key=None):
    """Updates a specific sublist and optionally renames the key."""
    if old_key in data and 0 <= sublist_index < len(data[old_key]):
        # Update the specific sublist
        data[old_key][sublist_index].append(new_value)
        
        # Rename key if new_key is provided and different
        if new_key and new_key != old_key:
            data[new_key] = data.pop(old_key)

    else:
        print(f"Invalid key '{old_key}' or sublist index {sublist_index}")


def filter_best_match_keys(new_company, best_match_keys):
    new_company_stripped = new_company.replace(" ", "").lower()
    return [
        key for key in best_match_keys
        if abs(len(new_company_stripped) - len(key.replace(" ", "").lower())) < 4
    ]


def find_mismatched_words(list1, list2, known_words):
    """
    Finds words in list1 and list2 that are not common and may require similarity checks.
    Returns `not_valid1` and `not_valid2` while preserving the order of `list1` and `list2`.
    """    
    valid_list1 = [word for word in list1 if word in known_words]
    valid_list2 = [word for word in list2 if word in known_words]

    common_words = set(valid_list1) & set(valid_list2)

    not_valid1 = [word for word in list2 if word not in common_words]
    not_valid2 = [word for word in list1 if word not in common_words]

    return not_valid1, not_valid2


def custom_sort(word, key):
    return [word[i] if i < len(word) and i < len(key) and 
            word[i] == key[i] else "" for i in range(len(key))], word

def phonetic_similarity_check(word1, word2):
    length = min(len(word1), len(word2))
    
    if length <= 3:
        edtx_thresh, mra_thresh = 0.6, 0.6 
    elif length <= 4:
        edtx_thresh, mra_thresh = 0.65, 0.65
    elif length <= 8:
        edtx_thresh, mra_thresh = 0.6, 0.5
    else:
        edtx_thresh, mra_thresh = 0.5, 0.4
    
    edtx_sim = editex.sim(word1, word2) or 0.0
    mra_sim = mra.sim(word1, word2) or 0.0

    sim_ = (edtx_sim + mra_sim) / 2
    
    if sim_ > ((edtx_thresh+ mra_thresh )/2):
        return sim_
    return 0.0

def phonetic_scorer(word1, word2, score_cutoff=0):
    score = int(phonetic_similarity_check(word1, word2) * 100)
    return score if score >= score_cutoff else 0

def all_phonetic_matches(un_intersected_list1, un_intersected_list2):
    phonetic_matches = {}
    
    sorted_list1 = sorted(un_intersected_list1)
    sorted_list2 = sorted(un_intersected_list2)
    
    for word1 in sorted_list1:
        if sorted_list2:
            sorted_list2 = sorted(sorted_list2, key=partial(custom_sort, key=word1), reverse=True)
            
            best_match = process.extractOne(
                word1, sorted_list2, scorer=phonetic_scorer
            )
            
            if best_match and best_match[1] > 0:
                phonetic_matches[word1] = (best_match[0], best_match[1])
    
    return phonetic_matches

def analyze_lists_with_phonetics(list1, list2):
    set1, set2 = set(list1), set(list2)
    
    un_intersected_list1 = set1 - set2
    un_intersected_list2 = set2 - set1
    
    phonetic_matches = all_phonetic_matches(un_intersected_list1, un_intersected_list2)
    
    return {
        "un_intersected_list1": un_intersected_list1,
        "un_intersected_list2": un_intersected_list2,
        "phonetic_matches": phonetic_matches,
    }

def are_different_known_words(list1, list2):
    # Get intersection with known_words
    valid_list1 = set(list1) & known_words
    valid_list2 = set(list2) & known_words

    # If both are empty, consider them similar (no valid words to compare)
    if not valid_list1 or not valid_list2:
        return False

    # Compare the sets
    return not valid_list1 == valid_list2

def compare_company_names(new_company, company_groups):
    """Compare a new company name with existing company groups using lists of lists."""
    best_match_keys = find_best_matching_groups(new_company, company_groups)
    best_match_keys = filter_best_match_keys(new_company, best_match_keys)
    tokens = new_company.split()
    first_word = tokens[0]
    processed_tokens = [first_word] + [preprocess_company_name(word) for word in tokens[1:]]
    new_company_norm = " ".join(processed_tokens)

    accepted = False
    reject_reason = None
    
    if new_company == 'amazing r aa skin care':
        print("")

    if not best_match_keys:
        company_groups[get_name([new_company])] = [[new_company]]
        accepted = True
    else:
        for best_match_key in best_match_keys:
            if (not are_different_known_words(best_match_key.split()[1:],new_company.split()[1:])):
                sublists = company_groups[best_match_key]

                for sublist_index, sublist in enumerate(sublists):
                    if new_company in sublist:
                        return {"accepted": False, "reason": f"'{new_company}' already exists."}

                    fuzzy_matches = process.extract(new_company_norm, sublist, scorer=fuzz.ratio, limit=None)
                    # fuzzy_matches = [(match[0], match[1]) for match in fuzzy_matches] if fuzzy_matches else []
                    fuzzy_matches = [(match[0], match[1]) for match in fuzzy_matches if match[1] > 80] if fuzzy_matches else []


                    if len(fuzzy_matches) != len(sublist) or not fuzzy_matches:
                        continue 

                    all_conditions_met = True
                    
                    for existing_company, _ in fuzzy_matches:

                        if not existing_company.replace(" ", "")[:3].lower() == new_company_norm.replace(" ", "")[:3].lower():
                            reject_reason = f" {new_company_norm} Failed acronym check {existing_company}'"
                            all_conditions_met = False
                            break

                        existing_words = existing_company.split()
                        new_words = new_company_norm.split()
                        existing_words = [existing_words[0]] + [preprocess_company_name(word) for word in existing_words[1:]]

                        if fuzz.ratio(new_words[0], existing_words[0]) < 86:
                            all_conditions_met = False
                            reject_reason = f"'{new_words[0]}' failed first word similarity check with '{existing_words[0]}'"
                            break

                        not_valid1, not_valid2 = find_mismatched_words(new_words[1:], existing_words[1:], known_words)

                        result = analyze_lists_with_phonetics(not_valid1, not_valid2)

                        discrepancies = {
                            word: match
                            for word, (match, _) in result['phonetic_matches'].items()
                            if word in known_words and match in known_words and word != match
                        }

                        if discrepancies:
                            all_conditions_met = False
                            reject_reason = f"'{new_words[0]}' failed phonetic check"
                            break

                        
                        phonetic_sim = mra.sim(" ".join(not_valid1), " ".join(not_valid2)) 
                        token_ratio = fuzz.token_ratio(" ".join(not_valid1), " ".join(not_valid2))/100
                        
                        if phonetic_sim < 0.7:
                            all_conditions_met = False
                            reject_reason = f"'{new_words[0]}' failed first word similarity check with '{existing_words[0]}'"
                            break

                        if are_different_known_words(not_valid1,not_valid2):
                            all_conditions_met = False
                            reject_reason = f"'{not_valid1}' different words '{not_valid2}'"
                            break

                    
                    if all_conditions_met:
                        update_and_rename_key(company_groups, best_match_key, sublist_index, new_company)
                        new_key_name = get_name([company for sublist in sublists for company in sublist])
                        if new_key_name != best_match_key:
                            company_groups[new_key_name] = company_groups.pop(best_match_key)
                        return {"accepted": True, "reason": f"'{new_company}' added to sublist {sublist_index}, key updated to '{new_key_name}'."}

        # If none of the conditions were met, check for phonetic similarity with the best match key
        root_key_match = process.extractOne(new_company_norm, best_match_keys, scorer=fuzz.ratio)

        if root_key_match and root_key_match[1] > 88:
            best_match_key = root_key_match[0]  # Extracted best match key

            # Extract words excluding the first word
            best_match_words = best_match_key.split()[1:]  
            new_company_words = new_company.split()[1:]
            suffix1, suffix2 =  " ".join(best_match_words), " ".join(new_company_words)
            mra_sim = mra.sim(suffix1, suffix2)

            suffix_ratio = fuzz.ratio(suffix1,suffix2)


            result = analyze_lists_with_phonetics(best_match_words, new_company_words)

            discrepancies = {
                word: match
                for word, (match, _) in result['phonetic_matches'].items()
                if word in known_words and match in known_words and word != match
            }

            # Compare phonetic similarity word by word
            if suffix_ratio > 86 and mra_sim > 0.8 and not discrepancies:
                if len(best_match_words) ==  0 and len(new_company_words) == 0:
                    company_groups[best_match_key].append([new_company])
                    return {"accepted": True, "reason": f"'{new_company}' added as a new sublist under '{best_match_key}'."}
                else:
                    new_key = get_name([new_company])
                    company_groups[new_key].append([new_company])
                    return {"accepted": True, "reason": f"'{new_company}' added as a new sublist under '{best_match_key}'."}

        # If no suitable match is found, create a new key
        new_key = get_name([new_company])

        # If the generated key already exists, append the new company to it
        if new_key in company_groups:
            company_groups[new_key].append([new_company])
            return {"accepted": True, "reason": f"'{new_company}' added to existing key '{new_key}' as a new sublist."}
        else:
            company_groups[new_key] = [[new_company]]
            return {"accepted": False, "reason": f"'{new_company}' did not meet conditions for any existing sublist: {reject_reason}. Created a new key."}

    return {"accepted": accepted}



existing_companies = defaultdict(list)


# fix and fit are coming as synonyms

new_companies =  [
    'amazing ra skin care', 'amazing r aa skin care',
    "all team satffing", "all team staffing", "all teams staffing", 'all temps staffing', 'all time staffing',
    "handy fix", "handy fit",
    "home depot support center", "home depot suppor center",
    "allied univ security", "alied university", "allied university", "alieed unversity", "alied unversity", "alied university", "allied univ",
    "infosys contracts", "infosys constructions", "infysys bds construction", "infosys contrac",  "amazon flex", "amazons flex", "amazon",
    "amazon flix", "amazon xlx", "amazon fle", "amazon xlx", "amazon y", "amazon flix",
    "amazon lex", "amazon xlx", "amazon dls", "amazonn", "amazon", "amazpnp", "amazonj", "amazon tw", "amazon", "amzn",
    "amazoan l", "amazone a", "amazon flexz", "amazonf lez", "amzon r", "amazon rfd", "amazon ada", "amazon litaasad",
    "amazon wt", "amazon flrex", "amazon es", "amazon dava", "amazon c", "amazon y", "amazon flexes", "amazon llez", "amazon llex",
    "amazon bf", "amazon fulfilment center", "amazon ful fulmetn centr", "amzon full fillment xenter"
]


new_companies = sorted(new_companies, key=lambda item: item.lower())

for new_comp in new_companies:
    result = compare_company_names(new_comp, existing_companies)
    print(result)
    print('existing_companies', existing_companies)

