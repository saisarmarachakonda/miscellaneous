import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from rapidfuzz import fuzz, process
from abydos.distance import Editex, MRA
from spellchecker import SpellChecker
import itertools
from collections import defaultdict

spell = SpellChecker()
known_words = set(spell.known(spell.word_frequency.keys()))

specific_words = ['lex']

print('compact' in known_words)

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
editex = Editex()
mra = MRA()

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

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
        return False
    wordnet_presence =  word in known_words
    return wordnet_presence

def check_synonym_match(word1, word2):
    syn1 = {lemma.name() for synset in wordnet.synsets(word1) for lemma in synset.lemmas()}
    syn2 = {lemma.name() for synset in wordnet.synsets(word2) for lemma in synset.lemmas()}
    is_synonym = bool(syn1 & syn2)
    return is_synonym

def phonetic_similarity(word1, word2):
    length = min(len(word1), len(word2))
    
    # Adjust thresholds dynamically based on length
    if length <= 3:
        edtx_thresh, mra_thresh = 0.6, 0.6 
    elif length <= 4:
        edtx_thresh, mra_thresh = 0.7, 0.7  # Stricter for short words
    elif length <= 8:
        edtx_thresh, mra_thresh = 0.6, 0.5  # Moderate
    else:
        edtx_thresh, mra_thresh = 0.5, 0.4  # More lenient for longer words

    edtx_sim = editex.sim(word1, word2)
    mra_sim = mra.sim(word1, word2)
    
    return edtx_sim >= edtx_thresh and mra_sim >= mra_thresh
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

from spellchecker import SpellChecker
import nltk
from nltk.stem import WordNetLemmatizer
from collections import Counter

# nltk.download("wordnet")

# Initialize SpellChecker and Lemmatizer
# spell = SpellChecker()
lemmatizer = WordNetLemmatizer()

raw_known_words = spell.known(spell.word_frequency.keys())
known_words = {lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in raw_known_words}


def get_name(phrases):
    # Get known words and lemmatize them with correct POS
    # Lemmatize phrases
    lemmatized_phrases = {
        phrase: " ".join(lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in phrase.split()) for phrase in phrases
    }

    # Extract words from lemmatized phrases and filter only known words
    words = list(itertools.chain(*[phrase.split() for phrase in lemmatized_phrases.values()]))
    valid_word_counts = Counter(word for word in words if word in known_words)

    if not valid_word_counts:  # If no valid words are found, return the longest phrase
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

        # Remove words that are < 4 letters and not valid lemmas
        words = new_name.split()
        filtered_words = []
        if 'lex' in words:
            print()
            
        for word in words:
            lemmatized_word = lemmatizer.lemmatize(word, get_wordnet_pos(word))
            if len(word) > 2:
                if (len(word) >= 4 or lemmatized_word in known_words):
                    filtered_words.append(word)
                elif lemmatized_word in ['lex']:
                    filtered_words.append(word)

        new_name = " ".join(filtered_words)

    else:
        new_name = max(phrases, key=lambda phrase: len(phrase.split()))

    if new_name == 'amazon lex':
        print()
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

def compare_company_names(new_company, company_groups):
    """Compare a new company name with existing company groups using lists of lists."""
    best_match_keys = find_best_matching_groups(new_company, company_groups)
    best_match_keys = filter_best_match_keys(new_company, best_match_keys)

    accepted = False
    reject_reason = None
    
    if new_company == 'amazon fle':
        print("")
    if not best_match_keys:
        company_groups[new_company] = [[new_company]]
        accepted = True
    else:
        for best_match_key in best_match_keys:
            sublists = company_groups[best_match_key]

            for sublist_index, sublist in enumerate(sublists):
                tokens = new_company.split()
                first_word = tokens[0]
                processed_tokens = [first_word] + [preprocess_company_name(word) for word in tokens[1:]]
                new_company_norm = " ".join(processed_tokens)

                if new_company in sublist:
                    return {"accepted": False, "reason": f"'{new_company}' already exists."}

                fuzzy_matches = process.extract(new_company_norm, sublist, scorer=fuzz.ratio, limit=None)
                fuzzy_matches = [(match[0], match[1]) for match in fuzzy_matches] if fuzzy_matches else []
                # fuzzy_matches = [(match[0], match[1]) for match in fuzzy_matches if match[1] > 80] if fuzzy_matches else []


                if len(fuzzy_matches) != len(sublist) or not fuzzy_matches:
                    continue 

                all_conditions_met = True
                
                for existing_company, _ in fuzzy_matches:
                    existing_words = existing_company.split()
                    new_words = new_company_norm.split()

                    if fuzz.ratio(new_words[0], existing_words[0]) < 86:
                        all_conditions_met = False
                        reject_reason = f"'{new_words[0]}' failed first word similarity check with '{existing_words[0]}'"
                        break

                    for word in new_words[1:]:
                        if len(word) < 3:
                            continue  # Skip words with less than 2 letters

                        if check_wordnet_presence(word):
                            valid_existing_words = [
                                ex_word for ex_word in existing_words[1:]
                                if check_wordnet_presence(ex_word)
                            ]
                            if valid_existing_words and not any(check_synonym_match(word, ex_word) for ex_word in valid_existing_words):
                                all_conditions_met = False
                                reject_reason = f"'{word}' failed synonym check."
                                break
                            elif not any(phonetic_similarity(word, ex_word) for ex_word in existing_words[1:]):
                                all_conditions_met = False
                                reject_reason = f"'{word}' failed phonetic similarity check."
                                break
                        else:
                            if not any(phonetic_similarity(word, ex_word) for ex_word in existing_words[1:]):
                                all_conditions_met = False
                                reject_reason = f"'{word}' failed phonetic similarity check."
                                break
                    if not all_conditions_met:
                        break
                
                if all_conditions_met:
                    update_and_rename_key(company_groups, best_match_key, sublist_index, new_company)
                    new_key_name = get_name([company for sublist in sublists for company in sublist])
                    if new_key_name != best_match_key:
                        company_groups[new_key_name] = company_groups.pop(best_match_key)
                    return {"accepted": True, "reason": f"'{new_company}' added to sublist {sublist_index}, key updated to '{new_key_name}'."}

        # If none of the conditions were met, check for phonetic similarity with the best match key
        check2 = process.extractOne(new_company_norm, best_match_keys, scorer=fuzz.ratio)

        if check2 and check2[1] > 88:
            best_match_key = check2[0]  # Extracted best match key

            # Extract words excluding the first word
            best_match_words = best_match_key.split()[1:]  
            new_company_words = new_company.split()[1:]

            suffix_ratio = fuzz.ratio(" ".join(best_match_words), " ".join(new_company_words))

            # Compare phonetic similarity word by word
            if suffix_ratio > 86 and any(phonetic_similarity(word, ex_word) for word in new_company_words for ex_word in best_match_words):
                company_groups[best_match_key].append([new_company])
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
