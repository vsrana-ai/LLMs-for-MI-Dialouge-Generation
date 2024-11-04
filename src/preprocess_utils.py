import os
import re
import json
import nltk
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
import tqdm
from string import punctuation

from gensim.parsing.preprocessing import STOPWORDS
gensim_stop_words = STOPWORDS.difference(set(['not']))


current_file = os.path.dirname(os.path.realpath(__file__))


def remove_spaces_and_new_lines(text):
    formatted_text = (
        text.replace('\\n', ' ')
            .replace('\n', ' ')
            .replace('\t', ' ')
            .replace('\\', ' ')
            .replace('. com', '.com')
    )
    pattern = re.compile(r'\s+')
    without_whitespace = re.sub(pattern, ' ', formatted_text)
    formatted_text = without_whitespace.replace('?', ' ? ').replace(')', ') ')
    return formatted_text


def replace_number(text):
    text = re.sub(r" [0-9]+ ", " NN ", text)
    text = re.sub(r'([a-z]*)[0-9]+([a-z]+)|([a-z]+)[0-9]+([a-z]*)', r'\1\2', text)
    return text


def annotate_links(text):
    link_pattern = re.sub(r'http\S+', 'LINK', text)
    annotated = re.sub(r"\ [A-Za-z]*\.com", "LINK", link_pattern)
    return annotated


def remove_special_chars(text):
    text = re.sub(r"[^A-Za-z ]+", '', text)
    return text.translate(str.maketrans('', '', punctuation))


def remove_stopwords(text):
    word_list = [w for w in word_tokenize(text) if w.lower() not in gensim_stop_words]
    word_combined = ' '.join(word_list)
    return word_combined


def expand_contractions(text, contraction_mapping):
    token_list = text.lower().split(' ')
    for word in token_list:
        if word in contraction_mapping:
            token_list = [item.replace(word, contraction_mapping[word]) for item in token_list]
    combined = ' '.join(str(e) for e in token_list)
    return combined


def preprocess_text(text_list,
                    use_stopwords_removal=False,
                    contraction_map_filename="contraction_map.json"):
    with open(os.path.join(current_file, contraction_map_filename), 'r') as contraction_map_file:
        contraction_map = json.load(contraction_map_file)

    cleaned_text = []
    for text in tqdm.tqdm(text_list, desc="Preprocessing text"):
        if isinstance(text, str):
            text = text.lower()
            text = expand_contractions(text, contraction_map)
            text = replace_number(text)
            # text = remove_emoji(text)
            # text = annotate_links(text)
            text = remove_special_chars(text)
            text = remove_spaces_and_new_lines(text)
            if use_stopwords_removal:
                text = remove_stopwords(text)
        cleaned_text.append(text)
    return cleaned_text
