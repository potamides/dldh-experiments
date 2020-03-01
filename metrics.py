from g2p_en import G2p
from Levenshtein import ratio
from nltk import download
from nltk.corpus import stopwords
from string import ascii_lowercase
from textstat import flesch_reading_ease
import re
import numpy as np

download('stopwords', quiet=True)
# sound an alliteration can start with
sounds = ["ch", "ph", "sh", "th"] + [letter for letter in ascii_lowercase]

def get_rhyme_scores(poems: str) -> float:
    """
    assumptions:
    * split rhymes don't count
    * rhyme scheme shouldn't be evaluated
    * a rhyme begins at the first stressed vocal
    * rhymes are "tuples"
    * only end rhymes count
    """
    scores = list()
    rhymes = list()
    regex = re.compile(r'[^a-zA-Z\s]')
    g2p = G2p()

    for poem in poems:
        poem = regex.sub("", poem)
        phonemes = list()
        concat_phonemes = list()
        scores.append(0)
        rhymes.append(0)

        for line in poem.split('\n'):
            if line.split():
                last_word = line.split()[-1]
                phonemes.append((last_word, g2p(last_word)))
            else:
                rhymes[-1] +=  1
        for word, phoneme_list in phonemes:
            for idx, phoneme in enumerate(phoneme_list):
                if '1' in phoneme:  # stressed vocal
                    concat_phonemes.append((word, "".join(phoneme_list[idx:])))
                    break
            # if the word doesn't have a stressed vocal (only happens rarely)
            else:
                concat_phonemes.append((word, "".join(phoneme_list)))
        while concat_phonemes:
            candidate = concat_phonemes.pop(0)
            if concat_phonemes:
                match = max([(word, ratio(word[1], candidate[1])) for word in
                             concat_phonemes], key=lambda x: x[1])
                concat_phonemes.remove(match[0])
                scores[-1] += match[1]
            rhymes[-1] += 1

    return np.expand_dims(np.asarray(scores) / np.asarray(rhymes), axis=1)

def get_alliteration_scores(poems: str) -> float:
    """
    assumptions:
    * repetition is no alliteration
    * stopwords may occur between alliterations
    * alliterations are only computed per line
    """
    scores = list()
    num_lines = list()
    for poem in poems:
        scores.append(0)
        num_lines.append(0)
        for line in poem.split('\n'):
            last_sound = '--'
            num_lines[-1] += 1
            seen_words = list()
            for word in line.split(' '):
                if word in stopwords.words("english"):
                    continue
                if word.lower().startswith(last_sound) and word not in seen_words:
                    scores[-1] += 1
                    seen_words.append(word.lower())
                    continue
                else:
                    last_sound = '--'
                for sound in sounds:
                    if word.lower().startswith(sound):
                        last_sound = sound
                        del seen_words[:]
                        break

    return np.expand_dims(np.asarray(scores) / np.asarray(num_lines), axis=1)

def get_readability_scores(poems: str) -> float:
    scores = list()
    for poem in poems:
        scores.append(flesch_reading_ease(poem))

    return np.expand_dims(np.asarray(scores), axis=1)
