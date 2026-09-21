# coding: utf-8
"""
BLEU and chrF, extracted from sacreBLEU v1.4.x.

Only the corpus-level BLEU and chrF paths used by :mod:`signspark_bt.metrics`
are kept. The implementations are unchanged, so scores remain comparable with
the sign language translation literature, which reports sacreBLEU numbers.

Original: https://github.com/mjpost/sacrebleu (Apache-2.0), by Matt Post.
"""

import functools
import logging
import math
import re
import sys
import unicodedata
from collections import Counter, namedtuple
from itertools import zip_longest
from typing import List, Iterable, Tuple, Union


NGRAM_ORDER = 4

CHRF_ORDER = 6

CHRF_BETA = 2

SMOOTH_VALUE_DEFAULT = 0.0

def tokenize_13a(line):
    """
    Tokenizes an input line using a relatively minimal tokenization that is however equivalent to mteval-v13a, used by WMT.

    :param line: a segment to tokenize
    :return: the tokenized line
    """

    norm = line

    # language-independent part:
    norm = norm.replace("<skipped>", "")
    norm = norm.replace("-\n", "")
    norm = norm.replace("\n", " ")
    norm = norm.replace("&quot;", '"')
    norm = norm.replace("&amp;", "&")
    norm = norm.replace("&lt;", "<")
    norm = norm.replace("&gt;", ">")

    # language-dependent part (assuming Western languages):
    norm = " {} ".format(norm)
    norm = re.sub(r"([\{-\~\[-\` -\&\(-\+\:-\@\/])", " \\1 ", norm)
    norm = re.sub(
        r"([^0-9])([\.,])", "\\1 \\2 ", norm
    )  # tokenize period and comma unless preceded by a digit
    norm = re.sub(
        r"([\.,])([^0-9])", " \\1 \\2", norm
    )  # tokenize period and comma unless followed by a digit
    norm = re.sub(
        r"([0-9])(-)", "\\1 \\2 ", norm
    )  # tokenize dash when preceded by a digit
    norm = re.sub(r"\s+", " ", norm)  # one space only between words
    norm = re.sub(r"^\s+", "", norm)  # no leading space
    norm = re.sub(r"\s+$", "", norm)  # no trailing space

    return norm

class UnicodeRegex:
    """Ad-hoc hack to recognize all punctuation and symbols.

    without depending on https://pypi.python.org/pypi/regex/."""

    @staticmethod
    def _property_chars(prefix):
        return "".join(
            chr(x)
            for x in range(sys.maxunicode)
            if unicodedata.category(chr(x)).startswith(prefix)
        )

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def punctuation():
        return UnicodeRegex._property_chars("P")

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def nondigit_punct_re():
        return re.compile(r"([^\d])([" + UnicodeRegex.punctuation() + r"])")

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def punct_nondigit_re():
        return re.compile(r"([" + UnicodeRegex.punctuation() + r"])([^\d])")

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def symbol_re():
        return re.compile("([" + UnicodeRegex._property_chars("S") + "])")

def tokenize_v14_international(string):
    r"""Tokenize a string following the official BLEU implementation.

    See https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/mteval-v14.pl#L954-L983
    In our case, the input string is expected to be just one line
    and no HTML entities de-escaping is needed.
    So we just tokenize on punctuation and symbols,
    except when a punctuation is preceded and followed by a digit
    (e.g. a comma/dot as a thousand/decimal separator).

    Note that a number (e.g., a year) followed by a dot at the end of sentence is NOT tokenized,
    i.e. the dot stays with the number because `s/(\p{P})(\P{N})/ $1 $2/g`
    does not match this case (unless we add a space after each sentence).
    However, this error is already in the original mteval-v14.pl
    and we want to be consistent with it.
    The error is not present in the non-international version,
    which uses `$norm_text = " $norm_text "` (or `norm = " {} ".format(norm)` in Python).

    :param string: the input string
    :return: a list of tokens
    """
    string = UnicodeRegex.nondigit_punct_re().sub(r"\1 \2 ", string)
    string = UnicodeRegex.punct_nondigit_re().sub(r" \1 \2", string)
    string = UnicodeRegex.symbol_re().sub(r" \1 ", string)
    return string.strip()

def tokenize_zh(sentence):
    """MIT License
    Copyright (c) 2017 - Shujian Huang <huangsj@nju.edu.cn>

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    The tokenization of Chinese text in this script contains two steps: separate each Chinese
    characters (by utf-8 encoding); tokenize the non Chinese part (following the mteval script).
    Author: Shujian Huang huangsj@nju.edu.cn

    :param sentence: input sentence
    :return: tokenized sentence
    """

    def is_chinese_char(uchar):
        """
        :param uchar: input char in unicode
        :return: whether the input char is a Chinese character.
        """
        if (
            uchar >= u"\u3400" and uchar <= u"\u4db5"
        ):  # CJK Unified Ideographs Extension A, release 3.0
            return True
        elif (
            uchar >= u"\u4e00" and uchar <= u"\u9fa5"
        ):  # CJK Unified Ideographs, release 1.1
            return True
        elif (
            uchar >= u"\u9fa6" and uchar <= u"\u9fbb"
        ):  # CJK Unified Ideographs, release 4.1
            return True
        elif (
            uchar >= u"\uf900" and uchar <= u"\ufa2d"
        ):  # CJK Compatibility Ideographs, release 1.1
            return True
        elif (
            uchar >= u"\ufa30" and uchar <= u"\ufa6a"
        ):  # CJK Compatibility Ideographs, release 3.2
            return True
        elif (
            uchar >= u"\ufa70" and uchar <= u"\ufad9"
        ):  # CJK Compatibility Ideographs, release 4.1
            return True
        elif (
            uchar >= u"\u20000" and uchar <= u"\u2a6d6"
        ):  # CJK Unified Ideographs Extension B, release 3.1
            return True
        elif (
            uchar >= u"\u2f800" and uchar <= u"\u2fa1d"
        ):  # CJK Compatibility Supplement, release 3.1
            return True
        elif (
            uchar >= u"\uff00" and uchar <= u"\uffef"
        ):  # Full width ASCII, full width of English punctuation, half width Katakana, half wide half width kana, Korean alphabet
            return True
        elif uchar >= u"\u2e80" and uchar <= u"\u2eff":  # CJK Radicals Supplement
            return True
        elif uchar >= u"\u3000" and uchar <= u"\u303f":  # CJK punctuation mark
            return True
        elif uchar >= u"\u31c0" and uchar <= u"\u31ef":  # CJK stroke
            return True
        elif uchar >= u"\u2f00" and uchar <= u"\u2fdf":  # Kangxi Radicals
            return True
        elif uchar >= u"\u2ff0" and uchar <= u"\u2fff":  # Chinese character structure
            return True
        elif uchar >= u"\u3100" and uchar <= u"\u312f":  # Phonetic symbols
            return True
        elif (
            uchar >= u"\u31a0" and uchar <= u"\u31bf"
        ):  # Phonetic symbols (Taiwanese and Hakka expansion)
            return True
        elif uchar >= u"\ufe10" and uchar <= u"\ufe1f":
            return True
        elif uchar >= u"\ufe30" and uchar <= u"\ufe4f":
            return True
        elif uchar >= u"\u2600" and uchar <= u"\u26ff":
            return True
        elif uchar >= u"\u2700" and uchar <= u"\u27bf":
            return True
        elif uchar >= u"\u3200" and uchar <= u"\u32ff":
            return True
        elif uchar >= u"\u3300" and uchar <= u"\u33ff":
            return True

        return False

    sentence = sentence.strip()
    sentence_in_chars = ""
    for char in sentence:
        if is_chinese_char(char):
            sentence_in_chars += " "
            sentence_in_chars += char
            sentence_in_chars += " "
        else:
            sentence_in_chars += char
    sentence = sentence_in_chars

    # TODO: the code above could probably be replaced with the following line:
    # import regex
    # sentence = regex.sub(r'(\p{Han})', r' \1 ', sentence)

    # tokenize punctuation
    sentence = re.sub(r"([\{-\~\[-\` -\&\(-\+\:-\@\/])", r" \1 ", sentence)

    # tokenize period and comma unless preceded by a digit
    sentence = re.sub(r"([^0-9])([\.,])", r"\1 \2 ", sentence)

    # tokenize period and comma unless followed by a digit
    sentence = re.sub(r"([\.,])([^0-9])", r" \1 \2", sentence)

    # tokenize dash when preceded by a digit
    sentence = re.sub(r"([0-9])(-)", r"\1 \2 ", sentence)

    # one space only between words
    sentence = re.sub(r"\s+", r" ", sentence)

    # no leading or trailing spaces
    sentence = sentence.strip()

    return sentence

TOKENIZERS = {
    "13a": tokenize_13a,
    "intl": tokenize_v14_international,
    "zh": tokenize_zh,
    "none": lambda x: x,
}

DEFAULT_TOKENIZER = "13a"

def my_log(num):
    """
    Floors the log function

    :param num: the number
    :return: log(num) floored to a very low number
    """

    if num == 0.0:
        return -9999999999
    return math.log(num)

def extract_ngrams(line, min_order=1, max_order=NGRAM_ORDER) -> Counter:
    """Extracts all the ngrams (min_order <= n <= max_order) from a sequence of tokens.

    :param line: A segment containing a sequence of words.
    :param min_order: Minimum n-gram length (default: 1).
    :param max_order: Maximum n-gram length (default: NGRAM_ORDER).
    :return: a dictionary containing ngrams and counts
    """

    ngrams = Counter()
    tokens = line.split()
    for n in range(min_order, max_order + 1):
        for i in range(0, len(tokens) - n + 1):
            ngram = " ".join(tokens[i : i + n])
            ngrams[ngram] += 1

    return ngrams

def extract_char_ngrams(s: str, n: int) -> Counter:
    """
    Yields counts of character n-grams from string s of order n.
    """
    return Counter([s[i : i + n] for i in range(len(s) - n + 1)])

def ref_stats(output, refs):
    ngrams = Counter()
    closest_diff = None
    closest_len = None
    for ref in refs:
        tokens = ref.split()
        reflen = len(tokens)
        diff = abs(len(output.split()) - reflen)
        if closest_diff is None or diff < closest_diff:
            closest_diff = diff
            closest_len = reflen
        elif diff == closest_diff:
            if reflen < closest_len:
                closest_len = reflen

        ngrams_ref = extract_ngrams(ref)
        for ngram in ngrams_ref.keys():
            ngrams[ngram] = max(ngrams[ngram], ngrams_ref[ngram])

    return ngrams, closest_diff, closest_len

class Result:
    def __init__(self, score: float):
        self.score = score

    def __str__(self):
        return self.format()

class BLEU:
    def __init__(self, scores, counts, totals, precisions, bp, sys_len, ref_len):

        self.scores = scores
        self.counts = counts
        self.totals = totals
        self.precisions = precisions
        self.bp = bp
        self.sys_len = sys_len
        self.ref_len = ref_len

    def format(self, width=2):
        precisions = "/".join(["{:.1f}".format(p) for p in self.precisions])
        return "BLEU = {scores} {precisions} (BP = {bp:.3f} ratio = {ratio:.3f} hyp_len = {sys_len:d} ref_len = {ref_len:d})".format(
            scores=self.scores,
            width=width,
            precisions=precisions,
            bp=self.bp,
            ratio=self.sys_len / self.ref_len,
            sys_len=self.sys_len,
            ref_len=self.ref_len,
        )

class CHRF(Result):
    def __init__(self, score: float):
        super().__init__(score)

    def format(self, width=2):
        return "{score:.{width}f}".format(score=self.score, width=width)

def compute_bleu(
    correct: List[int],
    total: List[int],
    sys_len: int,
    ref_len: int,
    smooth_method="none",
    smooth_value=SMOOTH_VALUE_DEFAULT,
    use_effective_order=False,
) -> BLEU:
    """Computes BLEU score from its sufficient statistics. Adds smoothing.

    Smoothing methods (citing "A Systematic Comparison of Smoothing Techniques for Sentence-Level BLEU",
    Boxing Chen and Colin Cherry, WMT 2014: http://aclweb.org/anthology/W14-3346)

    - exp: NIST smoothing method (Method 3)
    - floor: Method 1
    - add-k: Method 2 (generalizing Lin and Och, 2004)
    - none: do nothing.

    :param correct: List of counts of correct ngrams, 1 <= n <= NGRAM_ORDER
    :param total: List of counts of total ngrams, 1 <= n <= NGRAM_ORDER
    :param sys_len: The cumulative system length
    :param ref_len: The cumulative reference length
    :param smooth: The smoothing method to use
    :param smooth_value: The smoothing value added, if smooth method 'floor' is used
    :param use_effective_order: If true, use the length of `correct` for the n-gram order instead of NGRAM_ORDER.
    :return: A BLEU object with the score (100-based) and other statistics.
    """

    precisions = [0 for x in range(NGRAM_ORDER)]

    smooth_mteval = 1.0
    effective_order = NGRAM_ORDER
    for n in range(NGRAM_ORDER):
        if smooth_method == "add-k" and n > 1:
            correct[n] += smooth_value
            total[n] += smooth_value
        if total[n] == 0:
            break

        if use_effective_order:
            effective_order = n + 1

        if correct[n] == 0:
            if smooth_method == "exp":
                smooth_mteval *= 2
                precisions[n] = 100.0 / (smooth_mteval * total[n])
            elif smooth_method == "floor":
                precisions[n] = 100.0 * smooth_value / total[n]
        else:
            precisions[n] = 100.0 * correct[n] / total[n]

    # If the system guesses no i-grams, 1 <= i <= NGRAM_ORDER, the BLEU score is 0 (technically undefined).
    # This is a problem for sentence-level BLEU or a corpus of short sentences, where systems will get no credit
    # if sentence lengths fall under the NGRAM_ORDER threshold. This fix scales NGRAM_ORDER to the observed
    # maximum order. It is only available through the API and off by default

    brevity_penalty = 1.0
    if sys_len < ref_len:
        brevity_penalty = math.exp(1 - ref_len / sys_len) if sys_len > 0 else 0.0

    scores = []
    for effective_order in range(1, NGRAM_ORDER + 1):
        scores.append(
            brevity_penalty
            * math.exp(sum(map(my_log, precisions[:effective_order])) / effective_order)
        )

    return BLEU(scores, correct, total, precisions, brevity_penalty, sys_len, ref_len)

def corpus_bleu(
    sys_stream: Union[str, Iterable[str]],
    ref_streams: Union[str, List[Iterable[str]]],
    smooth_method="exp",
    smooth_value=SMOOTH_VALUE_DEFAULT,
    force=False,
    lowercase=False,
    tokenize=DEFAULT_TOKENIZER,
    use_effective_order=False,
) -> BLEU:
    """Produces BLEU scores along with its sufficient statistics from a source against one or more references.

    :param sys_stream: The system stream (a sequence of segments)
    :param ref_streams: A list of one or more reference streams (each a sequence of segments)
    :param smooth: The smoothing method to use
    :param smooth_value: For 'floor' smoothing, the floor to use
    :param force: Ignore data that looks already tokenized
    :param lowercase: Lowercase the data
    :param tokenize: The tokenizer to use
    :return: a BLEU object containing everything you'd want
    """

    # Add some robustness to the input arguments
    if isinstance(sys_stream, str):
        sys_stream = [sys_stream]
    if isinstance(ref_streams, str):
        ref_streams = [[ref_streams]]

    sys_len = 0
    ref_len = 0

    correct = [0 for n in range(NGRAM_ORDER)]
    total = [0 for n in range(NGRAM_ORDER)]

    # look for already-tokenized sentences
    tokenized_count = 0

    fhs = [sys_stream] + ref_streams
    for lines in zip_longest(*fhs):
        if None in lines:
            raise EOFError("Source and reference streams have different lengths!")

        if lowercase:
            lines = [x.lower() for x in lines]

        if not (force or tokenize == "none") and lines[0].rstrip().endswith(" ."):
            tokenized_count += 1

            if tokenized_count == 100:
                logging.warning("That's 100 lines that end in a tokenized period ('.')")
                logging.warning(
                    "It looks like you forgot to detokenize your test data, which may hurt your score."
                )
                logging.warning(
                    "If you insist your data is detokenized, or don't care, you can suppress this message with '--force'."
                )

        output, *refs = [TOKENIZERS[tokenize](x.rstrip()) for x in lines]

        ref_ngrams, closest_diff, closest_len = ref_stats(output, refs)

        sys_len += len(output.split())
        ref_len += closest_len

        sys_ngrams = extract_ngrams(output)
        for ngram in sys_ngrams.keys():
            n = len(ngram.split())
            correct[n - 1] += min(sys_ngrams[ngram], ref_ngrams.get(ngram, 0))
            total[n - 1] += sys_ngrams[ngram]

    return compute_bleu(
        correct,
        total,
        sys_len,
        ref_len,
        smooth_method=smooth_method,
        smooth_value=smooth_value,
        use_effective_order=use_effective_order,
    )

def raw_corpus_bleu(sys_stream, ref_streams, smooth_value=SMOOTH_VALUE_DEFAULT) -> BLEU:
    """Convenience function that wraps corpus_bleu().
    This is convenient if you're using sacrebleu as a library, say for scoring on dev.
    It uses no tokenization and 'floor' smoothing, with the floor default to 0 (no smoothing).

    :param sys_stream: the system stream (a sequence of segments)
    :param ref_streams: a list of one or more reference streams (each a sequence of segments)
    """
    return corpus_bleu(
        sys_stream,
        ref_streams,
        smooth_method="floor",
        smooth_value=smooth_value,
        force=True,
        tokenize="none",
        use_effective_order=True,
    )

def delete_whitespace(text: str) -> str:
    """
    Removes whitespaces from text.
    """
    return re.sub(r"\s+", "", text).strip()

def get_sentence_statistics(
    hypothesis: str,
    reference: str,
    order: int = CHRF_ORDER,
    remove_whitespace: bool = True,
) -> List[float]:
    hypothesis = delete_whitespace(hypothesis) if remove_whitespace else hypothesis
    reference = delete_whitespace(reference) if remove_whitespace else reference
    statistics = [0] * (order * 3)
    for i in range(order):
        n = i + 1
        hypothesis_ngrams = extract_char_ngrams(hypothesis, n)
        reference_ngrams = extract_char_ngrams(reference, n)
        common_ngrams = hypothesis_ngrams & reference_ngrams
        statistics[3 * i + 0] = sum(hypothesis_ngrams.values())
        statistics[3 * i + 1] = sum(reference_ngrams.values())
        statistics[3 * i + 2] = sum(common_ngrams.values())
    return statistics

def get_corpus_statistics(
    hypotheses: Iterable[str],
    references: Iterable[str],
    order: int = CHRF_ORDER,
    remove_whitespace: bool = True,
) -> List[float]:
    corpus_statistics = [0] * (order * 3)
    for hypothesis, reference in zip(hypotheses, references):
        statistics = get_sentence_statistics(
            hypothesis, reference, order=order, remove_whitespace=remove_whitespace
        )
        for i in range(len(statistics)):
            corpus_statistics[i] += statistics[i]
    return corpus_statistics

def _avg_precision_and_recall(
    statistics: List[float], order: int
) -> Tuple[float, float]:
    avg_precision = 0.0
    avg_recall = 0.0
    effective_order = 0
    for i in range(order):
        hypotheses_ngrams = statistics[3 * i + 0]
        references_ngrams = statistics[3 * i + 1]
        common_ngrams = statistics[3 * i + 2]
        if hypotheses_ngrams > 0 and references_ngrams > 0:
            avg_precision += common_ngrams / hypotheses_ngrams
            avg_recall += common_ngrams / references_ngrams
            effective_order += 1
    if effective_order == 0:
        return 0.0, 0.0
    avg_precision /= effective_order
    avg_recall /= effective_order
    return avg_precision, avg_recall

def _chrf(avg_precision, avg_recall, beta: int = CHRF_BETA) -> float:
    if avg_precision + avg_recall == 0:
        return 0.0
    beta_square = beta ** 2
    score = (
        (1 + beta_square)
        * (avg_precision * avg_recall)
        / ((beta_square * avg_precision) + avg_recall)
    )
    return score

def corpus_chrf(
    hypotheses: Iterable[str],
    references: Iterable[str],
    order: int = CHRF_ORDER,
    beta: float = CHRF_BETA,
    remove_whitespace: bool = True,
) -> CHRF:
    """
    Computes Chrf on a corpus.

    :param hypotheses: Stream of hypotheses.
    :param references: Stream of references
    :param order: Maximum n-gram order.
    :param remove_whitespace: Whether to delete all whitespace from hypothesis and reference strings.
    :param beta: Defines importance of recall w.r.t precision. If beta=1, same importance.
    :return: Chrf score.
    """
    corpus_statistics = get_corpus_statistics(
        hypotheses, references, order=order, remove_whitespace=remove_whitespace
    )
    avg_precision, avg_recall = _avg_precision_and_recall(corpus_statistics, order)
    return CHRF(_chrf(avg_precision, avg_recall, beta=beta))
