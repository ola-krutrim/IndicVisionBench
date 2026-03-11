# Reference: https://github.com/nttcslab-nlp/RIBES/blob/master/RIBES.py

# Tsutomu Hirao, Hideki Isozaki, Katsuhito Sudoh, Kevin Duh, Hajime Tsukada, and Masaaki Nagata,
# "Evaluating Translation Quality with Word Order Correlations,"
# Journal of Natural Language Processing, Vol. 21, No. 3, pp. 421-444, June, 2014 (in Japanese).

# Hideki Isozaki, Tsutomu Hirao, Kevin Duh, Katsuhito Sudoh, and Hajime Tsukada,
# "Automatic Evaluation of Translation Quality for Distant Language Pairs,"
# Proceedings of the 2010 Conference on Empirical Methods in Natural Language Processing (EMNLP),
# pp. 944--952 Cambridge MA, October, 2010
# -- http://aclweb.org/anthology-new/D/D10/D10-1092.pdf


import sys
if type(sys.version_info) is not tuple and sys.version_info.major != 3:
    reload(sys)
    sys.setdefaultencoding("utf-8")
import re
from math import exp

_RIBES_VERSION = '1.03'
debug = 0


### "overlapping" substring counts ( string.count(x) returns "non-overlapping" counts... )
def overlapping_count (pattern, string):
    pos = string.find(pattern)
    if pos > -1:
        return 1 + overlapping_count (pattern, string[pos+1:])
    else:
        return 0

### calculate Kendall's tau
def kendall(ref, hyp, emptyref=False):
    """Calculates Kendall's tau between a reference and a hypothesis

    Calculates Kendall's tau (also unigram precision and brevity penalty (BP))
    between a reference word list and a system output (hypothesis) word list.

    Arguments:
        ref : list of reference words
        sub : list of system output (hypothesis) words
        (optional) emptyref : allow empty reference translations (ignored in the evaluation)

    Returns:
        A tuple (nkt, precision, bp)
            - nkt       : normalized Kendall's tau
            - precision : unigram precision
            - bp        : brevity penalty

    Raises:
        RuntimeError: reference has no words, possibly due to a format violation
    """

    # check reference length, raise RuntimeError if no words are found.
    if len(ref) == 0:
        if emptyref == True:
            return (None, None, None)
        else:
            raise RuntimeError ("Reference has no words")
    # check hypothesis length, return "zeros" if no words are found
    elif len(hyp) == 0:
        if debug > 1: print ("nkt=%g, precision=%g, bp=%g" % (0.0, 0.0, 0.0), file=sys.stderr)
        return (0.0, 0.0, 0.0)
    # bypass -- return 1.0 for identical hypothesis
    #elif ref == hyp:
    #    if debug > 1: print ("nkt=%g, precision=%g, bp=%g" % (nkt, precision, bp), file=sys.stderr)
    #    return (1.0, 1.0, 1.0)

    # calculate brevity penalty (BP), not exceeding 1.0
    bp = min(1.0, exp(1.0 - 1.0 * len(ref)/len(hyp)))

    
    ### determine which ref. word corresponds to each hypothesis word
    # list for ref. word indices
    intlist = []


    ### prepare helper pseudo-string representing ref. and hyp. word sequences as strings,
    ### by mapping each word into non-overlapping Unicode characters
    # Word ID (dictionary)
    worddict = {}
    # Unicode hexadecimal sequences for ref. and words
    _ref = ""
    _hyp = ""
    for w in ref:
        # if w is not found in dictironary "worddict", add it.
        if w not in worddict:
            worddict[w] = len(worddict)
        # append Unicode hexadecimal for word w (with offset of 0x4e00 -- CJK character range)
        _ref += str(hex(worddict[w] + 0x4e00)).replace('0x', '', 1)
    # decode Unicode (UTF-16 BigEndian) sequences to UTF-8
    if type(sys.version_info) is not tuple and sys.version_info.major == 3:
        if sys.version_info.minor > 1:
            mapped_ref = bytes.fromhex(_ref).decode(encoding="utf_16_be")
        else:
            mapped_ref = bytes.fromhex(_ref).decode("utf_16_be")
    else:
        mapped_ref = _ref.decode("hex").decode("utf_16_be")

    for w in hyp:
        # if w is not found in dictironary "worddict", add it.
        if w not in worddict:
            worddict[w] = len(worddict)
        # append Unicode hexadecimal for word w (with offset of 0x4e00 -- CJK character range)
        _hyp += str(hex(worddict[w] + 0x4e00)).replace('0x', '', 1)
    # decode Unicode (UTF-16 BigEndian) sequences to UTF-8
    if type(sys.version_info) is not tuple and sys.version_info.major == 3:
        if sys.version_info.minor > 1:
            mapped_hyp = bytes.fromhex(_hyp).decode(encoding="utf_16_be")
        else:
            mapped_hyp = bytes.fromhex(_hyp).decode("utf_16_be")
    else:
        mapped_hyp = _hyp.decode("hex").decode("utf_16_be")

    for i in range(len(hyp)):
        ### i-th hypthesis word hyp[i]
        if not hyp[i] in ref: 
            ### hyp[i] doesn't exist in reference
            pass
            # go on to the next hyp. word
        elif ref.count(hyp[i]) == 1 and hyp.count(hyp[i]) == 1:
            ### if we can determine one-to-one word correspondence by only unigram
            ### one-to-one correspondence
            # append the index in reference
            intlist.append(ref.index(hyp[i]))
            # go on to the next hyp. word
        else:
            ### if not, we consider context words...
            # use Unicode-mapped string for efficiency
            for window in range (1, max(i+1, len(hyp)-i+1)):
                if window <= i:
                    ngram = mapped_hyp[i-window:i+1]
                    if overlapping_count(ngram, mapped_ref) == 1 and overlapping_count(ngram, mapped_hyp) == 1:
                        intlist.append(mapped_ref.index(ngram) + len(ngram) -1)
                        break
                if i+window < len(hyp):
                    ngram = mapped_hyp[i:i+window+1]
                    if overlapping_count(ngram, mapped_ref) == 1 and overlapping_count(ngram, mapped_hyp) == 1:
                        intlist.append(mapped_ref.index(ngram))
                        break

    ### At least two word correspondences are needed for rank correlation
    n = len(intlist)
    if n == 1 and len(ref) == 1:
        if debug > 1: print ("nkt=%g, precision=%g, bp=%g" % (1.0, 1.0/len(hyp), bp), file=sys.stderr)
        return (1.0, 1.0/len(hyp), bp)
    elif n < 2:
        # if not, return score 0.0
        if debug > 1: print ("nkt=%g, precision=%g, bp=%g" % (0.0, 0.0, bp), file=sys.stderr)
        return (0.0, 0.0, bp)

    ### calculation of rank correlation coefficient
    # count "ascending pairs" (intlist[i] < intlist[j])
    ascending = 0.0
    for i in range(len(intlist)-1):  
        for j in range(i+1,len(intlist)):
            if intlist[i] < intlist[j]:
                ascending += 1

    # normalize Kendall's tau
    nkt = ascending / ((n * (n - 1))/2)

    # calculate unigram precision
    precision = 1.0 * n / len(hyp)

    # return tuple (Normalized Kendall's tau, Unigram Precision, and Brevity Penalty)
    if debug > 1: print ("nkt=%g, precision=%g, bp=%g" % (nkt, precision, bp), file=sys.stderr)
    return (nkt, precision, bp)
