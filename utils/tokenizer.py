"""
    this script tokenize hebrew input in two stages:
    1) seperate to sentnces
    2) seperate to words (as further input to Open Univ.'s YAP)
"""
# sourced from https://github.com/ygurman/hebrew_tokenizer/blob/master/tokenizer.py

import re
from typing import List

# special chars and regexes
# punctuations
re_paranthesisOpen = r"[\(\[\{\'\"`]"
re_paranthesisClose = r"[\)\]\}\'\"`]"
re_sentenceSeperators = r"[\.!?]"
re_internalPunct = r"[,;:\-&]"
# no space sequences
re_noSpaceSequence = r"[^ \t\f\v]+(?:[\n][^ \t\f\v]*)*" #including endline for sentence seperation
#regexes with quotemarks
re_nonAcronymQuoteMarks = r"(?P<head>\w)(?P<mark>[\"\'])(?P<tail>\w{2,})" # catches non acronym use of quote marks
# regexs with non-seperating dots
re_numbering = r"(?:(?:[א-י]|\d+)\.)+"
re_heb_dot_acronym = u"(?:(?:[א-ת]\.)+[א-ת]+)" #type: ignore
re_numeric = r"(?:[+-]?(?:[0-9][0-9.,\/\-:]*)?(?:[0-9])%?)"
# GENDER_NEUTRAL_SUFFIXES = [".ם",".ן","ת.ים",".ות",".ה",".ית",".ת"]
# re_genderNeutral = re_hebword + "|".join(GENDER_NEUTRAL_SUFFIXES)
re_3dotsStyleSequence = re_sentenceSeperators + "{2,}"
re_legalWithSeperator = r"{0}*{1}{2}*".format(re_paranthesisOpen
                                             ,"|".join((re_numbering, re_numeric, re_heb_dot_acronym
                                                        #, re_genderNeutral
                                                        ))
                                             ,re_paranthesisClose)
# url
re_url = r"[a-z]+://\S+"
# english words
re_eng = r"[a-zA-Z][a-zA-Z0-9'.]*"
# garbage - all non matching characters for tokenizer simplification
re_garbage = r"[^א-תa-zA-Z0-9!?.,:;\-()\[\]{}]+"
# hebrew words
re_hebword = u"(?:[\"\']?[א-ת]+[\"\']?[א-ת]+[\"\']?)"
re_heb_word_plus = r"[א-ת]([.'`\"\-/\\]?['`]?[א-ת0-9'`])*"
re_hebrew_dash=re_hebword+"-"+re_hebword
# definite end of sentences
re_sentenceEnd = r"(?:{0}{1}\n*)|\n+".format(re_paranthesisClose, re_sentenceSeperators)

def text2listOfSentences(conc_sent:str) -> List[str]:
    """
    splits concatenated string to sentences w/o prior assumptions by classifying optional seperators ("!" "?" "." and "\n")
    :param conc_sent: input string, assuming may or may not contain line seperators.
    :return: list of sentences
    """
    sentences = []
    current_sentence = []

    # iterate over sequencese seperated with whitespaces (keep new-line signs)
    for suspect_sequence in re.findall(pattern=re_noSpaceSequence, string=conc_sent, flags=(re.MULTILINE|re.UNICODE)):
        current_start = 0
        # iterate over suspect sequnce
        i = 0

        if len(sentences) == 57:
            x=8

        while i < len(suspect_sequence) :            # handle definite sentence endings (quotes and paranthesis ending followd by seperator or newline signs)
            match_end_sentence = re.match(pattern=re_sentenceEnd, string=suspect_sequence[i:])
            if match_end_sentence:
                # append previous sequence to sentence
                current_sentence.append(suspect_sequence[current_start:i])
                # append charecters seperatly (except for newline)
                current_sentence.extend(c for c in suspect_sequence[i:i+match_end_sentence.end()] if c!= '\n')
                # add the current sentence to the sentences list
                sentences.append(current_sentence)
                current_sentence = []
                # update indices
                i += match_end_sentence.end()
                current_start = i
                continue

            # for sentence seperation, only suspect seperators (!?.) matter
            if suspect_sequence[i] in ["!","?","."]:
                # handle multiple seperator sequence (by design choice not a sentence seperator #
                match_multiple_seps = re.match(pattern=re_3dotsStyleSequence,string=suspect_sequence[i:])
                if match_multiple_seps:
                    # add previous sequence
                    current_sentence.append(suspect_sequence[current_start:i])
                    # add multiple-seperators sequence
                    current_sentence.append(suspect_sequence[i:i+match_multiple_seps.end()])
                    # update indices
                    i += match_multiple_seps.end()
                    current_start = i
                    continue

                # seperator prior to quote mark doesn't end a sentence (while in quote) - i.e. no characters after the quotemark
                match_sep_before_closing_paranthesis = re.match(pattern=re_sentenceSeperators+re_paranthesisClose+"+$",string=suspect_sequence[i:])
                if match_sep_before_closing_paranthesis:
                    # add previous sequence
                    current_sentence.append(suspect_sequence[current_start:i])
                    # add seperator and closing paranthesis
                    current_sentence.extend([c for c in suspect_sequence[i+match_sep_before_closing_paranthesis.start():i+match_sep_before_closing_paranthesis.end()]])
                    #update indices
                    i += match_sep_before_closing_paranthesis.end()
                    current_start = i

                # assume [?!] in middle of string indicate sentence seperation (or "." before whitespace)
                elif suspect_sequence[i] in ["!", "?"] or ((suspect_sequence[i] == ".") and (i == len(suspect_sequence) - 1)):
                    # add previous sequence
                    current_sentence.append(suspect_sequence[current_start:i])
                    # add seperator
                    current_sentence.append(suspect_sequence[i])
                    # add the current sentence to the sentences
                    sentences.append(current_sentence)
                    current_sentence = []
                    # update indices
                    i += 1
                    current_start = i
                    continue

                # if seperator is "." check if it's a part of a legal token or a sentence seperator
                else:
                    match_legal_token_with_dots = re.match(pattern=re_legalWithSeperator,string=suspect_sequence[current_start:])
                    # if a part of legal token (dotted acronym, numerical, numbering, gender-neutral)
                    if match_legal_token_with_dots:
                        # add previous sequence
                        current_sentence.append(suspect_sequence[current_start:match_legal_token_with_dots.end()])
                        # update indices
                        i = current_start + match_legal_token_with_dots.end()
                        current_start = i
                        continue
                    # if not part of legal token, act as with seperator
                    else:
                        # add previous sequence
                        current_sentence.append(suspect_sequence[current_start:i])
                        # add seperator
                        current_sentence.append(suspect_sequence[i])
                        # add sentence to list
                        sentences.append(current_sentence)
                        current_sentence = []
                        # update indices
                        i+= 1
                        current_start = i
                        continue
            # if character is not a seperator move on
            i += 1

        # reching the suspect sequence end add the leftover to the sentence and finish
        if current_start < len(suspect_sequence) - 1:
            current_sentence.append(suspect_sequence[current_start:])

    return [" ".join(sent) for sent in sentences]

# handler functions for the Scanner (used as lambdas for regex and function in Scanner)
def handleUrl(s,t): return ('URL',t)
def handleEng(s,t): return ('ENG',t)
def handleHeb(s,t):
    # flip order in a non acronym quote marks
    match_nonAcronymQuoteMark = re.match(pattern=re_nonAcronymQuoteMarks, string=t, flags=(re.UNICODE))
    if match_nonAcronymQuoteMark:
        t = " ".join((match_nonAcronymQuoteMark.group("mark"),match_nonAcronymQuoteMark.group("head")+match_nonAcronymQuoteMark.group("tail")))
    # seperate inner dashes
    t = t.replace("-", " - ")
    return ('HEB',t)

def handleNonAcronymQuoteMark(s,t):
    # flip the order of the quote-marks to prevent segmentation mistakes in YAP
    match = re.match(pattern=re_nonAcronymQuoteMarks,string=t,flags=(re.UNICODE))
    t = " ".join((match.group("mark"), match.group("head")+match.group("tail"))) #type: ignore
    return('HEB',t)

def handleNum(s,t):
    if (t[-1] == "%"):
        t = t[:-1] + " %"
    return ('NUM',t)

def handlePunct(s,t): return ('PUNCT',t)
def handleGarbage(s,t): return ('GARBAGE', t)

# scanner - a built-in re package tool
scanner = re.Scanner([ #type: ignore
    (r"\s+", None),
    (re_url, handleUrl),
    (re_legalWithSeperator, handleHeb),
    (re_nonAcronymQuoteMarks, handleNonAcronymQuoteMark),
    (re_heb_word_plus,handleHeb),
    (re_eng, handleEng),
    (re_numeric, handleNum),
    (re_numbering, handleNum),
    (re_paranthesisOpen, handlePunct),
    (re_paranthesisClose, handlePunct),
    (re_3dotsStyleSequence, handlePunct),
    (re_sentenceSeperators, handlePunct),
    (re_internalPunct, handlePunct),
    (re_garbage, handleGarbage)
])

def tokenize(sent:str):
    tok = sent
    parts, remainder = scanner.scan(tok)
    assert (not remainder)
    return parts

def tokenize_sentences(input_sentences:List[str]) -> List[List[str]]:
    """
    an re.Scanner based tokenizer. based on Yoav Goldberg's Hebrew Tokenizer (2010) - https://www.cs.bgu.ac.il/~yoavg/software/hebtokenizer/
    seperates input text to sentences as list of words using regular expressions
    :param input_sentences: list of sentences in Hebrew (and possibly some English words or characters)
    :return: list of sentences as list of words
    """
    tokenized_sentences = []
    for sent in input_sentences:
        tokenized_sentences.append([tok for (which,tok) in tokenize(sent)])
    return tokenized_sentences


def listOfSents2File(sentences, path):
    """
    writes tokenized sentences (seperated by new lines) to output path
    :param sentences: list of sentences (each as list of words)
    :param path: output file path
    """
    with open(file=path, mode='wt', encoding='utf8') as f:
        for sent in sentences:
            f.write(" ".join(sent) + "\n")


def parseArgs():
    """
    handle program arguments
    :return: input path, out path and boolean indicating sentence seperation
    """
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", help="textual input file", metavar="FILE")
    parser.add_argument("output", nargs='?', help="textual output file. defaults to input file with \".tokenized\" "
                                                  "suffix", metavar="FILE")
    parser.add_argument("-s", "--seperate", dest="seperate", action="store_true",help="indicates that the script "
                                                                                      "should seperate the input to "
                                                                                      "sentences. NOTICE: RECOMMENDED "
                                                                                      "IF INPUT CONSISTS OF "
                                                                                      "SENTENCES WITHOUT 'NEWLINE' "
                                                                                      "BETWEEN THEM "
                        , required=False, default=False)
    args = parser.parse_args()
    input_path = args.input
    output_path = args.output if args.output else (input_path + ".tokenized")

    return input_path, output_path, args.seperate

def main():
    # handle input file
    _input_path, _output_path, _seperate2sentences = parseArgs()

    # read input file and handle windows-originated newline seperators and double single-quote marks with double quote mark
    input_sentences = open(file=_input_path, mode='rt', encoding='utf8').read().replace('\r', '').replace("\'\'", "\"")

    # seperate to sentences (if "-s" flag is off, assume sentences seperated with "\n"
    if _seperate2sentences:
        input_sentences = text2listOfSentences(input_sentences)
    else:
        input_sentences = input_sentences.split('\n')

    # tokenize input
    tokenized_sentences = tokenize_sentences(input_sentences)

    # write tokenized output to files
    listOfSents2File(sentences = tokenized_sentences, path = _output_path)


if  __name__ == '__main__':
    main()