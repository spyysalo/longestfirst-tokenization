#!/usr/bin/env python3

import sys
import re
import logging

from collections import defaultdict, namedtuple
from datetime import datetime
from itertools import groupby
from logging import warning, info


Span = namedtuple('Span', ['start', 'end', 'max_len', 'tokenized'])

# Regex filtering BERT special tokens
FILTER_RE = re.compile(r'^\[(PAD|UNK|CLS|SEP|MASK|unused[0-9]+)\]')


class NoMatch(Exception):
    pass


def argparser():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('--trie', default=False, action='store_true',
                    help='Use trie instead of Aho-Corasick')
    ap.add_argument('--verbose', default=False, action='store_true')
    ap.add_argument('vocab')
    ap.add_argument('text')
    return ap


def _load_vocab(path):
    vocab = set()
    with open(path) as f:
        for l in f:
            l = l.rstrip('\n')
            if FILTER_RE.match(l):
                continue
            if l.isspace() or not l:
                continue
            vocab.add(l)
    return vocab


class AhocorasickTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab
        if any(v.startswith('^^') for v in vocab):
            raise ValueError('"^^" in vocab')   # this would break things
        self.automata_by_len = self.build_automata(vocab)
        self.max_len = max(self.automata_by_len.keys())

    def _pick_match(self, matches):
        # Return (start, end) span of first permissible match from
        # ahocorasick Automaton.iter() results
        for match in matches:
            match_end, match_len = match
            match_end += 1    # ahocorasick end is inclusive
            match_start = match_end - match_len
            if match_start in (1, 2) or match_end < 3:
                # Text start "^^" marker must be matched as a part of
                # a longer word piece (e.g. "^^M")
                continue
            return match_start, match_end
        raise NoMatch()

    def _longest_match(self, text, max_len, start, end):
        for length in reversed(range(1, max_len+1)):
            if length not in self.automata_by_len:
                continue
            matches = self.automata_by_len[length].iter(text, start, end)
            try:
                match_start, match_end = self._pick_match(matches)
            except NoMatch:
                continue    # nothing matched at this length
            return match_start, match_end, length
        raise NoMatch()    # nothing matched at all

    def _tokenize_iterative(self, text, max_len, start, end):
        if start == end:
            return []
        # Maintain a list of untokenized and tokenized text spans and
        # tokenize the first untokenized span until all are tokenized.
        make_span = lambda s, e, m, t: [] if s == e else [Span(s, e, m, t)]
        spans = make_span(start, end, max_len, False)
        first_untokenized = 0
        while True:
            for i in range(first_untokenized, len(spans)):
                if not spans[i].tokenized:
                    break
            else:
                break    # done, all tokenized
            span = spans[i]
            try:
                m_start, m_end, length = self._longest_match(
                    text, span.max_len, span.start, span.end)
            except NoMatch:
                raise ValueError('failed to tokenize: {}'.format(
                    text[span.start:span.end]))
            # length-1 for first part b/c match is leftmost of longest
            before = make_span(span.start, m_start, length-1, False)
            match = make_span(m_start, m_end, None, True)
            after = make_span(m_end, span.end, length, False)
            spans[i:i+1] = before + match + after
            first_untokenized = i
        return [text[s.start:s.end] for s in spans]

    def _tokenize_recursive(self, text, max_len, start, end):
        if start == end:
            return []
        try:
            match_start, match_end, length = self._longest_match(
                text, max_len, start, end)
            # length-1 for first part b/c match is leftmost of longest
            return (self._tokenize_recursive(text, length-1, start, match_start) +
                    [text[match_start:match_end]] +
                    self._tokenize_recursive(text, length, match_end, end))
        except NoMatch:
            raise ValueError('failed to tokenize: {}'.format(text[start:end]))

    def tokenize(self, text):
        text = '^^'+text    # see build_automata()
        # tokens = self._tokenize_recursive(text, self.max_len, 0, len(text))
        tokens = self._tokenize_iterative(text, self.max_len, 0, len(text))
        equal = ''.join(tokens) == text, 'internal error'
        # Map to wordpiece continuation convention
        wptokens = [t[2:] if t.startswith('^^') else '##'+t for t in tokens]
        return wptokens

    @classmethod
    def load(cls, vocab_path):
        vocab = _load_vocab(vocab_path)
        return cls(vocab)

    @staticmethod
    def build_automata(vocab):
        # Build Aho-Corasick matching automata for vocabulary items
        # grouped by length. The wordpiece convention is inverted for
        # matching: continuations are unmarked (instead of "##") and
        # string start is marked by "^^".
        from ahocorasick import Automaton
        start_time = datetime.now()
        info('start building automata at {}'.format(
            start_time.strftime("%H:%M:%S")))
        strings = [v[2:] if v.startswith('##') else '^^'+v for v in vocab]
        max_len = max(len(s) for s in strings)
        strings.sort(key=lambda s: len(s))
        strings_by_len = defaultdict(list)
        for k, g in groupby(strings, lambda s: len(s)):
            strings_by_len[k] = list(g)
        automata_by_len = {}
        for i in range(1, max_len+1):
            if i not in strings_by_len:
                continue
            a = Automaton()
            for s in strings_by_len[i]:
                a.add_word(s, i)
            a.make_automaton()
            automata_by_len[i] = a
        end_time = datetime.now()
        info('finish building automata at {} (delta {})'.format(
            end_time.strftime("%H:%M:%S"), end_time-start_time))
        return automata_by_len


class TrieTokenizer(object):
    def __init__(self, alphabet, vocab):
        # The wordpiece convention is inverted for matching:
        # continuations are unmarked (instead of "##") and string
        # start is marked by "^^".
        vocab = [v[2:] if v.startswith('##') else '^^'+v for v in vocab]
        alphabet = list(alphabet)
        for required in ('^', '#'):
            if required not in alphabet:
                alphabet.append(required)
        self.trie = self.build_trie(alphabet, vocab)
        self._longest = None

    def _init_longest(self, text):
        self._longest = [0] * len(text)
        for i in range(len(text)):
            p = self.trie.longest_prefix(text[i:]).key
            if p is not None:
                self._longest[i] = len(p)

    def _update_longest(self, text, start, end):
        # Update self._longest for range [start:end] to contain the
        # length of the longest possible match for each position,
        # assuring that no match goes past the end position.
        for i in range(start, end):
            if i + self._longest[i] > end:
                p = self.trie.longest_prefix(text[i:end]).key
                if p is not None:
                    self._longest[i] = len(p)

    def _longest_match(self, text, start, end):
        self._update_longest(text, start, end)
        longest = self._longest[start:end]
        leftmost_longest = sorted([(l, i) for i, l in enumerate(longest)],
                                  key=lambda k: (-k[0], k[1]))
        # grab first permissible match
        match_start = match_end = None
        for length, offset in leftmost_longest:
            match_start, match_end = start+offset, start+offset+length
            # Text start "^^" must be matched as a part of a longer
            # word piece (e.g. "^^M")
            if match_start not in (1, 2) and match_end >= 3:
                break
        if match_start == match_end:
            raise NoMatch('failed to match "{}", {}:{} in "{}"'.format(
                text[start:end], start, end, text))
        assert match_start >= start and match_end <= end, '{}:{} for {}:{}'.format(match_start, match_end, start, end)
        return match_start, match_end

    def _tokenize_iterative(self, text, start, end):
        if start == end:
            return []
        self._init_longest(text)
        # Maintain a list of untokenized and tokenized text spans and
        # tokenize the first untokenized span until all are tokenized.
        make_span = lambda s, e, t: [] if s == e else [Span(s, e, None, t)]
        spans = make_span(start, end, False)
        first_untokenized = 0
        while True:
            for i in range(first_untokenized, len(spans)):
                if not spans[i].tokenized:
                    break
            else:
                break    # done, all tokenized
            span = spans[i]
            m_start, m_end = self._longest_match(
                text, span.start, span.end)
            before = make_span(span.start, m_start, False)
            match = make_span(m_start, m_end, True)
            after = make_span(m_end, span.end, False)
            spans[i:i+1] = before + match + after
            first_untokenized = i
        return [text[s.start:s.end] for s in spans]

    def _tokenize_recursive(self, text, start, end):
        if start == end:
            return []
        try:
            match_start, match_end = self._longest_match(text, start, end)
            return (self._tokenize_recursive(text, start, match_start) +
                    [text[match_start:match_end]] +
                    self._tokenize_recursive(text, match_end, end))
        except NoMatch:
            raise ValueError('failed to tokenize: {}'.format(text[start:end]))

    def tokenize(self, text):
        text = '^^'+text    # see __init__()
        tokens = self._tokenize_iterative(text, 0, len(text))
        # Map to wordpiece continuation convention
        wptokens = [t[2:] if t.startswith('^^') else '##'+t for t in tokens]
        return wptokens

    @classmethod
    def load(cls, vocab_path):
        vocab = _load_vocab(vocab_path)
        alphabet = sorted(list(set(c for v in vocab for c in v)))
        return cls(alphabet, vocab)

    @staticmethod
    def build_trie(alphabet, vocab):
        from pygtrie import CharTrie as Trie
        trie = Trie()
        start_time = datetime.now()
        info('start building trie at {}'.format(
            start_time.strftime("%H:%M:%S")))
        for v in vocab:
            trie[v] = 1
        end_time = datetime.now()
        info('finish building trie at {} (delta {})'.format(
            end_time.strftime("%H:%M:%S"), end_time-start_time))
        return trie


def main(argv):
    args = argparser().parse_args(argv[1:])
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    if args.trie:
        Tokenizer = TrieTokenizer
    else:
        Tokenizer = AhocorasickTokenizer
    tokenizer = Tokenizer.load(args.vocab)
    with open(args.text) as f:
        for l in f:
            text = l.rstrip()
            try:
                tokens = tokenizer.tokenize(text)
            except:
                print('failed: {}'.format(text), file=sys.stderr)
                raise
            print(' '.join(tokens))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))

