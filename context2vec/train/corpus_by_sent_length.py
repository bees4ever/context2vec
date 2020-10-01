'''
Converts a single large corpus file into a directory, in which for every sentence length k there is a separate file containing all sentences of that length. 
'''

import sys
import os
from collections import Counter
from context2vec.common.defs import SENT_COUNTS_FILENAME, WORD_COUNTS_FILENAME, TOTAL_COUNTS_FILENAME
from context2vec.train.sentence_reader import SentenceReaderDict


def get_file(sub_files, corpus_dir, num_filename):
    if num_filename not in sub_files:
        full_file_name = corpus_dir + '/' + num_filename
        sub_files[num_filename] = open(full_file_name, 'w')        
    return sub_files[num_filename]
   

corpus = [['till', 'this', 'moment', 'i', 'never', 'knew', 'myself', '.'],
               ['seldom', ',', 'very', 'seldom', ',', 'does', 'complete', 'truth', 'belong', 'to', 'any', 'human',
                'disclosure', ';', 'seldom', 'can', 'it', 'happen', 'that', 'something', 'is', 'not', 'a', 'little',
                'disguised', 'or', 'a', 'little', 'mistaken', '.'],
               ['i', 'declare', 'after', 'all', 'there', 'is', 'no', 'enjoyment', 'like', 'reading', '!', 'how', 'much',
                'sooner', 'one', 'tires', 'of', 'anything', 'than', 'of', 'a', 'book', '!', '”'],
               ['men', 'have', 'had', 'every', 'advantage', 'of', 'us', 'in', 'telling', 'their', 'own', 'story', '.',
                'education', 'has', 'been', 'theirs', 'in', 'so', 'much', 'higher', 'a', 'degree'],
               ['i', 'wish', ',', 'as', 'well', 'as', 'everybody', 'else', ',', 'to', 'be', 'perfectly', 'happy', ';',
                'but', ',', 'like', 'everybody', 'else', ',', 'it', 'must', 'be', 'in', 'my', 'own', 'way', '.'],
               ['there', 'are', 'people', ',', 'who', 'the', 'more', 'you', 'do', 'for', 'them', ',', 'the', 'less',
                'they', 'will', 'do', 'for', 'themselves', '.'],
               ['one', 'half', 'of', 'the', 'world', 'can', 'not', 'understand', 'the', 'pleasures', 'of', 'the',
                'other', '.']]


def read_in_corpus(corpus: list):
    max_sent_len = 128  # TODO parameter of method can replace it

    sent_counts = Counter()
    word_counts = Counter()


    batches = {} # wordnum is mapped to doc

    for words in corpus:
        wordnum = len(words)
        if 1 < wordnum <= max_sent_len:
            if wordnum not in batches.keys():
                batches[wordnum] = []

            batches[wordnum].append(words)
            sent_counts[wordnum] += 1
            for word in words:
                word_counts[word] += 1

    print('total sents read: {}\n'.format(sum(sent_counts.values())))
    print('total words read: {}\n'.format(sum(word_counts.values())))


    return {'batches': batches, 'sent_counts': sent_counts, 'word_counts': word_counts}

if __name__ == '__main__':

    prepared_corpus = read_in_corpus(corpus)

    print(prepared_corpus)
    trimfreq = 0 ## default
    batchsize = 100 ## default
    reader = SentenceReaderDict(prepared_corpus, trimfreq, batchsize)
    reader.open()

    #
    # if len(sys.argv) < 2:
    #     print("usage: %s <corpus-file> [max-sent-len]"  % (sys.argv[0]))
    #     sys.exit(1)
    #
    # corpus_file = open(sys.argv[1], 'r')
    # if len(sys.argv) > 2:
    #     max_sent_len = int(sys.argv[2])
    # else:
    #     max_sent_len = 128
    # print('Using maximum sentence length: ' + str(max_sent_len))
    #
    # corpus_dir = sys.argv[1]+'.DIR'
    # os.makedirs(corpus_dir)
    # sent_counts_file = open(corpus_dir+'/'+SENT_COUNTS_FILENAME, 'w')
    # word_counts_file = open(corpus_dir+'/'+WORD_COUNTS_FILENAME, 'w')
    # totals_file = open(corpus_dir+'/'+TOTAL_COUNTS_FILENAME, 'w')
    #
    # sub_files = {}
    # sent_counts = Counter()
    # word_counts = Counter()
    #
    # for line in corpus_file:
    #     words = line.strip().lower().split()
    #     wordnum = len(words)
    #     if wordnum > 1 and wordnum <= max_sent_len:
    #         num_filename = 'sent.' + str(wordnum)
    #         sub_file = get_file(sub_files, corpus_dir, num_filename)
    #         sub_file.write(line)
    #         sent_counts[num_filename] += 1
    #         for word in words:
    #             word_counts[word] += 1
    #
    # for sub_file in sub_files.values():
    #     sub_file.close()
    #
    # for num_filename, count in sent_counts.most_common():
    #     sent_counts_file.write(num_filename+'\t'+str(count)+'\n')
    #
    # for word, count in word_counts.most_common():
    #     word_counts_file.write(word+'\t'+str(count)+'\n')
    #
    # totals_file.write('total sents read: {}\n'.format(sum(sent_counts.values())))
    # totals_file.write('total words read: {}\n'.format(sum(word_counts.values())))
    #
    # corpus_file.close()
    # sent_counts_file.close()
    # word_counts_file.close()
    # totals_file.close()
    #
    # print('Done')