import base64

import PyPDF4
import numpy as np
from PIL.Image import Image
from stop_words import get_stop_words
from wordcloud import WordCloud
from wordcloud import WordCloud, STOPWORDS

from nltk import tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from operator import itemgetter
import math



def generateWordCloud(dataObject, hyperparameters):
    import matplotlib.pyplot as plt



    pdfBytes = base64.b64decode(dataObject, validate=True)
    if pdfBytes[0:4] != b'%PDF':
        raise ValueError('Missing the PDF file signature')
    f = open('app/TextAnalytics/file.pdf', 'wb')
    f.write(pdfBytes)
    f.close()

    texts = ''

    with open('app/TextAnalytics/file.pdf', 'rb') as paper:
        pdf = PyPDF4.PdfFileReader(paper)
        for page_num in range(pdf.getNumPages() - 1):  # skip reference
            page = pdf.getPage(page_num)
            texts += page.extractText()
    #print(texts)
    texts = ' '.join([word for word in texts.split() if word not in get_stop_words('english')])
    wordcloud = WordCloud(
        background_color='white',
        max_words=100,
        max_font_size=40,
        random_state=42
    ).generate(texts)
    fig = plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")

    import io
    s = io.BytesIO()
    plt.savefig(s, format='png', bbox_inches="tight")
    plt.savefig('foo.png')
    plt.close()
    s = base64.b64encode(s.getvalue()).decode("utf-8")
    return 1, s


def generateWordFrequency(dataObject, hyperParameters):
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from stop_words import get_stop_words
    from nltk.corpus import stopwords

    stop_words = list(get_stop_words('en'))  # About 900 stopwords
    nltk_words = list(stopwords.words('english'))  # About 150 stopwords
    stop_words.extend(nltk_words)

    pdfBytes = base64.b64decode(dataObject, validate=True)
    if pdfBytes[0:4] != b'%PDF':
        raise ValueError('Missing the PDF file signature')
    f = open('app/TextAnalytics/file.pdf', 'wb')
    f.write(pdfBytes)
    f.close()

    texts = ''

    with open('app/TextAnalytics/file.pdf', 'rb') as paper:
        pdf = PyPDF4.PdfFileReader(paper)
        for page_num in range(pdf.getNumPages() - 1):  # skip reference
            page = pdf.getPage(page_num)
            texts += page.extractText()

    texts = ' '.join([word for word in texts.split() if word not in get_stop_words('english')])
    total_words = texts.split()
    total_words = [word.lower() for word in total_words if word.isalpha()]
    total_word_length = len(total_words)
    total_sentences = tokenize.sent_tokenize(texts)
    total_sent_len = len(total_sentences)


    tf_score = {}
    for each_word in total_words:
        each_word = each_word.replace('.', '')
        if each_word not in stop_words:
            if each_word in tf_score:
                tf_score[each_word] += 1
            else:
                tf_score[each_word] = 1

    # Dividing by total_word_length for each dictionary element
    tf_score.update((x, y / int(total_word_length)) for x, y in tf_score.items())

    def check_sent(word, sentences):
        final = [all([w in x for w in word]) for x in sentences]
        sent_len = [sentences[i] for i in range(0, len(final)) if final[i]]
        return int(len(sent_len))

    idf_score = {}
    for each_word in total_words:
        each_word = each_word.replace('.', '')
        if each_word not in stop_words:
            if each_word in idf_score:
                idf_score[each_word] = check_sent(each_word, total_sentences)
            else:
                idf_score[each_word] = 1

    # Performing a log and divide
    idf_score.update((x, math.log(int(total_sent_len) / y)) for x, y in idf_score.items())
    tf_idf_score = {key: tf_score[key] * idf_score.get(key, 0) for key in tf_score.keys()}

    def get_top_n(dict_elem, n):
        result = dict(sorted(dict_elem.items(), key=itemgetter(1), reverse=True)[:n])
        return result


    return 1,get_top_n(tf_idf_score, 10)


def generateTopic(dataObject, hyperParameters):
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from stop_words import get_stop_words
    from nltk.corpus import stopwords

    stop_words = list(get_stop_words('en'))  # About 900 stopwords
    nltk_words = list(stopwords.words('english'))  # About 150 stopwords
    stop_words.extend(nltk_words)

    pdfBytes = base64.b64decode(dataObject, validate=True)
    if pdfBytes[0:4] != b'%PDF':
        raise ValueError('Missing the PDF file signature')
    f = open('app/TextAnalytics/file.pdf', 'wb')
    f.write(pdfBytes)
    f.close()

    texts = ''

    with open('app/TextAnalytics/file.pdf', 'rb') as paper:
        pdf = PyPDF4.PdfFileReader(paper)
        for page_num in range(pdf.getNumPages() - 1):  # skip reference
            page = pdf.getPage(page_num)
            texts += page.extractText()

    texts = ' '.join([word for word in texts.split() if word not in get_stop_words('english')])
    total_words = texts.split()
    total_words = [word.lower() for word in total_words if word.isalpha()]
    data_words = tokenize.sent_tokenize(texts)

    import gensim.corpora as corpora
    # Create Dictionary
    id2word = corpora.Dictionary(data_words)
    # Create Corpus
    texts = data_words
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    from pprint import pprint
    # number of topics
    num_topics = 10
    # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics)
    # Print the Keyword in the 10 topics
    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]

    import pyLDAvis.gensim
    import pickle
    import pyLDAvis
    import os
    # Visualize the topics
    pyLDAvis.enable_notebook()
    LDAvis_data_filepath = os.path.join('./results/ldavis_prepared_' + str(num_topics))
    # # this is a bit time consuming - make the if statement True
    # # if you want to execute visualization prep yourself
    if 1 == 1:
        LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
        with open(LDAvis_data_filepath, 'wb') as f:
            pickle.dump(LDAvis_prepared, f)
    # load the pre-prepared pyLDAvis data from disk
    with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)
    pyLDAvis.save_html(LDAvis_prepared, './results/ldavis_prepared_' + str(num_topics) + '.html')

    return None
