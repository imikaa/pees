import streamlit as st
import numpy as np
import pandas as pd
import os
from streamlit_option_menu import option_menu
import string # for removing punctuation
import nltk #natural language toolkit
from nltk import word_tokenize, download, stem, RegexpTokenizer #preprocessing
from nltk.corpus import stopwords, words #remove stopwords
import itertools, math
import matplotlib.pyplot as plt
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')

with st.sidebar :
    choose = option_menu("Menu", ["Home", "Descriptors", "Information Retrieval", "Evaluation", "Dictionary per document", "TF-IDF", "Contact", "Help"],
                         icons=['house', 'file-earmark-binary', 'search', 'graph-up', 'table 2 columns', 'search', 'stars', 'person lines fill'],
                         menu_icon="menu-app",
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )

def read_data():
    doc_names = []
    for i in range( 6004 ) :
        doc_names.append( "D"+str(i+1) )
    docs = []
    for doc in doc_names :
        with open( 'Documents/'+doc, 'r' ) as file :
            docs.append( ' '.join( file.read().split() ) )
    return docs

def clean_preprocess( docs, tokenization_method, stemmer ) :
    new_docs = docs.copy()
    if( stemmer == 'Porter stemmer' ) :
        stemmer = stem.PorterStemmer()
    elif( stemmer == 'Lancaster stemmer' ) :
        stemmer = stem.LancasterStemmer()
        
    for i in range( len(new_docs) ) :
        new_docs[i] = new_docs[i].lower() # cleaning : lower case
        
        if( tokenization_method == 'nltk.RegexpTokenizer.tokenize()' ) :
            ExpReg = RegexpTokenizer('(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.\,]\d+)?%?|\w+(?:[\-/]\w+)*') 
            tokens = ExpReg.tokenize( new_docs[i] ) # preprocessing : tokenization
        elif( tokenization_method == 'split()' ) :
            tokens = new_docs[i].split()

        stop_words = nltk.corpus.stopwords.words( 'english' ) # preprocessing : stop words
        tokens = [token for token in tokens if token not in stop_words]# preprocessing : stop words removal
            
        new_docs[i] = ' '.join( [stemmer.stem(token) for token in tokens] ) # preprocessing : stemming
    return new_docs

def create_dicts(docs):
    dicts = []
    for i in range( len(docs) ) :
        words = list( docs[i].split(" ") )
        dictionary = {}
        for token in words :
            if token in dictionary.keys():
                dictionary[token] += 1
            else:
                dictionary[token] = 1
        
        dicts.append( dictionary )
    return dicts

def show_dicts(dict):
    words = [key for key, _ in dict.items()]
    freq = [value for _, value in dict.items()]

    dict_df = pd.DataFrame( {'Words':words, 'Occurrences':freq } )
    st.markdown( dict_df.style.hide( axis="index" ).to_html(), unsafe_allow_html = True )

def create_dict(docs):
    words = []
    distinct_words = []
    
    for i in range( len(docs) ) :
        tokens = list( docs[i].split(" ") )
        words.append( tokens )
        distinct_words.extend( tokens )
    
    distinct_words = sorted( list( set( distinct_words ) ) )

    doc_names = []
    for i in range( len(docs) ) :
        doc_names.append( "D"+str(i+1) )

    dictionary = { key : 0 for key in list( itertools.product( distinct_words, doc_names ) ) }
    for i in range( len(docs) ) :
        doc = 'D'+str(i+1)
        for word in words[i] :
            dictionary[ (word, doc) ] += 1

    return dictionary    

def show_dict( dict ):
    words = [key[0] for key, _ in dict.items()]
    docs = [key[1] for key, _ in dict.items()]
    freq = [value for _, value in dict.items()]

    dict_df = pd.DataFrame( {'Words':words, 'Documents':docs, 'Occurrences':freq } )
    st.markdown( dict_df.style.hide( axis="index" ).to_html(), unsafe_allow_html = True )
    
def query( tfidf, tokens, tokenization_method, stemmer ) :
    tokens = clean_preprocess( tokens, tokenization_method, stemmer )
    answers = [ "{:<10} {:<10} {:<10} {:<10}".format( 'Word', 'Document', 'Frequency', 'Weight') ]
    for line in tfidf :
        items = line.split()
        if( items[0] in tokens ) :
            answers.append( "{:<10} {:<10} {:<10} {:<10}".format( items[0], items[1], items[2], items[3] ) )
    if len(answers) == 1 :
        answers = []
    return answers

def matching( tfidf, tokens, tokenization_method, stemmer, matching_method ) :
    tokens = clean_preprocess( tokens, tokenization_method, stemmer )
    answers = []
    tfidff = tfidf.readlines()
    #st.write( type(tfidf), type(tfidff) )
    for line in tfidff[1:] :
        items = line.split()
        if( items[0] in tokens ) :
            answers.append( (items[0], items[1], float(items[3])) )
    if len(answers) == 0 :
        return answers
    
    documents = list( set( list(zip(*answers))[1] ) )
    sum_ = []
    for document in documents :
        sum_.append( round(np.sum( [answers[i][2] for i in range(len(answers)) if answers[i][1]==document] ), 4) )
    
    if( matching_method == 'scalar product' ) :    
        answers = sorted(list(zip(documents, sum_)), key=lambda x: x[1], reverse = True)
        return ["{:<10} {:<10}".format( answers[i][0], answers[i][1] ) for i in range(len(answers)) ]
    
    elif matching_method == 'cosine measure' :
        valid_tokens = list( set( list(zip(*answers))[0] ) )
        cosine_measure = []
        for i in range(len(sum_)) :
            sum_v = len( valid_tokens )
            sum_w = np.sum( [(float(line.split()[3]))**2 for line in tfidff if line.split()[1]==documents[i]] )
            #sum_w = np.sum( [(answers[j][1])**2 for j in range(len(answers)) if answers[j][0]==documents[i]] )
            cosine_measure.append( round(sum_[i] / ( math.sqrt(sum_v) * math.sqrt(sum_w) ), 4) )
        answers = sorted(list(zip(documents, cosine_measure)), key=lambda x: x[1], reverse = True)
        return ["{:<10} {:<10}".format( answers[i][0], answers[i][1] ) for i in range(len(answers)) ]
    
    elif matching_method == 'jaccard measure' :
        valid_tokens = list( set( list(zip(*answers))[0] ) )
        jaccard_measure = []
        for i in range(len(sum_)) :
            sum_v = sum_v = len( valid_tokens )
            sum_w = np.sum( [(float(line.split()[3]))**2 for line in tfidff if line.split()[1]==documents[i]] )
            jaccard_measure.append( round(sum_[i] / ( sum_v + sum_w - sum_[i] ), 4) )
        answers = sorted(list(zip(documents, jaccard_measure)), key=lambda x: x[1], reverse = True)
        return ["{:<10} {:<10}".format( answers[i][0], answers[i][1] ) for i in range(len(answers)) ]

def matching_bm25( tfidf, tokens, tokenization_method, stemmer, k, b ) :
    tokens = clean_preprocess( tokens, tokenization_method, stemmer )
    answers = []
    all_docs = []
    tfidff = tfidf.readlines()
    for line in tfidff[1:] :
        items = line.split()
        if items[1] not in all_docs :
            all_docs.append( items[1] )
        if( items[0] in tokens ) :
            answers.append( (items[0], items[1], items[2], items[3]) )
    if len(answers) == 0 :
        return answers
                        
    documents = sorted( list( set( list(zip(*answers))[1] ) ) )
    all_docs = sorted( all_docs )                       
    nbr_docs = len( all_docs )
                           
    lengths = [0 for i in range(nbr_docs)]                       
    for line in tfidff[1:] :
        items = line.split()
        lengths[int(items[1][-1])-1] += int(items[2])
    av_ln = np.mean( lengths )
    
    nbr_docs_having_token = [len([answers[i] for i in range(len(answers)) if answers[i][0] == token]) for token in tokens]
    
    relevances = []
    for i in range( len(documents) ) :
        data = [answers[j] for j in range(len(answers)) if answers[j][1] == documents[i]]
        data_dict = {(item[0], item[1]): (item[2], item[3]) for item in data}
        tokens_doc = [tokens[j] for j in range(len(tokens)) if tokens[j] in list(set(list(zip(*data))[0]))]
        cons = k * ( (1-b) + b*lengths[int(documents[i][-1])-1]/av_ln )
        relevances.append( round( np.sum([ int(data_dict[(tokens[j], documents[i])][0]) / (cons+int(data_dict[(tokens[j], documents[i])][0])) * math.log10( (nbr_docs-nbr_docs_having_token[j]+0.5)/(nbr_docs_having_token[j]+0.5) ) for j in range(len(tokens)) if tokens[j] in tokens_doc ]), 4 ) )
        
    answers = sorted(list(zip(documents, relevances)), key=lambda x: x[1], reverse = True)
    return ["{:<10} {:<10}".format( answers[i][0], answers[i][1] ) for i in range(len(answers)) ]

def query_valid( tokens ) :
    bool_oprtators = [ 'and', 'or', 'not' ]
    nbr_tokens = len( tokens )
    previous = 'blank'
    for i in range( nbr_tokens ) :
        if tokens[i] in bool_oprtators :
            if tokens[i] == 'not' :
                if i == nbr_tokens-1 :
                    return False
                if previous == 'bool_not' :
                    return False
                previous = 'bool_not'
            else :
                if i == 0 or i == nbr_tokens-1 :
                    return False
                if previous in ['bool_op', 'bool_not'] :
                    return False
                previous = 'bool_op'
        else :
            if previous == 'word' :
                return False
            if i != nbr_tokens-1 and tokens[i+1] == 'not' :
                return False
            previous = 'word'
    return True

def query_to_tokens( query, tokenization_method, stemmer ) :
    bool_operators = [ 'and', 'or', 'not' ]
    a_list = [item for item in enumerate( [token for token in query] ) ]
    nbr_items = len( a_list )
    b_list = [(num, token) for num, token in a_list if token not in bool_operators]
    nums, tokens = list(zip(*b_list))
    nums = list( nums )
    tokens = list( tokens )
    tokens = clean_preprocess( list(tokens), tokenization_method, stemmer )
    i = 0
    for num in nums :
        query[num] = tokens[i]
        i += 1
    return query

def build_index(documents):
    index = {}
    for doc_id, document in enumerate(documents):
        terms = set(document.split())
        for term in terms:
            if term not in index:
                index[term] = set()
            index[term].add(doc_id)
    return index

def matching_boolean( documents, query, tokenization_method, stemmer ) :
    query = list(query.lower().split())
    if not query_valid( query ) :
        return None
    tokens = query_to_tokens( query, tokenization_method, stemmer )
    set_doc = set( [i for i in range( len(documents) )] )
    
    index = build_index(documents)
    result = set(index.get(tokens[0], []))
    if tokens[0] in {'and', 'or', 'not'}:
        operator = tokens[0]
    
    for i in range( 1, len(tokens) ) :
        if tokens[i] in {'and', 'or', 'not'}:
            operator = tokens[i]
        else:
            term_set = set(index.get(tokens[i], []))

            if operator == 'and':
                result = result.intersection(term_set)
            elif operator == 'or':
                result = result.union(term_set)
            elif operator == 'not':
                if i == 1 :
                    result = set_doc - set(index.get(tokens[i], []))
                else :
                    result = result.difference(term_set)
    result = set( [res+1 for res in result] )                
    return ["{:<10} {:<10}".format( 'D'+str(i), 'Yes' ) for i in result ]

def precision( query_result, judgements, arg = None ) :
    documents_selec = len( query_result )
    query_result = [query_result[i].split() for i in range(documents_selec)]
    if arg != None :
        query_result = query_result[:arg]
        documents_selec = arg
    judg = [j.split(' ')[1] for j in judgements]
    pertinents_selec = len( [doc for doc in query_result if re.findall(r'\d+',doc[0])[0] in judg] )
    return float( pertinents_selec / documents_selec )

def recall( query_result, judgements ) :
    query_result = [query_result[i].split() for i in range(len(query_result))]
    documents_perti = len( judgements )
    judg = [j.split(' ')[1] for j in judgements]
    pertinents_selec = len( [doc for doc in query_result if re.findall(r'\d+',doc[0])[0] in judg] )
    return float( pertinents_selec / documents_perti )

def fscore( precision, recall ) :
    if (precision+recall) == 0 :
        return 0
    return round( float( 2*precision*recall / (precision+recall) ), 4 )

def interpolated_curve( query_result, judgements ) :
    # classement 
    query_result = [query_result[i].split() for i in range(len(query_result))]
    classement = list( zip( *query_result ) )[0]
    
    # pertinent
    pertinent = [ True if clas[-1] in list( zip( *judgements ) )[-1] else False for clas in classement ]
    
    # precision and recall
    precision = []
    recall = []
    nbr = len( classement )
    nbr_pert = len( judgements )
    nbr_true = 0
    for i in range( nbr ) :
        if( pertinent[i] == True ) :
            nbr_true += 1
        precision.append( float( nbr_true / (i+1) ) )
        recall.append( float( nbr_true / nbr_pert ) )
        
    # iterpolation
    interpo_recall = [i / 10.0 for i in range(11)]
    interpo_precision = []
    for interpo_rec in interpo_recall :
        interpo_precision.append( max( [precision[i] for i in range(nbr) if recall[i] >= interpo_rec] ) )
        
    return list( zip( interpo_recall, interpo_precision ) )

if choose == "Home" :
    st.title( "Information Representation and Retrieval : Evaluation" )

elif choose == "Dictionary per document" :
    st.title( "Dictionary per document" )
    
    docs = read_data()
    option = st.selectbox( "Choose the term extraction method : ", ('-', 'split()', 'nltk.RegexpTokenizer.tokenize()') )
    
    option1 = st.selectbox( "Choose the stemmer : ", ('-', 'Porter stemmer', 'Lancaster stemmer') )
    st.write("")
    st.write("")
    st.write("")

    if( option == 'split()' and option1 == 'Porter stemmer' ) :
        preprocessed_docs = clean_preprocess(docs, 'split()', 'Porter stemmer')
    if( option == 'split()' and option1 == 'Lancaster stemmer' ) :
        preprocessed_docs = clean_preprocess(docs, 'split()', 'Lancaster stemmer')
    if( option == 'nltk.RegexpTokenizer.tokenize()' and option1 == 'Porter stemmer' ) :
        preprocessed_docs = clean_preprocess(docs, 'nltk.RegexpTokenizer.tokenize()', 'Porter stemmer')
    if( option == 'nltk.RegexpTokenizer.tokenize()' and option1 == 'Lancaster stemmer' ) :
        preprocessed_docs = clean_preprocess(docs, 'nltk.RegexpTokenizer.tokenize()', 'Lancaster stemmer')
    
    if( option != '-' and option1 != '-' ) :
        for i in range( len(docs) ) :
            st.write( "## Text n", i+1 )
            st.write( "### - Original text :" )
            st.write( "#####", docs[i] )
            
            st.write( "### - Text after cleaning and preprocessing : lower case, stopwords and non-words removal, ", option1.split(" ")[0], " stemming :" )
            st.write( "#####", preprocessed_docs[i] )
    
            dicts = create_dicts( preprocessed_docs )
            st.write( "### Dictionary :" )
            show_dicts( dicts[i] )        

elif choose == "Descriptors" :
    st.title( "Descriptors" )
    option = st.selectbox( "Choose the term extraction method : ", ('-', 'split()', 'nltk.RegexpTokenizer.tokenize()') )
    option1 = st.selectbox( "Choose the stemmer : ", ('-', 'Porter stemmer', 'Lancaster stemmer') )
    st.write("")
    st.write("")
    st.write("")   
    
    if( option == 'split()' and option1 == 'Porter stemmer' ) :
        desc = open( "Files/descriptor_split_porter.txt", 'r' )
        
    if( option == 'split()' and option1 == 'Lancaster stemmer' ) :
        desc = open( "Files/descriptor_split_porter.txt", 'r' )
        
    if( option == 'nltk.RegexpTokenizer.tokenize()' and option1 == 'Porter stemmer' ) :
        desc = open( "Files/descriptor_split_porter.txt", 'r' )
        
    if( option == 'nltk.RegexpTokenizer.tokenize()' and option1 == 'Lancaster stemmer' ) :
        desc = open( "Files/descriptor_tokenize_lancaster.txt", 'r' )

    if( option != '-' and option1 != '-' ) :
        st.write( "## ⚬ Descriptor :" )
        for line in desc :
            st.text( line )
            
elif choose == "Information Retrieval" :
    st.title( "Information Retrieval" )
    option = st.selectbox( "Choose the term extraction method : ", ('-', 'split()', 'nltk.RegexpTokenizer.tokenize()') )
    option1 = st.selectbox( "Choose the stemmer : ", ('-', 'Porter stemmer', 'Lancaster stemmer') )
    option2 = st.selectbox( "Choose what do you want to search about : ", ('-', 'Informations about a specific query', 'Matching for a specific query') )
    
    if( option == 'split()' and option1 == 'Porter stemmer' ) :
        inv = open( "Files/tfidf_split_porter.txt", 'r' )
        
    if( option == 'split()' and option1 == 'Lancaster stemmer' ) :
        inv = open( "Files/tfidf_split_lancaster.txt", 'r' )
        
    if( option == 'nltk.RegexpTokenizer.tokenize()' and option1 == 'Porter stemmer' ) :
        inv = open( "Files/tfidf_tokenize_porter.txt", 'r' )
        
    if( option == 'nltk.RegexpTokenizer.tokenize()' and option1 == 'Lancaster stemmer' ) :
        inv = open( "Files/tfidf_split_porter.txt", 'r' )
            
    if( option != '-' and option1 != '-' and option2 == 'Matching for a specific query' ) :
        option3 = st.selectbox( "Choose the matching model : ", ('-', 'Vector space model', 'Probabilistic model (BM25)', 'Boolean model') )
    
        if( option3 == 'Vector space model' ) :
            option4 = st.selectbox( "Choose matching measure : ", ('-', 'scalar product', 'cosine measure', 'jaccard measure') )
            
        elif( option3 == 'Probabilistic model (BM25)' ) :
            col3, col4 = st.columns(2)
            with col3 :
                option5 = st.text_input( "Enter K : " )
            with col4 :
                option6 = st.text_input( "Enter B : " )
                
    st.write("")
    st.write("")
    st.write("")  
    
    if( option != '-' and option1 != '-' and (option2 == 'Informations about a specific query' or ( option2 == 'Matching for a specific query' and option3 == 'Probabilistic model (BM25)' and len(option5)>0 and len(option6)>0 ) or (option2 == 'Matching for a specific query' and option3 == 'Vector space model' and option4 != '-') ) ) :
        col1, col2 = st.columns(2)
        with col1 :
            tokens = st.text_input( "Query 👇" )
        with col2 :
            button_search = st.button( 'search', key='search' )
    if( option != '-' and option1 != '-' and option2 == 'Matching for a specific query' and option3 == 'Boolean model' ) :
        col3, col4, col5 = st.columns(3)
        with col3 :
            nbr_docs = st.text_input( "Number of douments 👇" )
        with col4 :
            tokens = st.text_input( "Query 👇" )
        with col5 :
            button_search = st.button( 'search', key='search' )
        
    if( option != '-' and option1 != '-' and option2 != '-' ) :
        if( option2 == 'Informations about a specific query' ) :
            if( button_search ) :
                tokens = [ token for token in tokens.split() ]
                st.write( "## ⚬ Query's result :" )
                results = query( inv, tokens, option, option1 )
                if not results :
                    st.write( "No results found for this (these) word(s)." )
                else :
                    for line in results :
                        st.text( line )
        elif( option2=='Matching for a specific query' and option3=='Vector space model' and option4 != '-' ) :
            if( button_search ) :
                tokens = [ token for token in tokens.split() ]
                st.write( "## ⚬ Query's result :" )
                results = matching( inv, tokens, option, option1, option4 )
                if not results :
                    st.write( "No results found for this (these) word(s)." )
                else :
                    st.write( "{:<10} {:<10}".format( 'Document', 'Relevance' ) )
                    for line in results :
                        st.text( line )
        elif( option2=='Matching for a specific query' and option3=='Probabilistic model (BM25)' and len(option5)>0 and len(option6)>0 ) :
            if( button_search ) :
                tokens = [ token for token in tokens.split() ]
                st.write( "## ⚬ Query's result :" )
                results = matching_bm25( inv, tokens, option, option1, float(option5), float(option6) )
                if not results :
                    st.write( "No results found for this (these) word(s)." )
                else :
                    st.write( "{:<10} {:<10}".format( 'Document', 'Relevance' ) )
                    for line in results :
                        st.text( line )
        elif( option2=='Matching for a specific query' and option3=='Boolean model' ) :
            if( button_search and len(nbr_docs)>0 ) :
                documents = []
                #for i in range( int(nbr_docs) ) : 
                    #doc_id = 'D'+str(i+1)+'.txt'
                    #doc = open( "Documents/"+doc_id, 'r' ).read()
                    #documents.append( doc )
                dir_path = r'Documents'
                nbr_docss = 0
                for path in os.listdir(dir_path):
                    if os.path.isfile(os.path.join(dir_path, path)):
                        nbr_docss += 1
                        
                for i in range( int(nbr_docss) ) : 
                    doc_id = 'D'+str(i+1)+'.txt'
                    doc = open( "Documents/"+doc_id, 'r' ).read()
                    documents.append( doc )
                documents = clean_preprocess( documents, option, option1 )
                st.write( "## ⚬ Query's result :" )
                results = matching_boolean( documents, tokens, option, option1 )
                if results == None :
                    st.write( "The query is invalid. Check it." )
                elif not results :
                    st.write( "No results found for this (these) word(s)." )
                else :
                    st.write( "{:<10} {:<10}".format( 'Document', 'Relevance' ) )
                    for line in results :
                        st.text( line )

elif choose == "Evaluation" :
    st.title( "Evaluation" )
    option = st.selectbox( "Choose the term extraction method : ", ('-', 'split()', 'nltk.RegexpTokenizer.tokenize()') )
    option1 = st.selectbox( "Choose the stemmer : ", ('-', 'Porter stemmer', 'Lancaster stemmer') )
    
    if( option == 'split()' and option1 == 'Porter stemmer' ) :
        inv = open( "Files/tfidf_split_porter.txt", 'r' )
        
    if( option == 'split()' and option1 == 'Lancaster stemmer' ) :
        inv = open( "Files/tfidf_split_lancaster.txt", 'r' )
        
    if( option == 'nltk.RegexpTokenizer.tokenize()' and option1 == 'Porter stemmer' ) :
        inv = open( "Files/tfidf_tokenize_porter.txt", 'r' )
        
    if( option == 'nltk.RegexpTokenizer.tokenize()' and option1 == 'Lancaster stemmer' ) :
        inv = open( "Files/tfidf_split_porter.txt", 'r' )
            
    if( option != '-' and option1 != '-' ) :
        option3 = st.selectbox( "Choose the matching model : ", ('-', 'Vector space model', 'Probabilistic model (BM25)', 'Boolean model') )
        queries = open( "Queries.txt", 'r' ).readlines()
        nbr_queries = len( queries )
        judgements = open( "Judgements.txt", 'r' ).readlines()
        judgements = [ judg.replace('\t', ' ').replace('      ', ' ').replace('\n', '') for judg in judgements ]
        
        queriesss = [ que.split() for que in queries ]

        indexes = [[] for i in range(len(queriesss))]
        for i in range( len(judgements) ) :
            indexes[ int(judgements[i].split(" ")[0])-1 ].append( i )
    
        if( option3 == 'Vector space model' ) :
            option4 = st.selectbox( "Choose matching measure : ", ('-', 'scalar product', 'cosine measure', 'jaccard measure') )
            
        elif( option3 == 'Probabilistic model (BM25)' ) :
            col3, col4 = st.columns(2)
            with col3 :
                option5 = st.text_input( "Enter K : " )
            with col4 :
                option6 = st.text_input( "Enter B : " )
                
    st.write("")
    st.write("")
    st.write("")  
    
    if( option != '-' and option1 != '-' and ( ( option3 == 'Probabilistic model (BM25)' and len(option5)>0 and len(option6)>0 ) or (option3 == 'Vector space model' and option4 != '-') ) ) :
        col1, col2, col3 = st.columns(3)
        with col1 :
            id_query = st.number_input( "Query in dataset", min_value=1, max_value=nbr_queries, value=1 )
            #tokens = st.text_input( "Query 👇" )
        with col2 :
            tokens = st.text_area( "Query 👇", queries[id_query-1] )
        with col3 :
            button_search = st.button( 'search', key='search' )
            
    if( option != '-' and option1 != '-' and option3 == 'Boolean model' ) :
        col3, col4, col5 = st.columns(3)
        with col3 :
            nbr_docs = st.text_input( "Number of douments 👇" )
        with col4 :
            tokens = st.text_input( "Query 👇" )
        with col5 :
            button_search = st.button( 'search', key='search' )
    
    if( option != '-' and option1 != '-' ) :
        if( option3=='Vector space model' and option4 != '-' ) :
            if( button_search ) :
                tokens = [ token for token in tokens.split() ]
                st.write( "## ⚬ Query's result :" )
                results = matching( inv, tokens, option, option1, option4 )
                if not results :
                    st.write( "No results found for this (these) word(s)." )
                else :
                    st.write( "{:<10} {:<10}".format( 'Document', 'Relevance' ) )
                    for line in results :
                        st.text( line )
                    prec = precision( results, [judgements[ind] for ind in indexes[id_query-1]] )
                    prec5 = precision( results, [judgements[ind] for ind in indexes[id_query-1]], 5 )
                    prec10 = precision( results, [judgements[ind] for ind in indexes[id_query-1]], 10 )
                    rec = recall( results, [judgements[ind] for ind in indexes[id_query-1]] )
                    fsc = fscore( prec, rec )
                    inter_curve = interpolated_curve(results, [judgements[ind] for ind in indexes[id_query-1]])
                    st.write( "## ⚬ Results' evaluation :" )
                    col_p, col_p5, col_p10, col_r, col_f = st.columns( 5 )
                    with col_p :
                        st.write( '#### Precision: ', prec )
                    with col_p5 :
                        st.write( '#### Precision@5: ', prec5 )
                    with col_p10 :
                        st.write( '#### Precision@10: ', prec10 )
                    with col_r :
                        st.write( '#### Recall: ', rec )
                    with col_f :
                        st.write( '#### F-score: ', fsc )
                    st.write( '#### Interpolated curve recall/precision' )
                    df = pd.DataFrame( inter_curve, columns=['Recall', 'Precision'] )
                    df.set_index( 'Recall', inplace=True )
                    st.line_chart( df, y='Precision', color = '#228B22' )
                
        elif( option3=='Probabilistic model (BM25)' and len(option5)>0 and len(option6)>0 ) :
            if( button_search ) :
                tokens = [ token for token in tokens.split() ]
                st.write( "## ⚬ Query's result :" )
                results = matching_bm25( inv, tokens, option, option1, float(option5), float(option6) )
                if not results :
                    st.write( "No results found for this (these) word(s)." )
                else :
                    st.write( "{:<10} {:<10}".format( 'Document', 'Relevance' ) )
                    for line in results :
                        st.text( line )
                    prec = precision( results, [judgements[ind] for ind in indexes[id_query-1]] )
                    prec5 = precision( results, [judgements[ind] for ind in indexes[id_query-1]], 5 )
                    prec10 = precision( results, [judgements[ind] for ind in indexes[id_query-1]], 10 )
                    rec = recall( results, [judgements[ind] for ind in indexes[id_query-1]] )
                    fsc = fscore( prec, rec )
                    inter_curve = interpolated_curve(results, [judgements[ind] for ind in indexes[id_query-1]])
                    st.write( "## ⚬ Results' evaluation :" )
                    col_p, col_p5, col_p10, col_r, col_f = st.columns( 5 )
                    with col_p :
                        st.write( '#### Precision: ', prec )
                    with col_p5 :
                        st.write( '#### Precision@5: ', prec5 )
                    with col_p10 :
                        st.write( '#### Precision@10: ', prec10 )
                    with col_r :
                        st.write( '#### Recall: ', rec )
                    with col_f :
                        st.write( '#### F-score: ', fsc )
                    st.write( '#### Interpolated curve recall/precision' )
                    df = pd.DataFrame( inter_curve, columns=['Recall', 'Precision'] )
                    df.set_index( 'Recall', inplace=True )
                    st.line_chart( df, y='Precision', color = '#228B22' )
                    
        elif( option3=='Boolean model' ) :
            if( button_search ) :
                documents = []
                for i in range( int(nbr_docs) ) : 
                    doc_id = 'D'+str(i+1)+'.txt'
                    doc = open( "Documents/"+doc_id, 'r' ).read()
                    documents.append( doc )
                documents = clean_preprocess( documents, option, option1 )
                st.write( "## ⚬ Query's result :" )
                results = matching_boolean( documents, tokens, option, option1 )
                if results == None :
                    st.write( "The query is invalid. Check it." )
                elif not results :
                    st.write( "No results found for this (these) word(s)." )
                else :
                    st.write( "{:<10} {:<10}".format( 'Document', 'Relevance' ) )
                    for line in results :
                        st.text( line )
                        
elif choose == "TF-IDF" :
    st.title( "TF-IDF" )
    docs = read_data()
    
    option = st.selectbox( "Choose the term extraction method : ", ('-', 'split()', 'nltk.RegexpTokenizer.tokenize()') )
    
    option1 = st.selectbox( "Choose the stemmer : ", ('-', 'Porter stemmer', 'Lancaster stemmer') )
    st.write("")
    st.write("")
    st.write("")

    if( option == 'split()' and option1 == 'Porter stemmer' ) :
        preprocessed_docs = clean_preprocess(docs, 'split()', 'Porter stemmer')
    if( option == 'split()' and option1 == 'Lancaster stemmer' ) :
        preprocessed_docs = clean_preprocess(docs, 'split()', 'Lancaster stemmer')
    if( option == 'nltk.RegexpTokenizer.tokenize()' and option1 == 'Porter stemmer' ) :
        preprocessed_docs = clean_preprocess(docs, 'nltk.RegexpTokenizer.tokenize()', 'Porter stemmer')
    if( option == 'nltk.RegexpTokenizer.tokenize()' and option1 == 'Lancaster stemmer' ) :
        preprocessed_docs = clean_preprocess(docs, 'nltk.RegexpTokenizer.tokenize()', 'Lancaster stemmer')

    if( option != '-' and option1 != '-' ) :
        st.write( "## ⚬ Original texts :" )
        for i in range( len(docs) ) :
            st.write( docs[i] )
    
        st.write( "## ⚬ Preprocessed texts :" )
        for i in range( len(docs) ) :
            st.write( preprocessed_docs[i] )
    
        dict = create_dict( preprocessed_docs )
        st.write( "## Dictionary :" )
        show_dict( dict )
        
elif choose == "Contact" :
    st.write( "### We are always happy to hear from you!" )
    st.write( "### Send us an email and tell us how we can help you via this email: tfidf@gmail.com" )
    
elif choose == "Help" :
    st.write( "### - Dictionary : A Python data structure storing informations per index/value" )
    st.write( "### - TF-IDF : Term Frequency–Inverse Document Frequency" )
    st.write( "### - Vector space model : is an algebraic model for representing items as vectors such that the distance between vectors represents the relevance between the documents" )
    st.write( "### - Probailistic BM25 model (BM is an abbreviation of best matching) : is a ranking function used by search engines to estimate the relevance of documents to a given search query. It is based on the probabilistic retrieval framework developed by Stephen E. Robertson, Karen Spärck Jones, and others." )
    st.write( "### - Boolean model : is based on Boolean logic and classical set theory in that both the documents to be searched and the user's query are conceived as sets of terms" )
