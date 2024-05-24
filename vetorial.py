from genericpath import exists
from itertools import count
import os
import pdb
from string import punctuation
import math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from yaml import DocumentEndEvent



current_directory = os.getcwd()
path = current_directory +"/VetorialSearch/Texts/"
os.chdir(path)



#funcao que le o documento recebido
def readDoc (file):
    with open(file) as outputfile:
        text = outputfile.read()
        return text

#funcao para tokenizar o documento
def tokenize(file):
    return file.split()

#funcao que remove stopwords do documento
def remove_stopwords(document):
    tokens_filtered= [word for word in document if not word in stopwords ]
    return tokens_filtered

#funcao que remove pontuacao do documento
def remove_punctuation(document):
    output = []
    
    for word in document:
        for letter in word:
            if letter in punctuation:
                word = word.replace(letter,"")   
        output.append(word)
    return output

#funcao filtro que realiza a remocao das stopwords e pontuacoes do documento e ja o retorna todo em letra minuscula
def filter(document):
    
    output = []
    
    for word in document:
        output.append(word.lower())
      
    filter = remove_punctuation(output)
    filter = remove_stopwords(filter)
    
    return filter
    
#funcao que cria o dicionario e organiza os indices invertidos dentro do mesmo
def makeDictionary(document):
    
    terms = []
    Dict = {}
    
    #separo todas as palavras de todos os documentos em um vetor
    i = 1 
    for doc in document:
        for word in doc:
            if word in Dict:
                if i not in Dict[word]:
                    Dict[word].append(i)
            else:
                Dict.setdefault(word, [])
                Dict[word].append(i)
        i+=1        
    
    print(Dict)
    return(Dict)

def makeMatrix(document):
    
    terms = []
    
    #separo todas as palavras individuais do documento em um novo vetor
    for doc in document:
        for word in doc:
            if word not in terms:
                terms.append(word)
    
    #crio uma nova matriz de zeros do tipo object
    term_document = np.zeros( (len(terms)+1 , len(document)+1), dtype=np.object_)
    
    #i controla as linhas da matriz, enquanto y controla as colunas da matriz para preencher a primeira linha com os nomes dos documentos
    i = 1
    y = 0
    for word in terms:
        while y < 6:
            if y == 0:
                term_document[0][y] = 'Documents: '
                y+=1
            term_document[0][y] = "Doc "+ str(y)
            y+=1
        term_document[i][0] = word
        
        i+=1
        
     #for que realiza a filtragem termo-documento de cada palavra dentro de cada documento
    y=1
    for word in document:
        i = 1
        for letter in word:
            x = 1
            while x < len(terms)+1:
                if letter == term_document[x][0]:
                    term_document[x][y] += 1
                x+=1
            i +=1    
        y+=1
        
        
    return term_document

def tfIdf(document):
    
    vec=TfidfVectorizer(stop_words= stopwords)
    matrix=vec.fit_transform(document)
    print("Feature Names \n",vec.get_feature_names_out())
    print("Sparse Matrix \n",matrix.shape,"\n",matrix.toarray())

    
#função que realiza a busca vetorial
def vetorialSearch(document, query):
    tf_idf = tfIdf(document)
    Dict = makeDictionary(document)
    queryVector = []
    documentVector = []
    wordFrequency = {}
    
    
    for word in query:
  
        if(word not in wordFrequency):
            wordFrequency[word] = 1
        else:
            wordFrequency[word] += 1
            
    for term in Dict:
        if term not in wordFrequency:
            queryVector.append(0)
        else:
            tf_idf_query = (1 + math.log(wordFrequency[term], 10)) * math.log((len(document) / len(Dict[word])), 10)
            tf_idf_query = np.around(tf_idf_query, 2)
            queryVector.append(tf_idf_query)
            
    auxColumn = 1
    auxVector = []
    for word in document:
        auxVector = []
        x = 0
        auxLine = 1
        while x < len(Dict):
            auxVector.append(tf_idf[auxLine][auxColumn])
            auxLine += 1
            x+=1
        documentVector.append(auxVector)
        auxColumn += 1  
    
    result = []    
    i = 0
    for doc in documentVector:
        normDoc = [doc[i]**2 for i in range(len(doc))]
        normDoc = np.sum(normDoc, dtype = np.float32)
        normQuery = [queryVector[i] for i in range(len(queryVector))]
        normQuery = np.sum(normQuery, dtype = np.float32)
        norm = math.sqrt(normDoc) * math.sqrt(normQuery)
        result.append(np.dot(doc, queryVector) / norm) 
        i += 1
        
    aux = 1
    print("Rankeamento dos documentos baseado na busca:")
    for i in range(len(result)):    
        print("Doc",aux,": " ,np.around(result[i], 2))    
        aux +=1
            

#funcao que le diversos documentos de um determinado diretorio e ja as tokeniza totalmente (removendo stopwords, pontuacoes e deixando em letra minuscula)
def readCollection():
    i = 0
    docs = []
    
    for file in sorted(os.listdir()):
        if file.endswith('.txt'):
            file_path =f"{path}{file}"
            output = readDoc(file_path)
        docs.append(output) 
        i+=1
    return docs            
    
#optei por deixar o documento das stopwords e pontuacao global pois assim poderia acessa-lo de qualquer funcao
stopwords = readDoc(current_directory + "/Filters/stopwords_ptbr.txt")
stopwords = tokenize(stopwords)
    
punctuation = readDoc(current_directory +"/Filters/punctuation.txt")
#punctuation = tokenize(punctuation)

def main():
    
    #newFile recebe uma lista dos documentos do diretorio ja totalmente tokenizados (sem stopword, pontuacao e tudo em minusculo)
    newFile = readCollection()
    
    x=1
    for doc in newFile:
        print("Documento", x, ":", doc)
        x+=1
    print("\n")
    
    
    term = []
    term = input()
    term = tokenize(term)
    term = filter(term)
    
    tfIdf(newFile)
    #vetorialSearch(newFile, term)


if __name__ == "__main__":
    main()
