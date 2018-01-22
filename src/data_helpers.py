# -*- coding: utf-8 -*-
# baseado no original em: https://github.com/dennybritz/cnn-text-classification-tf/blob/master/data_helpers.py

import re
import codecs
import numpy as np

def clean_str(string):
    """
    Tokenização/limpeza do dataset baseado no trabalho do yoonkin
    em https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    # tratamento dos caracteres especiais em português
    string = re.sub(r"ç", "c", string)
    string = re.sub(r"ã", "a", string)
    string = re.sub(r"á", "a", string)
    string = re.sub(r"à", "a", string)
    string = re.sub(r"â", "a", string)
    string = re.sub(r"é", "e", string)
    string = re.sub(r"ê", "e", string)
    string = re.sub(r"í", "i", string)
    string = re.sub(r"õ", "o", string)
    string = re.sub(r"ó", "o", string)
    string = re.sub(r"ô", "o", string)
    string = re.sub(r"ú", "u", string)

    # substitui tudo que não for esses caracteres abaixo por espaço
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)    

    # parte do treinamento original do yoonkin para inglês    
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)

    # tratamento da pontuação
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\"", " \" ", string)
    string = re.sub(r"\?", " \? ", string)
    
    # tratamento dos espaços duplicados
    string = re.sub(r"\s{2,}", " ", string)
    
    return string.strip().lower()

def compact_str(string):
    # Detecção: oi, oooi, ooooieeeee, olaaa, olar
    string = re.sub(r"^(o+i+e*|o+l+a+r*)", "oie", string)
    # Detecção: hahahahaha, heheheh, kkkk, rsrs
    string = re.sub(r"([aei]?(h[aei]){2,}h?|k{3,})|(rs){2,}", "hahaha", string)
    # Detecção: nooooossa --> nossa (mais de 2 caracteres consecutivos viram 1 só)
    # Resultado aparentemente insatisfatório.
    #string = re.sub(r"(.)\1{2,}", "$1", string)
    return string

def gen_label(size, index):
    """
    Gera uma lista de tamanho 'size' toda de zeros, com excessão de colocar um '1' na posição index
    """
    aux = [0] * size
    aux[index] = 1
    return aux
    
def load_data_and_labels(data_folder):
    """
    Carrega os dados de arquivos de dados na pasta, faz o split dos textos em palavras e gera os labels
    Baseado no original em: https://github.com/dennybritz/cnn-text-classification-tf/blob/master/data_helpers.py
    Retorna as frases splitadas e os labels
    """

    import os
    files = [file for file in os.listdir(data_folder) if file.lower().endswith(".txt")]
    num_files = len(files)

    x_text = []
    y = []

    for index_file, file in enumerate(files):
        text_examples = list(codecs.open(os.path.join(data_folder, file), "r", "utf-8").readlines())
        text_examples = [compact_str(clean_str(s)).strip() for s in text_examples]
        text_labels = [gen_label(num_files, index_file) for _ in text_examples]
        x_text += text_examples
        y += text_labels
        
    y = np.array(y)

    return [files, x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]