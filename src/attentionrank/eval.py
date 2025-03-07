import csv
import numpy as np
import os
import time

from .attentions import  clean_folder
from .utils import write_list_file,get_files_from_path,get_files_ids




# Variables globales (valores iniciales)
DATASET_NAME = "default_dataset"
ROOT_FOLDER = os.path.join(".", DATASET_NAME)
DOCS_FOLDER = os.path.join(ROOT_FOLDER, "docsutf8")
PROCESSED_FOLDER = os.path.join(ROOT_FOLDER, f"processed_{DATASET_NAME}")

def update_paths_eval(dataset_name):
    """Actualiza las rutas globales cuando cambia el dataset"""
    global DATASET_NAME, ROOT_FOLDER, DOCS_FOLDER, PROCESSED_FOLDER
    DATASET_NAME = dataset_name
    ROOT_FOLDER = os.path.join(".", DATASET_NAME)
    DOCS_FOLDER = os.path.join(ROOT_FOLDER, "docsutf8")
    PROCESSED_FOLDER = os.path.join(ROOT_FOLDER, f"processed_{DATASET_NAME}")

def read_term_list_file(filepath):
    lst = []
    with open(filepath, "r", encoding='utf-8') as f:
        for line in f:
            k = line.replace("\n", '').replace("  ", " ").replace("  ", " ")
            k = k.strip()
            lst.append(k.lower())
    return lst


def f1(a, b):
    return a * b * 2 / (a + b)


def mean_f_p_r(actual, predicted, best=10, pr_plot=False):
    list_f1 = []
    list_p = []
    list_r = []
    for r in range(len(actual)):
        y_actual = actual[r]
        y_predicted = predicted[r][:best]
        y_score = 0
        for p, prediction in enumerate(y_predicted):
            if prediction in y_actual and prediction not in y_predicted[:p]:
                y_score += 1
        if not y_predicted:
            y_p = 0
            y_r = 0
        else:
            y_p = y_score / len(y_predicted)
            y_r = y_score / len(y_actual)
        if y_p != 0 and y_r != 0:
            y_f1 = 2 * (y_p * y_r / (y_p + y_r))
        else:
            y_f1 = 0
        list_f1.append(y_f1)
        list_p.append(y_p)
        list_r.append(y_r)
    if pr_plot:
        return list_f1, list_p, list_r
    else:
        return np.mean(list_f1), np.mean(list_p), np.mean(list_r)




def generate_results( language, f1_top=10):
    """
        Normed Integration and evaluation
    """
    # f1_top = 10

    #datasetpath = './' + datasetname + '/'

    datasetpath = ROOT_FOLDER
    # my_key = os.path.join(label_path, file + ".key")

    if language =='es':
        ## poner el español
        #stopwords_file = './src/attentionrank/UGIR_stopwords_es.txt'
        stopwords_file = os.path.join(".","src", "attentionrank","UGIR_stopwords_es.txt")
        mystopwords = read_term_list_file(stopwords_file)
    else:

        stopwords_file = os.path.join(".","src", "attentionrank","UGIR_stopwords.txt") #stopwords_file = './src/attentionrank/.txt'
        mystopwords = read_term_list_file(stopwords_file)

    dataset = DATASET_NAME
    #text_path = os.path.join(datasetpath,"docsutf8") # datasetpath + '/docsutf8/'
    #output_path = os.path.join(datasetpath,'processed_' + dataset)  #datasetpath + '/processed_' + dataset + '/'

    accumulated_self_attn_path = os.path.join(PROCESSED_FOLDER,'candidate_attn_paired')  # output_path + 'candidate_attn_paired/'

    #save_path = './' + dataset + '/res' + str(f1_top) + '/'
    save_path = os.path.join(".", dataset, 'res' + str(f1_top))


    if os.path.exists(save_path):
        clean_folder(save_path + 'sentence_paired_text')

    else:
        os.makedirs(save_path)

    files = get_files_from_path(DOCS_FOLDER)
    files = get_files_ids(files)
    print('Files to process:', len(files))

    # load  df
    df_dict = {}
    df_path = os.path.join( PROCESSED_FOLDER , 'df_dict')
    with open(os.path.join(df_path , dataset + "_candidate_df.csv"), newline='',encoding="utf-8") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            k = row[0].lower()
            v = float(row[1])
            df_dict[k] = v

    predicted = []
    actual = []

    start_time = time.time()

    for n, file in enumerate(files):
        #print('file', file, '\n')
        pred_single = []
        # load cross attn dict
        cross_attn_dict_first = {}
        cross_attn_dict_path = os.path.join(PROCESSED_FOLDER , 'candidate_cross_attn_value')
        tail = "_candidate_cross_attn_value.csv"
        # print(cross_attn_dict_path + file + tail)
        with open( os.path.join(cross_attn_dict_path , file + tail), newline='',encoding="utf-8") as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            v=0
            k=''

            for row in spamreader:
                k = row[0].lower()
                if k.find('.') == -1:
                    try:
                    # print(df_dict)
                        df = df_dict[k]
                        if df_dict[k]:  # < 44:
                            v = float(row[1])
                            cross_attn_dict_first[k] = v
                    except:

                        print('next',k)
            ## toberemoved
            if len(cross_attn_dict_first.items()) == 0:

                cross_attn_dict_first[k] = v

        #print(df_dict)
        # print(cross_attn_dict_first)

        try:

            cross_attn_dict = {}
            for k, v in cross_attn_dict_first.items():
                if k[-1] == "s" and k[:-1] in cross_attn_dict:
                    cross_attn_dict[k[:-1]] = max(v, cross_attn_dict[k[:-1]])
                elif k + 's' in cross_attn_dict:
                    cross_attn_dict[k] = max(v, cross_attn_dict[k + 's'])
                    cross_attn_dict.pop(k + 's')
                else:
                    cross_attn_dict[k] = v

            # norm cross attn dict
            # print('-------')
            # print(cross_attn_dict)
            a0 = min(cross_attn_dict.values())
            b0 = max(cross_attn_dict.values())
            for k, v in cross_attn_dict.items():
                cross_attn_dict[k] = (v - a0) / (b0 - a0)

            # load accumulated self attn ranking
            accumulated_self_attn_dict_first = {}
            tail0 = "_attn_paired.csv"

            with open( os.path.join(accumulated_self_attn_path , file + tail0), newline='',encoding="utf-8") as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',')
                for row in spamreader:
                    k = row[0].lower()
                    if k.find('.') == -1:
                        v = float(row[1])  # /len(k.split(' '))
                        accumulated_self_attn_dict_first[k] = v

            accumulated_self_attn_dict = {}
            #print(accumulated_self_attn_dict_first)
            for k, v in accumulated_self_attn_dict_first.items():
                if k[-1] == "s" and k[:-1] in accumulated_self_attn_dict:
                    accumulated_self_attn_dict[k[:-1]] = v + accumulated_self_attn_dict[k[:-1]]
                elif k + 's' in accumulated_self_attn_dict:
                    accumulated_self_attn_dict[k] = v + accumulated_self_attn_dict[k + 's']
                    accumulated_self_attn_dict.pop(k + 's')
                else:
                    accumulated_self_attn_dict[k] = v
            # print('--->')
            # print(accumulated_self_attn_dict)

            # norm attn-candidate dict
            t = 8
            ranking_dict = {}
            #print(accumulated_self_attn_dict)
            a1 = min(accumulated_self_attn_dict.values())
            b1 = max(accumulated_self_attn_dict.values())
            for k, v in accumulated_self_attn_dict.items():
                # print('term', k)
                if k in cross_attn_dict.keys() and k.split(' ')[0] not in mystopwords:
                    # print('passterm', k)
                    accumulated_self_attn_dict[k] = (v - a1) / (b1 - a1)
                    ranking_dict[k] = accumulated_self_attn_dict[k] * (t) / 10 + cross_attn_dict[k] * (10 - t) / 10
                #else:
                 #   print('Not passed ' + k, k.split(' ')[0] not in mystopwords)

            f1_k = 0
            # print('Prediction:')

            for k, v in sorted(ranking_dict.items(), key=lambda item: item[1], reverse=True):
                if f1_k < f1_top:
                    # print(k, v)
                    pred_single.append(k)
                    f1_k += 1
        except:
            print('Error fatale')

        # load keys



        #label_path = os.path.join(".", dataset, "keys")
        #my_key = os.path.join(label_path, file + ".key")


        # print('\n Truth keys:')
        #actual_single = read_term_list_file(my_key)
        #actual_single = list(set(actual_single))


        '''
        actual.append(actual_single)
        predicted.append(pred_single)
        
        '''

        # save predicted single
        savefile= file+'.key'
        write_list_file(os.path.join(save_path,savefile),pred_single)




    '''
    # print('Len actual', len(actual), len(predicted))
    mean_f1, mean_p, mean_r = mean_f_p_r(actual, predicted, f1_top)
    straight_f1 = f1(mean_p, mean_r)
    print('Precission, recall, f1, mean_f1')
    print(mean_p, mean_r, straight_f1, mean_f1)
    '''


def evaluate_results( f1_top):
    datasetpath = ROOT_FOLDER

    keys_path = os.path.join(datasetpath , 'keys')
    res_path = os.path.join(datasetpath ,'res' + str(f1_top) )
    predicted = []
    actual = []

    keyfiles = os.listdir(keys_path)
    resfiles = os.listdir(res_path)
    if len(keyfiles) != len(resfiles):
        print('FATAL ERROR',keyfiles,resfiles)
        return

    print('Files to process:', len(keyfiles))

    for keyf in keyfiles:
        key_single = read_term_list_file(keys_path+keyf)
        key_single = list(set(key_single))
        actual.append(key_single)

        pred_single = read_term_list_file(res_path+keyf)
        pred_single = list(set(pred_single))
        predicted.append(pred_single)

    mean_f1, mean_p, mean_r = mean_f_p_r(actual, predicted, f1_top)
    straight_f1 = f1(mean_p, mean_r)
    print('Precission, recall, f1, mean_f1')
    print(mean_p, mean_r, straight_f1, mean_f1)
