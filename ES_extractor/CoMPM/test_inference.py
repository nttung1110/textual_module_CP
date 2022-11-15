# -*- coding: utf-8 -*-
from tqdm import tqdm
import os
import random
import torch
import json
import torch.nn as nn
import pdb
import os.path as osp

from transformers import RobertaTokenizer
from DARPA_infer_dataset import *
from model import ERC_model
from utils import make_batch_roberta, make_batch_bert, make_batch_gpt

from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
import pdb
import argparse, logging
from sklearn.metrics import precision_recall_fscore_support
    
def format_darpa_emotion_prediction(json_result, data_type):
    wrt_conv = ""
    wrt_emotion_cat = ""

    path_write = '/home/nttung/research/Monash_CCU/mini_eval/text_module/darpa_emotion_dataset'
    
    name_data_conv = 'darpaEmo_' + data_type + '.txt'
    name_data_emo = 'darpaEmo_emotion' + data_type + '.txt'

    path_data_conv = osp.join(path_write, data_type, name_data_conv)
    path_data_emo = osp.join(path_write, data_type, name_data_emo)

    for file_name in json_result:
        conv = json_result[file_name]

        tmp_conv_str = []
        tmp_emo_str = []
        for pair in conv:
            [utt, emo] = pair
            
            tmp_conv_str.append(utt)
            tmp_emo_str.append(str(emo))
        

        tmp_conv_str.append(file_name) # make sure to keep track of file name for the conv
        tmp_emo_str.append(file_name)

        tmp_conv_str = '__eou__'.join(tmp_conv_str)
        tmp_emo_str = ' '.join(tmp_emo_str)

        wrt_conv += tmp_conv_str + '\n'
        wrt_emotion_cat += tmp_emo_str + '\n'

    
    with open(path_data_conv, 'w') as fp:
        fp.write(wrt_conv)

    with open(path_data_emo, 'w') as fp:
        fp.write(wrt_emotion_cat)



## finetune RoBETa-large
def main():    
    initial = args.initial
    model_type = args.pretrained
    if 'roberta' in model_type:
        make_batch = make_batch_roberta
    elif model_type == 'bert-large-uncased':
        make_batch = make_batch_bert
    else:
        make_batch = make_batch_gpt      
    freeze = args.freeze
    if freeze:
        freeze_type = 'freeze'
    else:
        freeze_type = 'no_freeze'    
    sample = args.sample
    if 'gpt2' in model_type:
        last = True
    else:
        last = False
    
    """Dataset Loading"""
    dataset_list = ['dailydialog']
    DATA_loader_list = [Darpa_raw_loader]
    dataclass = args.cls
    dataType = 'multi'
    
    """Log"""
    log_path = os.path.join('test.log')
    fileHandler = logging.FileHandler(log_path)
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)    
    logger.setLevel(level=logging.DEBUG)    
    
    """Model Loading"""
    for dataset, DATA_loader in zip(dataset_list, DATA_loader_list):
        if dataset == 'MELD':
            data_path = os.path.join('dataset', dataset, dataType)
        else:
            data_path = os.path.join('dataset', dataset)
        save_path = os.path.join('ckpt/models/'+dataset+'_models', model_type, initial, freeze_type, dataclass, str(sample))
        print("###Save Path### ", save_path)
    
        # dev_path = os.path.join(data_path, dataset+'_dev.txt')
        # test_path = os.path.join(data_path, dataset+'_test.txt')

        # darpa
        inference_path = '/home/nttung/research/Monash_CCU/mini_eval/text_module/src/dataset/dd_format_darpa_dev.txt'

        # dev_dataset = DATA_loader(dev_path, dataclass)
        # dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=make_batch)        

        test_dataset = DATA_loader(inference_path, dataclass)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=make_batch)
        

        print('Data: ', dataset, '!!!')
        # clsNum = len(test_dataset.labelList)        
        clsNum = 7 # 7 emotion categories in daily dialog
        model = ERC_model(model_type, clsNum, last, freeze, initial)
        modelfile = os.path.join(save_path, 'model.bin')
        model.load_state_dict(torch.load(modelfile))
        model = model.cuda()    
        model.eval()           

        """Dev & Test evaluation"""
        logger.info('####### ' + dataset + ' #######')
        if dataset == 'dailydialog': # micro & macro
            # dev_acc, dev_pred_list, dev_label_list = _CalACC(model, dev_dataloader)
            # dev_pre_macro, dev_rec_macro, dev_fbeta_macro, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='macro')
            # dev_pre_micro, dev_rec_micro, dev_fbeta_micro, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, labels=[0,1,2,3,5,6], average='micro') # neutral x

            test_acc, test_pred_list, test_label_list = _CalACC(model, test_dataloader)
            test_pre_macro, test_rec_macro, test_fbeta_macro, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='macro')
            test_pre_micro, test_rec_micro, test_fbeta_micro, _ = precision_recall_fscore_support(test_label_list, test_pred_list, labels=[0,1,2,3,5,6], average='micro') # neutral x
        else: # weight
            dev_acc, dev_pred_list, dev_label_list = _CalACC(model, dev_dataloader)
            dev_pre, dev_rec, dev_fbeta, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='weighted')

            test_acc, test_pred_list, test_label_list = _CalACC(model, test_dataloader)
            test_pre, test_rec, test_fbeta, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='weighted')

        if dataset == 'dailydialog': # micro & macro
            logger.info('Fscore ## accuracy: {}, dev-macro: {}, dev-micro: {}, test-macro: {}, test-micro: {}'\
                        .format(dev_acc*100, dev_fbeta_macro, dev_fbeta_micro, test_fbeta_macro, test_fbeta_micro))
        else:
            logger.info('Fscore ## accuracy: {}, dev-fscore: {}, test-fscore: {}'.format(test_acc*100, dev_fbeta, test_fbeta))
        logger.info('')
    
def _CalACC(model, dataloader):
    model.eval()
    correct = 0
    label_list = []
    pred_list = []
    
    distinct_pred_labels = []
    list_pos_emotion = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    json_result = {}

    # label arragne
    with torch.no_grad():
        for i_batch, data in enumerate(tqdm(dataloader)):
            """Prediction"""
            batch_input_tokens, batch_labels, batch_speaker_tokens, batch_utterance, batch_filename = data
            batch_input_tokens, batch_labels = batch_input_tokens.cuda(), batch_labels.cuda()
            
            pred_logits = model(batch_input_tokens, batch_speaker_tokens) # (1, clsNum)
            
            """Calculation"""    
            pred_label = pred_logits.argmax(1).item()

            if pred_label not in distinct_pred_labels:
                distinct_pred_labels.append(pred_label)

            true_label = batch_labels.item()
            
            pred_list.append(pred_label)
            label_list.append(true_label)
            if pred_label == true_label:
                correct += 1

            if batch_filename[0] not in json_result:
                json_result[batch_filename[0]] = []

            pair = [batch_utterance[0], pred_label]
            json_result[batch_filename[0]].append(pair)


        acc = None # dump, no meaning here
        format_darpa_emotion_prediction(json_result, dataloader.dataset.type)
        # path_write = 'predicted_emotion_darpa_'+dataloader.dataset.type+'.json'
        # with open(path_write, "w") as fp:
        #     json.dump(json_result, fp, indent=4)
    return acc, pred_list, label_list

if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    """Parameters"""
    parser  = argparse.ArgumentParser(description = "Emotion Classifier" )    
    parser.add_argument( "--pretrained", help = 'roberta-large', default = 'roberta-large')
    parser.add_argument('-dya', '--dyadic', action='store_true', help='dyadic conversation')
    parser.add_argument( "--cls", help = 'emotion or sentiment', default = 'emotion')
    parser.add_argument( "--initial", help = 'pretrained or scratch', default = 'pretrained')
    parser.add_argument('-fr', '--freeze', action='store_true', help='freezing PM')
    parser.add_argument( "--sample", type=float, help = "sampling trainign dataset", default = 1.0) # 
        
    args = parser.parse_args()
    
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    
    main()
    