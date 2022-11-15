import os
import numpy as np
import pdb
import copy
import torch

from dotmap import DotMap
from tqdm import tqdm

from CoMPM.utils import make_batch_roberta as make_batch
from CoMPM.DARPA_infer_dataset import *
from CoMPM.model import ERC_model
from googletrans import Translator
from transformers import RobertaTokenizer


def DARPA_text_data_read(path, translator):
    '''
        Processing DARPA textual dataset to separate utterances by speakers
            + Input: Only accept txt file at the moment
            + Output: Return dictionary indexing to specific speaker along with that speaker list of utterances
    '''
    
    with open(path, "r") as fp:
        text_data = fp.read()

    text_data = text_data.split("\n")
    if text_data[-1] == "":
        text_data = text_data[:-1] # ignore last empty segment    

    # iterate and separate each segment
    s1 = [] # track utterance of speaker1 through conversation
    s2 = [] # track utterance of speaker2 through conversation

    # translate all text
    for idx, data in enumerate(text_data):
        # translate
        text_data[idx] = translator.translate(text_data[idx]).text

    # preprocess by eliminating auto-reply utterance

    # whoever is the first speaker will be assigned to s1
    first_speaker = None
    conversation_list = []

    running_offset = 0
    start_offset_s1 = []
    start_offset_s2 = []


    for segment in text_data:
        seg_id = int(segment.split(':')[0])
        seg_text = segment.split(':')[1]
        conversation_list.append(seg_text)

        if first_speaker == None:
            first_speaker = seg_id
            start_offset_s1.append(0)
            s1.append(seg_text)

            running_offset += len(seg_text)
            continue

        if seg_id == first_speaker:
            s1.append(seg_text)
            start_offset_s1.append(running_offset)
        else:
            s2.append(seg_text)
            start_offset_s2.append(running_offset)

        running_offset += len(seg_text)

    # also return start offset in character of every utterance
    
    return {'s1': s1, 's2': s2}, text_data, {'s1': start_offset_s1, 's2': start_offset_s2}


class TextualESCoMPM():
    def __init__(self, args):
        self.args = args
        self.init_model()

    
    def init_model(self):
        initial = self.args.initial
        model_type = self.args.pretrained
        freeze = self.args.freeze
        if freeze:
            freeze_type = 'freeze'
        else:
            freeze_type = 'no_freeze' 

        sample = self.args.sample
        if 'gpt2' in model_type:
            last = True
        else:
            last = False


        model_path = self.args.pretrained_model_ckpt
        clsNum = 7 # 7 emotion categories in daily dialog
        self.model = ERC_model(model_type, clsNum, last, freeze, initial)
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.cuda()    
        self.model.eval() 
        

    def extract_ES_emotion_per_track(self, conversation_list, dict_speaker_utt):
        
        print("==========Extracting ES from utterance===========")
        # init dataloader for that conversation_list
        dataset = Darpa_raw_loader(conversation_list, 'emotion')


        test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=make_batch)

        list_feat_s1 = []
        list_feat_s2 = []

        count = 0
        with torch.no_grad():
            for i_batch, data in enumerate(tqdm(test_dataloader)):
                """Prediction"""
                batch_input_tokens, _, batch_speaker_tokens, batch_utterance = data
                batch_input_tokens = batch_input_tokens.cuda()

                pred_logits = self.model(batch_input_tokens, batch_speaker_tokens) # (1, clsNum)

                feat_emotion = torch.nn.Softmax(dim=1)(pred_logits)

                # check whether this feat belongs to which speaker
                cur_utterance = batch_utterance[0]

                if cur_utterance in dict_speaker_utt['s1']:
                    list_feat_s1.append(feat_emotion[0].tolist())
                else:
                    list_feat_s2.append(feat_emotion[0].tolist())
                
                count += 1

        feat_s1 = np.array(list_feat_s1)
        feat_s2 = np.array(list_feat_s2)

        all_es_text_feat_tracks = [feat_s1, feat_s2]
        return all_es_text_feat_tracks

if __name__ == "__main__":
    # test args 
    args = DotMap()
    args.pretrained = 'roberta-large'
    args.dyadic = False
    args.cls = 'emotion'
    args.initial = 'pretrained'
    args.freeze = False
    args.sample = 1.0
    args.pretrained_model_ckpt = '/home/nttung/research/Monash_CCU/mini_eval/text_module/src_multistage_approach/ES_extractor/CoMPM/ckpt/models/dailydialog_models/roberta-large/pretrained/no_freeze/emotion/1.0/model.bin'

    path_csv_test = '/home/nttung/research/Monash_CCU/mini_eval/text_data/en_train/M01000G9C_en.txt'
    utterance_by_speakers, conversation_list = DARPA_text_data_read(path_csv_test)
    # textES = TextualES(args)


    torch.cuda.empty_cache()
    textES = TextualESCoMPM(args)

    textES.extract_ES_emotion_per_track(conversation_list, utterance_by_speakers)
