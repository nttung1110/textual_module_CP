from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence
import random
import pdb
    
class Darpa_raw_loader(Dataset):
    def __init__(self, conversation_list, dataclass):
        self.dialogs = []
        
        # f = open(txt_file, 'r')
        # dataset = f.readlines()
        # f.close()
        dataset = conversation_list
        
        temp_speakerList = []
        context = []
        context_speaker = []
        self.speakerNum = []      
        self.emoSet = set()
        self.sentiSet = set()
        # {'anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise'}
        pos = ['happiness']
        neg = ['anger', 'disgust', 'fear', 'sadness']
        neu = ['neutral', 'surprise']
        emodict = {'anger': "anger", 'disgust': "disgust", 'fear': "fear", 'happiness': "happy", 'neutral': "neutral", 'sadness': "sad", 'surprise': "surprise"}
        self.sentidict = {'positive': pos, 'negative': neg, 'neutral': neu}
        for i, data in enumerate(dataset):
            if data == '\n' and len(self.dialogs) > 0:
                self.speakerNum.append(len(temp_speakerList))
                temp_speakerList = []
                context = []
                context_speaker = []
                continue
                
            speaker = data.strip().split(':')[0]
            utt = data.strip().split(':')[1]
            # emo = data.strip().split('\t')[-1]
            
            # if emo in pos:
            #     senti = "positive"
            # elif emo in neg:
            #     senti = "negative"
            # elif emo in neu:
            #     senti = "neutral"
            # else:
            #     print('ERROR emotion&sentiment')       
            senti = 'None' # fake
            emo = 'None' 
            
            context.append(utt)
            if speaker not in temp_speakerList:
                temp_speakerList.append(speaker)
            speakerCLS = temp_speakerList.index(speaker)
            context_speaker.append(speakerCLS)
            
            self.dialogs.append([context_speaker[:], context[:], 'None', senti])
            self.emoSet.add('None')
        
        self.emoList = sorted(self.emoSet)   
        self.sentiList = sorted(self.sentiSet)
        self.labelList = self.emoList       
        self.speakerNum.append(len(temp_speakerList))
        
    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        return self.dialogs[idx], self.labelList, self.sentidict


if __name__ == "__main__":
    # test dataset loader
    txt_file = '/home/nttung/research/Monash_CCU/mini_eval/text_data/en_train/M01000G9C_en.txt'
    dataloader = DD_loader(txt_file, 'emotion')

    for idx, data in enumerate(dataloader):
        pdb.set_trace()

    pdb.set_trace()