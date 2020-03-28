import torch.utils.data.dataset
import numpy as np
import torch
import re
PAD_token=0
SOS_token=1
EOS_token=2
UNK_token=3
MAX_LENGTH=80
def normalizeString(s):
    s = s.lower()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s
class VOC():
    def __init__(self):
        self.word2index = {"PAD":0 ,"SOS": 1, "EOS":2,"UNK":3 }#word转索引，字典，word字符串做查询键值
        self.index2word = {0: "PAD", 1: "SOS", 2:"EOS",3:"UNK"}
        self.freq_count={}
        self.n_words = 4  # Count SOS and EOS
        self.fiter_freq_shred=4
    def addsentense(self,s):
        for word in s.split():
            self.addWord(word)
    def addWord(self,w):
        if  w not in self.word2index.keys() and w not in self.freq_count.keys():
            self.freq_count[w]=1
        elif w not in self.word2index.keys() and self.freq_count[w] == self.fiter_freq_shred:
            self.word2index[w]=self.n_words 
            self.index2word[self.n_words]=w
            self.n_words +=1
        else:self.freq_count[w]+=1
    def idxs2words(self,words):
        word_encoding=[str(self.index2word[word]) for word in words] 
        word_encoding=" ".join(word_encoding) 
        return word_encoding
    def Build_VOC(self,data):
        for s in data:
            self.addsentense(s)  
        print("vocsize: "+ str(self.n_words))   
    def words2idxs(self,sentence):
        if isinstance(sentence, str): 
            words = sentence.split(' ')
            word_encoding=[SOS_token]
            for w in words:
                if w in self.word2index.keys():
                    word_encoding.append(int(self.word2index[w]))
                else: word_encoding.append(UNK_token)
            word_encoding.append(EOS_token)
            return word_encoding
        elif isinstance(sentence, list): 
            tokens_list = [self.words2idxs(t) for t in sentence]
            return tokens_list
class Dataset(object):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
class My_dataset(Dataset):
    # root 是训练集的根目录， mode可选的参数是train，test，validation，分别读取相应的文件夹
    def __init__(self,  mode="train",dir="C:\\Users\\10718\\PycharmProjects\\data\\movie_subtitles.txt",voc_save_path="dkn_duconv/"):
        self.mode = mode
        self.dir = dir
        self.voc=VOC()
        self.Qs,self.As=self.read_file()
    def read_file(self):
        Lines=[]
        with open(self.dir,"r",encoding='utf-8') as f:
            for idx,line in enumerate(f):
                Lines.append(normalizeString(line.strip()))
        length=len(Lines)
        Qs=Lines[0:length:2]
        As=Lines[1:length:2]
        F_Qs=[]
        F_As=[]
        for idx in range(0,len(Qs)):
                if len(Qs[idx].split())<=MAX_LENGTH-2 and len(As[idx].split()) <=MAX_LENGTH-2:
                    F_Qs.append(Qs[idx])
                    F_As.append(As[idx])
        Lines=F_Qs+F_As
        # max_len1=max([len(it.split()) for it in Qs])
        # max_len2=max([len(it.split()) for it in As])
        self.voc.Build_VOC(Lines)
        Qs=[self.voc.words2idxs(line) for line in F_Qs]
        As=[self.voc.words2idxs(line) for line in F_As]
        return Qs,As
    def __getitem__(self, index):
        fetch={
            "Q": self.Qs[index],
            "A": self.As[index]
        }
        return fetch
    def __len__(self):
        return len(self.Qs)
def collate_fn(batch):
    Q = [torch.LongTensor(item['Q'] )for item in batch]
    A = [torch.LongTensor(item['A']) for item in batch]
    return {
        'Q': Q,
        'A': A
    }

