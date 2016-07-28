# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 22:04:56 2016

@author: Natay
"""

from itertools import chain, permutations
import numpy as np
import os
import glob
import pickle, gzip

from sklearn.cross_validation import train_test_split 


def blomap_encoder(r_data):
    
    decoder = open(r'D:/data/decoder.txt', 'r').readlines()
    bmap = {x.split('=')[0].strip(): x.split('=')[1].split() for x in decoder}
    vectors = []
    for s in r_data:
        e = list(chain.from_iterable([bmap[aa] for aa in s[0]]))
        vectors.append(([float(x) for x in e], int(s[1].strip())))

    return np.array(vectors)
    
def options(show_full=False):
    
    full_options = """
    
                Data set options are (Full option details):
                      - HIV-1 protease cleavage (hiv-1) has four data sets.
                            - schilling (3272), 746, 1625, and impens (947) 
                            -.txt files formatted like :-1AAAMKRHG,-1AAAMSSAI,
                            - -1/1 binary labels (0/1 if using Theano net)
                      - Mice protein expression (mice)- has one data set.
                            - excel file, missing values
                            - eight classes, based on trisomy of mice.
                      - Splice junction (splice) - one data set.
                            - .txt file, has ambigous genetic data.
                            - 3 classes, (introns, exons, neither)
                      - Chess (KRK) - one data set.
                            - personal project to test whole learning library
                            - .txt file
                            - 16 classes, based on number of moves before end
                      - Indian Liver Patients (ilpd)- one data set
                            -excel file, missing values
                            -1/2 binary labels
                    """
    if not show_full:
        print ('''
                Data set options are (options):
                    - HIV-1 protease cleavage (hiv-1)- TESTING
                    - Mice protein expression (mice)- not tested
                    - Splice Junction         (splice)- not tested
                    - chess (KRK)             (chess)- tested 
                    - test (mnist data)       (test)- tested
                    - Indian Liver Patients   (ilpd)- not tested
                ''')
                
    elif show_full:
        print(full_options)

# Data generator class from list of options.                               
class Generate:

    def __init__(self, name):
        
        self.name = name
        
    # add data_path param
    def fetch_data(self):

        if self.name == 'hiv-1':
            
            data_dir = r'D:/data/humanSNP'
            ds = {}
            
            for data_path in glob.glob(os.path.join(data_dir, '*.txt')):
                rawdata = [x.split(',') for x in open(r'' + data_path, 'r').readlines()]
                ds[data_path.split('/')[-1]] = rawdata
            return ds

        if self.name == 'test':
            
            full = gzip.open(r'D:\mnist.pkl.gz', 'rb')
            tr_set, va_set, te_set = pickle.load(full, encoding='latin1')
            full.close()
            return tr_set, va_set, te_set
            
        if self.name == 'chess':
            
            data_path = r'D:/data/chess(KRK)/chess(KRK).txt'
            rawdata = [x.split(',') for x in open(data_path,'r').readlines()]
            return rawdata
        
        if self.name == 'splice':
            data_path = r'D:/data/mol_bio(splice-junction)/splice_junct.txt'
            rawdata = [x.split(',') for x in open(data_path,'r').readlines()] 
                
        else:
            options()
            raise ValueError('%s is not loadable. See options above'%self.name)


# Data processing class          
class Process:
    
    def __init__(self, mode):
        self.mode = mode


    def train_test_generator(self, mdata, select_tr=None, select_te=None,
                             return_count=None, test_size=None, random_state=50):
        
        def update(mapper, combo):
            if str(combo) not in mapper.keys():
                mapper[str(combo)] = []
                mapper[str(combo)].append([train_data, test_data]) 
                
            elif str(combo) in hivdata.keys():
                mapper[str(combo)].append([train_data, test_data])
            
        if self.mode == 'hiv-1':
            train_test_combos = permutations(mdata.keys(), 2)
            hivdata = {}
            
            for i, dat in enumerate(train_test_combos):
                train_data = (np.array([x[0] for x in mdata[dat[0]]]),
                              np.array([x[1] for x in mdata[dat[0]]]))
                test_data = (np.array([x[0] for x in mdata[dat[1]]]),
                             np.array([x[1] for x in mdata[dat[1]]]))
                
                if (select_tr is not None and select_tr in dat[0]) and \
                    (select_te is None):
                    update(hivdata,dat)
                elif (select_te and select_tr is not None) and \
                    (select_tr in dat[0] and select_te in dat[1]):
                    update(hivdata,dat)
                elif select_tr and select_te is None:
                    update(hivdata,dat)
    
            unfilterd_keys = [x for x in hivdata.values()]

            if return_count is None:
                return unfilterd_keys

            elif type(return_count) is int and return_count>=len(unfilterd_keys):
                return unfilterd_keys
                
            elif type(return_count) is int and len(unfilterd_keys)>return_count > 0:
                filterd_keys = unfilterd_keys[:return_count]
                return filterd_keys
                
            else:
                options()
                raise ValueError('% is not an integer or None'%return_count)
                

        if self.mode =='chess':
            
            all_labels= [x[-1].strip() for x in mdata]
            spc_cache = {all_labels.index(x):x for x in set(all_labels)}
            spc_labels = [spc_cache[x] for x in sorted(spc_cache)]
            full_lables = dict(zip(spc_labels, range(len(spc_cache))))
            
            xpoints = list(set(','.join([','.join([x[0],x[2],x[4]]) for x in mdata])))
            point_encoder = dict(zip(sorted(xpoints), range(len(xpoints))))
            processed_data = []
            
            for point in mdata:
                
                allx = [point_encoder[point[0]], int(point[1]),
                        point_encoder[point[2]], int(point[3]),
                        point_encoder[point[4]], int(point[5])]
                        
                corrd =[allx, full_lables[point[-1].strip()]]        
                processed_data.append(corrd)
            X = np.array([x[0] for x in processed_data])
            y = np.array([x[1] for x in processed_data])
            
            xtr, xte, ytr, yte = train_test_split(X, y, test_size=test_size,
                                                  random_state=random_state)
                                                  
            return xtr, ytr, xte, yte

    def handle_missing_values(self, X):
        pass

    @staticmethod
    def transform_labels(full_labels, label_in, label_out):
        
        for i,n in enumerate(full_labels):
            if n == label_out:
                full_labels[i] = label_in
                
 
        return full_labels


# wrapping functon
def load(mode, select_tr=None, select_te=None, return_count=None, 
         transformlabels=False, inoutlabels=None, test_size=None, rnd_st=50):

    fulldata = Generate(mode).fetch_data()

    if mode == 'test':
        train, valid, test = fulldata
        return train[0], train[1], test[0], test[1]

    if mode == 'hiv-1':
        proc = Process(mode)
        encoded = {}
        for pro in fulldata:
            encoded[pro] = blomap_encoder(fulldata[pro])

        uncompdata = proc.train_test_generator(mdata=encoded, 
                                               select_tr=select_tr, 
                                               return_count=return_count,
                                               select_te=select_te)
    
        if transformlabels:
            for se in uncompdata:
                for p,sepd in enumerate(se):
                    se[p] = list(sepd)
                    for i,f in enumerate(se[p]):
                        sepd[i] = list(f)
                        sepd[i][1] = proc.transform_labels(full_labels=sepd[i][1],
                                                        label_in=inoutlabels[0],
                                                        label_out=inoutlabels[1])
            return uncompdata
            
        elif not transformlabels:
            return uncompdata
            
    if mode == 'chess':
        rawdata = Generate(mode).fetch_data()
        xtr, ytr, xte, yte = Process(mode).train_test_generator(rawdata,
                                                                test_size=test_size,
                                                                random_state=rnd_st)
        return xtr, ytr, xte, yte
    
    else:
        options()
        raise ValueError('%s is not a loadable dataset. See options()'%mode)
       


if __name__ == '__main__':
    
    d = load('test')
    
    hiv1 = load('hiv-1',
                transformlabels=True, 
                inoutlabels=[0,-1],
                select_tr='schilling',
                select_te='impens')
                






