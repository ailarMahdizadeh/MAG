import pandas as pd
import numpy as np
import scipy.io as scio
from torch.utils.data import Dataset, TensorDataset

import torch
import torch.nn as nn
import MyModels
from sklearn import metrics

status = 'AD'
#train_los=[]
for ROInum in ([329]):
    qual_all = []
    for cv in ([1,2]):
        
        subInfo = pd.read_csv('/Users/ailarmahdizadeh/Desktop/samples' + status + '.csv')
        ## Ailar -> pay attention 
        temp = scio.loadmat('/public/home/liumx1/gcrn.tar/gcrn/kfold/' + status + '/shuffled_index_' + str(cv)) # I am thinking to not use this shuffling, this code is still debugging
        ##Ailar -<
        index = temp['index'][0]

        y_data1 = torch.from_numpy(subInfo['group'].to_numpy())
        y_data1 = torch.tensor(y_data1, dtype=torch.float32)
        # 1 NC 2 eMCI 3 MCI 4 lMCI 5 SMC 6 AD
        y_data1[y_data1 != 1] = 9
        y_data1[y_data1 == 1] = 0
        y_data1[y_data1 == 9] = 1

        y_data1 = y_data1[index]
        index = torch.tensor(index, dtype=torch.int)

        cut = int(len(subInfo) * 0.8)
        dataset_train = TensorDataset(index[0:cut], y_data1[0:cut])
        dataset_test = TensorDataset(index[cut:], y_data1[cut:])

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=30, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=30)

        ratio = y_data1[0:cut].sum() / (y_data1[0:cut].shape[0] - y_data1[0:cut].sum())
        if ratio < 1:
           weight = torch.cuda.FloatTensor([1, 1 / ratio])
        else:
           weight = torch.cuda.FloatTensor([ratio, 1])
        loss_func = nn.CrossEntropyLoss(weight)  # the target label is not one-hotted
                
        lr = 0.001
        EPOCH = 100
        best = 0.75
        qualified = []

        while not qualified:
            model = MyModels.MAHGCNNET(ROInum = ROInum,layer=int(round(ROInum/150))) #200/150->1 329/150->2  
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
            model.cuda()
            
            test_auc = []
            
            test_los = []
            train_auc = []
            sen = []
            spe = []
            
            for epoch in range(EPOCH):
                for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data
                    model.train()
                    b_y = b_y.view(-1)
                    b_y = b_y.long()
                    
                    
                    b_y = b_y.cuda()
                    temp = b_x.numpy().tolist()
                    batch_size=b_x.shape[0]
                    A1 = np.zeros((batch_size, 200, 200))
                    A2 = np.zeros((batch_size, 329, 329))
                    subcount=0
                    for id in temp:
                        fn=subInfo['Subj Name'][id]
                        fn=fn[:len(fn)-4]+'_FC.mat'
                        FCfile=scio.loadmat('/Users/ailarmahdizadeh/Desktop/samples/cc200'+str(200)+'/' + fn)
                        A1[subcount, :, :]=FCfile['FC']
                        FCfile = scio.loadmat('/Users/ailarmahdizadeh/Desktop/samples/cc400' + str(329) + '/' + fn)
                        A2[subcount, :, :] = FCfile['FC']
                        subcount = subcount + 1

                    A1 = torch.tensor(A1, dtype=torch.float32)
                    A1.cuda()
                    A2 = torch.tensor(A2, dtype=torch.float32)
                    A2.cuda()
                    
                    

                    output = model(A1, A2)  # rnn output

                    loss = loss_func(output, b_y)  # cross entropy loss
                    optimizer.zero_grad()  # clear gradients for this training step
                    loss.backward()  # backpropagation, compute gradients
                    optimizer.step()  # apply gradients

                    predicted = torch.max(output.data, 1)[1]
                    correct = (predicted == b_y).sum()
                    accuracy = float(correct) / float(b_x.shape[0])
                    train_auc = accuracy
                    
                    
                    print('[Epoch %d, Batch %5d] loss: %.3f' %
                          (epoch + 1, step + 1, loss))
                    print('|train diag loss:', loss.data.item(), '|train accuracy:', accuracy
                          )

                    if epoch>=30 and accuracy>=0.85:
                        #lr = 0.001
                        #optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
                        predicted_all = []
                        test_y_all = []
                        model.eval()
                        with torch.no_grad():
                            for i, (test_x, test_y) in enumerate(test_loader):
                                test_y = test_y.view(-1)
                                test_y = test_y.long()
                                test_y = test_y.cuda()
                                temp = test_x.numpy().tolist()
                                batch_size = test_x.shape[0]
                                A1 = np.zeros((batch_size, 200, 200))
                                A2 = np.zeros((batch_size, 329, 329))
                                subcount = 0
                                for id in temp:
                                    fn = subInfo['Subj Name'][id]
                                    fn = fn[:len(fn) - 4] + '_FC.mat'
                                    FCfile = scio.loadmat('/Users/ailarmahdizadeh/Desktop/samples/cc200'+str(200)+'/' + fn)
                                    A1[subcount, :, :] = FCfile['FC']
                                    FCfile = scio.loadmat('/Users/ailarmahdizadeh/Desktop/samples/cc200' + str(329) + '/' + fn)
                                    A2[subcount, :, :] = FCfile['FC']
                                    subcount = subcount + 1

                                A1 = torch.tensor(A1, dtype=torch.float32)
                                A1.cuda()
                                A2 = torch.tensor(A2, dtype=torch.float32)
                                A2.cuda()
                                test_x.cuda()
                                
                                test_output = model(A1, A2)

                                test_loss = loss_func(test_output, test_y)
                                print('[Epoch %d, Batch %5d] valid loss: %.3f' %
                                      (epoch + 1, step + 1, test_loss))
                                predicted = torch.max(test_output.data, 1)[1]
                                correct = (predicted == test_y).sum()
                                accuracy = float(correct) / float(predicted.shape[0])
                                test_y = test_y.cpu()
                                predicted = predicted.cpu()
                                predicted_all = predicted_all + predicted.tolist()
                                test_y_all = test_y_all + test_y.tolist()

                        correct = (np.array(predicted_all) == np.array(test_y_all)).sum()
                        accuracy = float(correct) / float(len(test_y_all))
                        
                        sens = metrics.recall_score(test_y_all, predicted_all, pos_label=1)
                        
                        spec = metrics.recall_score(test_y_all, predicted_all, pos_label=0)
                        
                        auc = metrics.roc_auc_score(test_y_all, predicted_all)
                        print('|test accuracy:', accuracy,
                              '|test sen:', sens,
                              '|test spe:', spec,
                              '|test auc:', auc,
                              )
                        
                        if best < auc and sens>0.70 and spec>0.70:
                            best = auc
                            qualified.append([accuracy, sens, spec, auc])
                            torch.save(model.state_dict(), '/Users/ailarmahdizadeh/Desktop/samples' + str(ROInum) + '_magcn_hc_'+ status+'_catsum_cv' + str(cv)+'.pth')

        qual_all.append(qualified[-1])

    print(qual_all)
    print(np.mean(qual_all, axis=0))
    print(np.std(qual_all, axis=0))