# -*- coding = utf-8 -*-
# @Time :2021/1/26 13:46
# @Author : hys
# @File :。。.py
# @Sofeware : PyCharm
import numpy
import scipy.special
class neturalnetworks():

    #init the networks
    def __init__(self,inodes,hnodes,onodes,leaningrate):
        self.inodes = inodes
        self.onodes = onodes
        self.hnodes = hnodes
        self.lr = leaningrate
        self.W_i_h = numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.W_h_o = numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes ,self.hnodes))

        self.activation_function = lambda x:scipy.special.expit(x)

        pass
    #train the networks
    def train(self, input_list,targe_list):
        #将列表转化为矩阵
        inputW = numpy.array(input_list,ndmin = 2).T
        targeputW = numpy.array(targe_list,ndmin = 2 ).T


        #part1:
        hiddeninputW = numpy.dot(self.W_i_h, inputW)
        hiddenoutputW = self.activation_function(hiddeninputW)

        finalinputW = numpy.dot(self.W_h_o, hiddenoutputW)
        finaloutputW = self.activation_function(finalinputW)

        output_errorW = targeputW - finaloutputW

        hidden_errorW = numpy.dot(self.W_h_o.T,output_errorW)
        self.W_h_o += self.lr * numpy.dot((output_errorW * finaloutputW * (1.0 - finaloutputW)) , numpy.transpose(hiddenoutputW))
        self.W_i_h += self.lr * numpy.dot((hidden_errorW * hiddenoutputW * (1.0 - hiddenoutputW)), numpy.transpose(inputW))









        pass
    def query(self,input_list):

        inputs =numpy.array(input_list,ndmin = 2).T
        hiddeninputW = numpy.dot(self.W_i_h,inputs)
        hiddenoutputW = self.activation_function(hiddeninputW)
        finalinputW = numpy.dot(self.W_h_o,hiddenoutputW)
        finaloutputW = self.activation_function(finalinputW)
        return finaloutputW
        pass

input_nodes = 784
hidd_nodes = 100
output_nodes = 10

learning_rate = 0.09

n = neturalnetworks(input_nodes,hidd_nodes,output_nodes,learning_rate )
training_data_file = open("C:/Users/14752/Desktop/train.csv","r")
training_data_list = training_data_file.readlines()
training_data_list = training_data_list[1:]
training_data_file.close()

epoch = 7
for e in range(epoch):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
        targes = numpy.zeros(output_nodes)+0.01
        targes[int(all_values[0])]=0.99
        n.train(inputs,targes)

test_data_file = open("C:/Users/14752/Desktop/test.csv","r")
test_data_list = test_data_file.readlines()
test_data_list = test_data_list[1:]
test_data_file.close()

scorecard = []
temp = 0

import csv
import codecs


f = codecs.open('C:/Users/14752/Desktop/sample_submission.csv','w','gbk')
csv_writer = csv.writer(f)
csv_writer.writerow(["ImageId","Label"])

for record in test_data_list:
    all_values = record.split(',')
    # correct_label = int(all_values[0])
    intputs = (numpy.asfarray(all_values[0:])/255.0*0.99)+0.01
    outputs = n.query(intputs)
    label = numpy.argmax(outputs)
    scorecard.append(label)
i= 1
for e in scorecard:
    z =[]
    z.append(i)
    i+=1
    z.append(e)
    csv_writer.writerow(z)
f.close()
# scorecard_array = numpy.asfarray(scorecard)
# print("performance = ",scorecard_array.sum()/(scorecard_array.size))
