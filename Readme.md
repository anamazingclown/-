# Python 第三次考核：神经网络编程

## 题目简介

- 小林有一个朋友沉迷学（mo）习（yu）以致于只能认出0和1这两个数字，所以请你做一个手写数字识别出来救救小林的朋友吧

## 要求
- 使用numpy
- 并且尽可能的提高识别率
- 将训练好的模型拿去参加[kaggle](https://www.kaggle.com/c/digit-recognizer/overview)上的比赛

## 具体内容
### 数学思想
对于计算机而言，它能够实现繁琐而复杂的计算等传统任务，而对于人而言，人的思维更具有模糊和发散性。对于不能精确知道的事情的运作，我们常常都会使用模型来估计其运作方式，模型中包含了各种参数，这些参数影响着最终的结果。通过模型，模拟结果，将其得到的结果与真实示例之间的比较，得到误差，调整参数，最后达到模拟接近结果的效果。

### 让人又爱又恨的矩阵

枯燥费力的计算，消磨着我们的耐心和美好时光。但其确确实实简化了我们计算。而在神经网络中， `矩阵`的存在举重若轻！

通过一系列的推理我们能够得到通过神经网络馈送信号可以表示为矩阵的乘法，得到结果与示例结果的误差可以通过矩阵乘法反向传递误差，通过梯度下降的方法来慢慢的调整矩阵中链接权重来进行学习，减小误差。神经网络规模越大，矩阵的地位更加凸显。更加重要的是，由于计算机编程语言能够理解矩阵计算，这就允许了计算机进行高速有效的计算。

### 实现

* 框架代码
  >1.初始化
  
  我们需要设置输出层节点，隐藏层节点，输出层节点的数量，并对学习率进行初始化。
  
  对权重即链接权重矩阵的初始化，权重是网络的核心，其关系着最终结果的好坏。这里对于具体题目进行分析可得，我们选择的是简单流行的优化初始权重的方法--使用正态分布采集权重
  ```
    def __init__(self,inodes,hnodes,onodes,leaningrate):
      self.inodes = inodes
      self.onodes = onodes
      self.hnodes = hnodes
      self.lr = leaningrate
      self.W_i_h = numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
      self.W_h_o = numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes ,self.hnodes))

      self.activation_function = lambda x:scipy.special.expit(x)
  ```
  
  >2.训练
  >
  >>1.针对样本得到结果(本质和步骤3一样)
  >>
  >>
  >>2.**对误差进行调整学习，改变链接权重**
  >  
  ``` 
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
  
  
   >>>  
   ``` 
  >3.接收输入，得到输出（查询功能，得到模拟结果）
  ```
    def query(self,input_list):
      inputs =numpy.array(input_list,ndmin = 2).T
      hiddeninputW = numpy.dot(self.W_i_h,inputs)
      hiddenoutputW = self.activation_function(hiddeninputW)
      finalinputW = numpy.dot(self.W_h_o,hiddenoutputW)
      finaloutputW = self.activation_function(finalinputW)
      return finaloutputW
  ```

### 测试示例
按照要求训练并将结果保存到csv文件中
```
#给的节点与学习率
input_nodes = 784
hidd_nodes = 100
output_nodes = 10

learning_rate = 0.09

n = neturalnetworks(input_nodes,hidd_nodes,output_nodes,learning_rate )
training_data_file = open("C:/Users/14752/Desktop/train.csv","r")
training_data_list = training_data_file.readlines()
training_data_list = training_data_list[1:]
training_data_file.close()
#多个世代进行训练，提高准确度
epoch = 7
for e in range(epoch):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
        targes = numpy.zeros(output_nodes)+0.01
        targes[int(all_values[0])]=0.99
        n.train(inputs,targes)

#对目标数据进行模拟结果，并将其保存
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
```
## 总结
此神经网络的实现还算是比较成功的，平均精确的能够达到0.96 

参考材料：《Python神经网络编程》  bilibili 度娘 
          
          

