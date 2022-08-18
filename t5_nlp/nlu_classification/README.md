### mT5 Result more details 

#### Text Classification 
Tested on three data sizes in training model.

|           | precision | recall | f1     | data size |
|-----------|-----------|--------|--------|-----------|
| mT5-small | 0.6434    | 0.6347 | 0.6311 | n=100     |
| mT5-small | 0.8075    | 0.7935 | 0.7954 | n=1000    |
| mT5-small | 0.8543    | 0.8546 | 0.8544 | n=10000   |



##### 100-samples

Epoch: 010; loss = 3.2339 cost time  1.1108
Accuracy: 0.6270 Loss in test 0.0000
                                 precision    recall  f1-score   support

        <extra_id_0>education     0.7500    0.8824    0.8108        85
    <extra_id_0>entertainment     0.6632    0.5888    0.6238       107
          <extra_id_0>finance     0.6111    0.3826    0.4706       115
             <extra_id_0>game     0.5368    0.7157    0.6134       102
         <extra_id_0>politics     0.6705    0.6211    0.6448        95
           <extra_id_0>realty     0.8434    0.6364    0.7254       110
          <extra_id_0>science     0.5196    0.5300    0.5248       100
          <extra_id_0>society     0.5377    0.5938    0.5644        96
           <extra_id_0>sports     0.8636    0.7835    0.8216        97
           <extra_id_0>stocks     0.4385    0.6129    0.5112        93
    
                     accuracy                         0.6270      1000
                    macro avg     0.6434    0.6347    0.6311      1000
                 weighted avg     0.6443    0.6270    0.6273      1000
##### 1000-samples

Epoch: 008; loss = 0.8370 cost time 7.9410
Accuracy: 0.7930 Loss in test 0.0000

                                 precision    recall  f1-score   support

        <extra_id_0>education     0.9367    0.8706    0.9024        85
    <extra_id_0>entertainment     0.7661    0.8879    0.8225       107
          <extra_id_0>finance     0.6667    0.7478    0.7049       115
             <extra_id_0>game     0.8333    0.7843    0.8081       102
         <extra_id_0>politics     0.7119    0.8842    0.7887        95
           <extra_id_0>realty     0.8942    0.8455    0.8692       110
          <extra_id_0>science     0.8169    0.5800    0.6784       100
          <extra_id_0>society     0.9359    0.7604    0.8391        96
           <extra_id_0>sports     0.9149    0.8866    0.9005        97
           <extra_id_0>stocks     0.5981    0.6882    0.6400        93

                 accuracy                         0.7930      1000
                macro avg     0.8075    0.7935    0.7954      1000
             weighted avg     0.8052    0.7930    0.7940      1000

##### 10000-samples

Epoch: 019; loss = 0.3964 cost time 73.6844
Accuracy: 0.8546 Loss in test 0.0000

                    precision    recall  f1-score   support
    
        education     0.9221    0.9350    0.9285      1000
    entertainment     0.8498    0.8490    0.8494      1000
          finance     0.8697    0.8480    0.8587      1000
             game     0.8690    0.8690    0.8690      1000
         politics     0.8121    0.8340    0.8229      1000
           realty     0.8633    0.8840    0.8735      1000
          science     0.8070    0.7860    0.7964      1000
          society     0.8600    0.8600    0.8600      1000
           sports     0.9076    0.9140    0.9108      1000
           stocks     0.7827    0.7670    0.7747      1000
    
         accuracy                         0.8546     10000
        macro avg     0.8543    0.8546    0.8544     10000
     weighted avg     0.8543    0.8546    0.8544     10000