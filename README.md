# dp-lstm-harddrv

- SMART-sequence Harddrive failure prediction
- Tensorflow ![Tensorflow][https://i.ytimg.com/vi/oZikw5k_2FM/maxresdefault.jpg]


```bash
$ conda list
# packages in environment at XXX :
#
cycler                    0.10.0                   py35_0  
fontconfig                2.11.1                        6  
freetype                  2.5.5                         1  
libgfortran               3.0.0                         1  
libpng                    1.6.22                        0  
libxml2                   2.9.2                         0  
matplotlib                1.5.1               np111py35_0  
mkl                       11.3.3                        0  
numpy                     1.11.1                   py35_0  
openssl                   1.0.2h                        1  
pandas                    0.18.1              np111py35_0  
pip                       8.1.2                    py35_0  
protobuf                  3.0.0b2                   <pip>
pyparsing                 2.1.4                    py35_0  
pyqt                      4.11.4                   py35_3  
python                    3.5.2                         0  
python-dateutil           2.5.3                    py35_0  
pytz                      2016.4                   py35_0  
qt                        4.8.7                         3  
readline                  6.2                           2  
scikit-learn              0.17.1              np111py35_2  
scipy                     0.17.1              np111py35_1  
setuptools                23.0.0                   py35_0  
sip                       4.16.9                   py35_0  
six                       1.10.0                   py35_0  
sqlite                    3.13.0                        0  
tensorflow                0.10.0                    <pip>
tk                        8.5.18                        0  
wheel                     0.29.0                   py35_0  
xz                        5.2.2                         0  
zlib                      1.2.8                         3 

```

```bash
start training
Iter 1500, Minibatch Loss= 0.648513, training Accuracy= 0.77667
Iter 3000, Minibatch Loss= 0.359366, training Accuracy= 0.76000
Iter 4500, Minibatch Loss= 0.212365, training Accuracy= 0.92000
Iter 6000, Minibatch Loss= 0.106173, training Accuracy= 0.97000
Iter 7500, Minibatch Loss= 0.091992, training Accuracy= 0.98667
Iter 9000, Minibatch Loss= 0.063622, training Accuracy= 0.97333
Iter 10500, Minibatch Loss= 0.042866, training Accuracy= 0.99333
Iter 12000, Minibatch Loss= 0.031273, training Accuracy= 0.99333
Iter 13500, Minibatch Loss= 0.023324, training Accuracy= 0.99667
Iter 15000, Minibatch Loss= 0.022132, training Accuracy= 0.99667
Iter 16500, Minibatch Loss= 0.018568, training Accuracy= 0.99667
Iter 18000, Minibatch Loss= 0.016910, training Accuracy= 0.99667
Iter 19500, Minibatch Loss= 0.016658, training Accuracy= 0.99667
Iter 21000, Minibatch Loss= 0.015910, training Accuracy= 0.99667
Iter 22500, Minibatch Loss= 0.015321, training Accuracy= 0.99667
Iter 24000, Minibatch Loss= 0.014885, training Accuracy= 0.99667
Iter 25500, Minibatch Loss= 0.014400, training Accuracy= 0.99667
Iter 27000, Minibatch Loss= 0.013767, training Accuracy= 0.99667
Iter 28500, Minibatch Loss= 0.012972, training Accuracy= 0.99667
Optimization Finished!
Testing Accuracy: 0.985294
Execution time :1.1116789999998673

```