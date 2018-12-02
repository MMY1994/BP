import xlrd
import numpy as np
import tensorflow as tf


data = xlrd.open_workbook('ML_data1.xlsx')
sheeta = data.sheet_by_name(u'Sheet2')
sheetb = data.sheet_by_name(u'Sheet3')
#sheeta = sheeta.astype('float')
#sheetb = sheetb.astype('float')

data1 = np.zeros((452,279),dtype=float)
data1 = np.float32(data1)
label = np.zeros((452,16),dtype=float)
label = np.float32(label)
#print(sheeta.cell(2,2).value)
#print(min(sheeta.col_values(2)))
#print(max(sheeta.col_values(3)))
#print(type(max(sheeta.col_values(3))))

for i in range(452):
    for j in range(279):
        if(np.max(sheeta.col_values(j)) == np.min(sheeta.col_values(j))==0):
            data1[i,j] = 0
        else:
            data1[i,j]=(sheeta.cell(i,j).value - np.min(sheeta.col_values(j)))/(np.max(sheeta.col_values(j)) - np.min(sheeta.col_values(j)))
for a in range(452):
    for b in range(1,17):
        if(sheetb.cell(a+1,0).value == b):
            label[a,b-1] = 1

train_data = data1[:300,:]
train_label = label[:300,:]
test_data = data1[301:451,:]
test_label = label[301:451,:]



x=tf.placeholder("float",[None,279])
y=tf.placeholder("float",[None,16])

def layer(inputs,input_node,output_node,active_function=None):
    w1=tf.Variable(tf.random_normal([input_node,output_node],stddev=0.1))
    b1=tf.Variable(tf.random_normal([output_node]))
    y1=tf.add(tf.matmul(inputs,w1),b1)

    if(active_function==None):
        outputs=y1
    else:
        outputs=active_function(y1)
    return outputs

y2=layer(x,input_node=279,output_node=150,active_function=tf.nn.tanh)
y3=layer(y2,input_node=150,output_node=75,active_function=tf.nn.tanh)
y4=layer(y3,input_node=75,output_node=16)

loss=tf.reduce_mean(tf.square(y4-y))
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
corr=tf.equal(tf.argmax(y4,1),tf.argmax(y,1))
accr=tf.reduce_mean(tf.cast(corr,"float"))

init=tf.global_variables_initializer()
print("function ready")

saver=tf.train.Saver()
sess=tf.Session()

work_mode="test"

if(work_mode=="train"):
    sess.run(init)
    for i in range(20000):
        sess.run(train_step,feed_dict={x:train_data,y:train_label})
        saver.save(sess,"Model/model.ckpt")
        if(i%500==0):
            a=sess.run(accr,feed_dict={x:train_data,y:train_label})
            print("accr:%.3f" % a)
if(work_mode=="test"):
    saver.restore(sess,"./Model/model.ckpt")
    a = sess.run(accr,feed_dict={x:test_data,y:test_label})
    print("accr:%.3f" % a)