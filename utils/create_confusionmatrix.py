import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib

# array = [[53, 0, 4, 0, 0], [3, 30, 0, 13, 0], [0, 0, 61, 0, 3], [0, 10, 0, 80, 9], [0, 0, 2, 4, 51]]
array = [[10, 0, 0, 0, 0], [0, 11, 0, 1, 0], [1, 0, 8, 0, 0], [0, 3, 0, 11, 1], [0, 0, 0, 1,9]]
df_cm = pd.DataFrame(array, index=['O', 'A', 'AO', 'AA', 'GBM'],
                     columns=['O', 'A', 'AO', 'AA', 'GBM'])
# df_cm = pd.DataFrame(array, index=['KB', 'O', 'A'],
#                                           columns=['KB', 'O', 'A'])
f, ax = plt.subplots(figsize=(11, 8))
sn.set()
a = sn.heatmap(df_cm, annot=True)  #, annot_kws={"size": 10}
# sn.heatmap(df_cm, annot=True)
plt.ylabel('GroundTruth')
plt.xlabel('Prediction')
# matplotlib.rcParams.update({'font.size':18})
# fig = a.get_figure
# fig.savefig('confusion'+str(epoch_start) + '.png')
# plt.imshow(a)
plt.savefig('../result/confusion_final_test.png')
plt.close()
