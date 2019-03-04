#to find both f1 and accuracy graph by omitting graph
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

f1_acc_omit_index  = [(64.8,64.7,0),(65.3,65.23,1),(64.6,64.5,2),(64.7,64.7,3),(64.8,64.7,4),(64.51,64.19,5),(64.43,63.80,6),(64.54,64.19,7),
(64.75,64.65,8),(64.54,64.19,9),(64.71,64.47,10),(64.66,64.29,11),(64.37,64.13,12),(64.78,64.26,13),(64.87,64.38,14),
(64.46,64.32,15),(64.51,64.32,16)]
sorted_ans = sorted(f1_acc_omit_index,key = lambda tup:(tup[0],tup[1]))
omitted_feature_name = ["happiness","sadness","surprise","anger","disgust","fear","upper face","lower face","shape params", "pitch",\
"harmonic","quotient","mcep","pdm","pdd","formant","peak slope"]

f1=[]
feat_name=[]
for t in sorted_ans:
    f1.append(t[0])
    feat_name.append(omitted_feature_name[t[2]])
    
    

all_data={'F-1 score':f1, "Omitted feature":feat_name}
df = pd.DataFrame(all_data)
sns.set(style="whitegrid")
ax = sns.barplot(y="Omitted feature", x="F-1 score", data=df)
ax.set(xlim=(63.75, 65.5))
plt.savefig('f1_by_omitted_feat.pdf', bbox_inches="tight")
