# 朴素贝叶斯

## 相关问题回答

### 先验概率和后验概率
先验（A priori；又译：先天）在拉丁文中指“来自先前的东西”，或稍稍引申指“在经验之前”。近代西方传统中，认为先验指无需经验或先于经验获得的知识。它通常与后验知识相比较，后验意指“在经验之后”，需要经验。这一区分来自于中世纪逻辑所区分的两种论证，从原因到结果的论证称为“先验的”，而从结果到原因的论证称为“后验的”。

先验概率是指根据以往经验和分析得到的概率，如全概率公式 中的 ，它往往作为“由因求果”问题中的“因”出现。后验概率是指在得到“结果”的信息后重新修正的概率，是“执果寻因”问题中的“因” 。后验概率是基于新的信息，修正原来的先验概率后所获得的更接近实际情况的概率估计。先验概率和后验概率是相对的。如果以后还有新的信息引入，更新了现在所谓的后验概率，得到了新的概率值，那么这个新的概率值被称为后验概率。

为了很好的说明这个问题，在这里举一个例子：
玩英雄联盟占到中国总人口的60%，不玩英雄联盟的人数占到40%：

为了便于数学叙述，这里我们用变量X来表示取值情况，根据概率的定义以及加法原则，我们可以写出如下表达式：

P(X=玩lol)=0.6；P(X=不玩lol)=0.4，这个概率是统计得到的，即X的概率分布已知，我们称其为先验概率(prior probability)；

另外玩lol中80%是男性，20%是小姐姐,不玩lol中20%是男性，80%是小姐姐,这里我用离散变量Y表示性别取值，同时写出相应的条件概率分布：


P(Y=男性|X=玩lol)=0.8，P(Y=小姐姐|X=玩lol)=0.2

P(Y=男性|X=不玩lol)=0.2，P(Y=小姐姐|X=不玩lol)=0.8

那么我想问在已知玩家为男性的情况下，他是lol玩家的概率是多少：

依据贝叶斯准则可得：

P(X=玩lol|Y=男性)=P(Y=男性|X=玩lol)*P(X=玩lol)/

[ P(Y=男性|X=玩lol)*P(X=玩lol)+P(Y=男性|X=不玩lol)*P(X=不玩lol)]


最后算出的P(X=玩lol|Y=男性)称之为X的后验概率，即它获得是在观察到事件Y发生后得到的