
This is the Collaborative Filtering Version 3
implements the timeSVD++ algorithm in "Advances in Collaborative Filtering" Yehuda Koren and Robert Bell

In this version:

Added: 
1. implicit feedback
2. changing scale of user ratings
3. sudden drifts emerging as “spikes” associated with a single day
4. temporal dynamics affected user preferences and thereby the interaction between users and items (user factor as a function of time)

All of the variations was implemented in the file TrainingSparseUIVectorT_Implicit.py

 
In version No. 2:

1. added mini_batch sgd training, now should handle a much bigger size of amazon review dataset.
2. used a more efficient way for computation(similar computation cost as sparse matrix), instead of treat the user-item-ranking matrix as a dense matrix
3. I build the TrainingSparseUD.py first(it is the most complex version so far), and add the other three model by simply remove part of the code from it. So, you will see some unnecessory code in the other three simpler model, but it won't affect the computation time.
Still,

the bias_only model achieved the lowest mean square error.
With the size of the dataset increasing, the error of the model will get higher. If the data size is around 5-10MB, the loss will be around 0.7-0.8, with the data size increasing to over 100MB, the loss will be over 0.9. Well, they are the limited observation I got, with little hyper-parameter tuning.


