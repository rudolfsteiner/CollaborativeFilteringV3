
This is the Collaborative Filtering Version 3
implements the timeSVD++ algorithm in "Advances in Collaborative Filtering" Yehuda Koren and Robert Bell

In this version:

Added: 
1. implicit feedback
2. changing scale of user ratings
3. sudden drifts emerging as “spikes” associated with a single day
4. temporal dynamics affected user preferences and thereby the interaction between users and items (user factor as a function of time)

All of the variations was implemented in the file TrainingSparseUIVectorT_Implicit.py

I didn't implement the spline_based model for the gradual concept drift. The amazon dataset is too sparse, it makes no sense to use the spline_based model.

Anyway, it is the last version for the Amazon Recommender System using Matrix Factorization Model. I can still add the neighborhood_based model to improve the performance. But it just needs a little extra time and programming which provides little challenge.






