
This is the Collaborative Filtering Version 3
implements the timeSVD++ algorithm in "Advances in Collaborative Filtering" Yehuda Koren and Robert Bell

In this version:

Added: 
1. implicit feedback
2. changing scale of user ratings
3. sudden drifts emerging as “spikes” associated with a single day
4. temporal dynamics affected user preferences and thereby the interaction between users and items (user factor as a function of time)

All of the variations was implemented in the file TrainingSparseUIVectorT_Implicit.py





