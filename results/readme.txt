- Ablation_studies:
Contains the Mode+AugmentationSelection tests with three different seeds. This data was used for Figure 7, Table 3 and 4. 

- Target_augmentation_scores: Test accuracy scores of a RegNetY_400MF trained for 50 epochs with all the 22 different augmentations. We use these scores to test the zero-cost proxy augmentation selection method.

- zcost_population_regnet: 
Contains the fisher, jacob_cov and fisher_jacob zero-cost proxies for 20 random models from the search space for the 22 candidate augmentations, as well as the ranking they achieved based on the target ranking from the RegNetY_400MF model trained for 50 epochs with each augmentation that is in folder target_augmentation_scores. This scores are used in figure 4b in the paper.

- zcost_proxies_regnet:
Contains all the zero-cost proxies scores for the RegNetY_400MF model for the 22 candidate augmentations, as well as the ranking they achieved based on the target ranking that is in the folder target_augmentation_scores. Used for figure 4c in the paper to discover which are the best zero cost proxies for this task. 

