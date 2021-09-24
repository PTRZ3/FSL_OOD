## Few-shot Anomaly Detection By Representation Learning  

#### Methodology 1 - Siamese Networks with Contrastive Loss
+ Backbone:  
  siamese_resnet.py (Experiment 1-3)
+ Pretraining Stage:  
  siamese_without_oe_pretraining.ipynb (Experiment 1&3)
  siamese_with_oe_pretraining.ipynb (Experiment 2)
+ Few-shot Evaluation Stage:  
  siamese_test_evaluation.py (Experiment 1&2)
+ Representation Plot:  
  siamese_2d.html  
  siamese_3d.html  
    
    
#### Methodology 2 - Cross-Entropy Based Netwrork
+ Backbone:  
  baseline_resnet.py (Experiment 4&5)
+ Pretraining Stage:   
  ce_MaxSoftmax_pretraining.ipynb  (Experiment 4&5)
+ Few-shot Evaluation Stage:   
  ce_test_evaluation(ce backbone).py (Experiment 4&5)
  ce_test_evaluation(siamese backbone).py (Experiment 3)
+ Representation Plot:    
  ce_2d.html   
  ce_3d.html  
  

#### List of Experiments
![alt text](https://raw.githubusercontent.com/PTRZ3/fsl_ood/main/list_of_experiments.png)
  

