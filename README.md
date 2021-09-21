### Few-shot Anomaly Detection By Representation Learning  

## Methodology 1 - Siamese Networks with Contrastive Loss
+ Backbone:  
  siamese_resnet.py  
+ Pretraining Stage:  
  siamese_without_oe_pretraining.ipynb  
  siamese_with_oe_pretraining.ipynb  
+ Few-shot Evaluation Stage:  
  siamese_test_evaluation.py  
+ Representation Plot:  
  siamese_2d.html  
  siamese_3d.html  
    
    
## Methodology 2 - Cross-Entropy Based Netwrork
+ Backbone:  
  baseline_resnet.py  
+ Pretraining Stage:   
  ce_MaxSoftmax_pretraining.ipynb  
+ Few-shot Evaluation Stage:   
  ce_test_evaluation(ce backbone).py  
  ce_test_evaluation(siamese backbone).py  
+ Representation Plot:    
  ce_2d.html   
  ce_3d.html  
  
  
  

