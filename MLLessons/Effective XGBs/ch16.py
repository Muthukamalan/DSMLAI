# Training for different metric
## need a scaler value to compare between models and direction
# - by default accuracy
'''
- Accuracy measures how often the model correctly predicts the target class.
- Precision measures how often the model is correct when it predicts the positive class.
- Recall measures how often the model can correctly identify the positive class.
- The area under the ROC curve is one measure to balance precision and recall.
- The F1 score combines both precision and recall by using their harmonic mean.

  Tuning a model for recall differs from tuning for accuracy because recall and accuracy are different metrics that measure different aspects of a model's performance. 
  Recall measures the proportion of true positive instances that were correctly identified by the model, 
  while accuracy measures the proportion of all instances that were correctly classified. 
  
  As a result, optimizing for one metric may not necessarily optimize for the other

  
  For example, in a medical diagnosis scenario where failing to identify a disease could have serious consequences, it may be more important to prioritize recall (i.e., correctly identifying all positive cases) even if it means sacrificing some precision (i.e., having more false positives). On the other hand, in a spam email classification scenario where incorrectly classifying an email as spam could result in lost business opportunities, it may be more important to prioritize precision (i.e., minimizing false positives) even if it means sacrificing some recall (i.e., having more false negatives).
'''