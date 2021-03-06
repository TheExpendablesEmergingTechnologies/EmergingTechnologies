# Improving Sentiment Analysis with Mutli-Task Learning of Negation [Deployed using MLOps_MLflow]

## Course: CMPE 297 sec 49 Advance Deep Learning

## Team Members

Team Members | Contributions | 
--- | --- | 
Akshaya Nagarajan |  Model training, visualizations and tensor board integration |
Pooja Patil |  Literature survey, Ablation study and model training   |
Sivaranjani Kumar |  Collecting and preprocessing dataset, Model training   |
Vigneshkumar Thangarajan |  Model training and deployed web application   |

## Proposal: 

[Project_Proposal_link](https://github.com/TheExpendablesEmergingTechnologies/EmergingTechnologies/blob/main/Documents/ProjectProposal.pdf)

Sentiment Analysis is a process where one can mine people’s opinions from a piece of text. At first glance, this task may look like a simple text classification problem, but once deep dived, one can find out challenges which can affect the sentiment analysis accuracy. For instance, predicting sentiment just using the words in a sentence can lead to major pitfalls such as detecting sarcasm, irony, word ambiguity, use of negating words, also multipolarity. Out of all these phenomena, negation is the most prevalent. Any sentiment analysis model must be able to identify negation and try to remove the effect that its scope has on the final sentiment of a text. In this project, we propose to use Multi-task learning approach, using a cascading and hierarchical neural architecture of LSTM layers. This architecture will explicitly train the model with negation as an auxiliary task and eventually help in improving the main task of sentiment analysis. 

## Steps to run the application locally

1. virtualenv venv
2. source venv/bin/activate
3. pip install -r reqirements.txt
4. python app/server.py serve

This will deploy the model locally.


## Model Architecture

![alt text](https://github.com/TheExpendablesEmergingTechnologies/EmergingTechnologies/blob/main/arch.png)

## Implemented Models

- Single-task model
- Multi-task SFU
- Multi-task CD

## Datasets

- [SST](https://nlp.stanford.edu/sentiment/treebank.html)

Stanford Sentiment Treebank (SST): This dataset contains 11,855 sentences which are taken from the movie reviews specific to English-language. There are two different versions to this dataset. They are SST-fine and SST-binary setting. SST was annotated for fine-grained sentiment with five different class labels. They are strong negative, negative, neutral, positive and strong positive which are specific to the SST-fine. Whereas in the SST-binary setting has 9,613 sentences where the neutral label is removed, instead strong and the normal label in SST-fine are merged to form only positive and negative labels. 

- [Conan Doyle Neg (*Sem 2012)](https://www.clips.uantwerpen.be/sem2012-st-neg/)

Conan Doyle Neg (*Sem 2012): The negation detection model with annotation will work on the sentences to extract the cues (words that change the polarity if the sentences) and scopes (words that gets affected by the cues). This Conan Doyle dataset is used for training the negation model which contains the stories of Conan Doyle and are manually annotated for the negation cues, negation scopes and events. The Bioscope corpus has annotation schemes which was also employed in Conan Doyle but with some major changes. This dataset contains the shared task version from 2012 SEM negation detection. This version consists of 848 negation sentences, out of 787 development sentences, 144 are negated and comprising total 3,640 sentences in training set. The test set consist of total 1089 sentences and out of these 235 sentences are negated. Conan Doyle annotates the different cue types they are sub-token, word-based and multi-word negation scope.  

## Result and Evaluation

![alt text](https://github.com/TheExpendablesEmergingTechnologies/EmergingTechnologies/blob/main/Images/training.png)

The main objective of this paper is not to achieve new state-of-art results for sentiment analysis but rather to gauge the relative contribution of negation task as auxiliary task in sentiment analysis. However, we still achieved competitive results at the end. The single-task model achieves an average accuracy of 46.49 on SST-fine. These results are better than standard performance for a Bidirectional LSTM model 45.6 and competitive with similar models. The extensive analysis of the results reveals several effects of using negation detection as an auxiliary task. On one hand, we find that even a small amount of annotated negation data allows a multi-task learner to improve its performance, while on the other hand, it is necessary to have enough sentiment data to achieve relatively good performance in order to see improvements in single task learning models.

## Ablation Study

Trained the model with a different auxillary dataset for the Negation task.
- Original Dataset: cdt.conllu
- Dataset used for Ablation Study: cdd.conllu

[Model Training Code with Different Auxillary Dataset.](https://github.com/TheExpendablesEmergingTechnologies/EmergingTechnologies/blob/main/Model/DLProject_ModelTrain_Diff_Aux_Set.ipynb)

Below is the list of **best Validation Accuracies** for the 2 different datasets for each run.

Runs | cdt.conllu | cdd.conllu |
--- | --- | --- |
1 | 43.7 | 42.2 |
2 | 43.8 | 43.4 |
3 | 43.5 | **44.0** |
4 | 42.0 | 43.8 |
5 | **45.0** | 42.9 |

## MLFOW is used as complete Machine Learning Pipeline
There are 3 main modules in MLFLOW:
1. Mlflow Tracking - MLflow also exposes API for creating experiment, logging the model parameters and logging the model evaluation metrics.
2. Mlflow Project - Mlflow also uses Github artifact to train the model.
3. Mlflow models - Model artifacts, logs and parameters are saved in a common global location.

## Below is the Screenshot of MLFLOW comparision between experiments
![alt text](https://github.com/TheExpendablesEmergingTechnologies/EmergingTechnologies/blob/main/Images/Screen%20Shot%202020-12-11%20at%2011.55.55%20PM.png)

## Deployed Web application and predictions for each label[Strong negative, Negative, Neutral, Positive, Strong positive]

![alt text](https://github.com/TheExpendablesEmergingTechnologies/EmergingTechnologies/blob/main/Images/Webpage_1.png)
![alt text](https://github.com/TheExpendablesEmergingTechnologies/EmergingTechnologies/blob/main/Images/Webpage_2.png)
![alt text](https://github.com/TheExpendablesEmergingTechnologies/EmergingTechnologies/blob/main/Images/Webpage_3.png)
![alt text](https://github.com/TheExpendablesEmergingTechnologies/EmergingTechnologies/blob/main/Images/Webpage_4.png)
![alt text](https://github.com/TheExpendablesEmergingTechnologies/EmergingTechnologies/blob/main/Images/Webpage_5.png)
![alt text](https://github.com/TheExpendablesEmergingTechnologies/EmergingTechnologies/blob/main/Images/Webpage_6.png)


## Project Report and Presentation Links

- [Project_Presentation](https://github.com/TheExpendablesEmergingTechnologies/EmergingTechnologies/tree/main/Documents)

- [Project_Report](https://github.com/TheExpendablesEmergingTechnologies/EmergingTechnologies/tree/main/Documents)

## Reference: 

[1] Bingel, J. and Søgaard, A. (2017). Identifying beneficial task relations for multi-task learning in deep neural networks. In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics, pages 164–169, Valencia, Spain. 

[2] Improving Sentiment Analysis with Multi-task Learning of Negation J E R EM Y B A R N E S , E R I K V E L L D A L , and L I L J A Ø V R E L I D Language Technology Group, University of Oslo email: {jeremycb,erikve,liljao}@ifi.uio.no 

[3] De Mattei, Lorenzo., Cimino, Andrea., & Dell’Orletta, Felice, “Multi-Task learning in deep neural network for sentiment polarity and irony classification”, in ILC-CNR 

[4] Majumder, Navonil., Chhaya, Niyati., Poria, Soujanya., Cambria, Erik., Peng, Haiyun., & Gelbukh, Alexander (Mar 2019), “Sentiment and Sarcasm classification with multitask learning” 

[5] Hu, M. and Liu, B. (2004). Mining opinion features in customer reviews. In Proceedings of the 10th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 168–177, Seattle, USA. 

[6] Taboada, M., Brooke, J., Tofiloski, M., Voll, K., and Stede, M. (2011). Lexicon-based methods for sentiment analysis. Computational Linguistics, 37(2):267–307. 

[7] Pang, B., Lee, L., and Vaithyanathan, S. (2002). Thumbs up? Sentiment classification using machine learning techniques. In Proceedings of the 2002 Conference on Empirical Methods in Natural Language Processing, pages 79–86, Philadelphia, USA. 

[8] Councill, I., McDonald, R., and Velikovich, L. (2010). What’s great and what’s not: learning to classify the scope of negation for improved sentiment analysis. In Proceedings of the Workshop on Negation and Speculation in Natural Language Processing, pages 51–59, Uppsala, Sweden. 

[9] Cruz, N. P., Taboada, M., and Mitkov, R. (2016). A machine-learning approach to negation and speculation detection for sentiment analysis. Journal of the Association for Information Science and Technology, 67(9):2118–2136. 

[10] Lapponi, E., Velldal, E., Øvrelid, L., and Read, J. (2012b). UiO2: Sequence-labeling negation using dependency features. In Proceedings of the First Joint Conference on Lexical and Computational Semantics, pages 319–327, Montreal, Canada. 

[11] Morante, R. and Blanco, E. (2012). *SEM 2012 shared task: Resolving the scope and focus of negation. In Proceedings of the First Joint Conference on Lexical and Computational Semantics (*SEM), pages 265–274, Montr´eal, Canada. 

[12] Read, J., Velldal, E., Øvrelid, L., and Oepen, S. (2012). UiO1: Constituent-based discriminative ranking for negation resolution. In Proceedings of the First Joint Conference on Lexical and Computational Semantics (*SEM), Montreal, Canada. 

[13] Qian, Z., Li, P., Zhu, Q., Zhou, G., Luo, Z., and Luo, W. (2016). Speculation and negation scope detection via convolutional neural networks. In The 2016 Conference on Empirical Methods in Natural Language Processing. 

[14] Fancellu, F., Lopez, A., Webber, B., and He, H. (2017). Detecting negation scope is easy, except when it isn’t. In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics, pages 58–63, Valencia, Spain. 

[15] Velldal, E., Øvrelid, L., Read, J., and Oepen, S. (2012). Speculation and negation: Rules, rankers, and the role of syntax. Computational Linguistics, 38(2):369–410. 
