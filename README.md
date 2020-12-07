# Improving Sentiment Analysis with Mutli-Task Learning of Negation

Sentiment Analysis is a process where one can mine people’s opinions from a piece of text. At first glance, this task may look like a simple text classification problem, but once deep dived, one can find out challenges which can affect the sentiment analysis accuracy. For instance, predicting sentiment just using the words in a sentence can lead to major pitfalls such as detecting sarcasm, irony, word ambiguity, use of negating words, also multipolarity. Out of all these phenomena, negation is the most prevalent. Any sentiment analysis model must be able to identify negation and try to remove the effect that its scope has on the final sentiment of a text. In this project, we propose to use Multi-task learning approach, using a cascading and hierarchical neural architecture of LSTM layers. This architecture will explicitly train the model with negation as an auxiliary task and eventually help in improving the main task of sentiment analysis. 

## Model Architecture

![alt text](https://github.com/TheExpendablesEmergingTechnologies/EmergingTechnologies/blob/main/arch.png)

## Models

- Single-task model
- Multi-task SFU
- Multi-task CD

## Datasets

- [SST](https://nlp.stanford.edu/sentiment/treebank.html)
- [Conan Doyle Neg (*Sem 2012)](https://www.clips.uantwerpen.be/sem2012-st-neg/)


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
