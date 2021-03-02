# DBRF<br>
------
This repository contains code for the paper:<br>
###DBRF: a density-based algorithm for improving the recall of minority class with Random Forest<br>
[QQian](#), [Jia Dong](#)<br>
[[OpenReview](#)][[PDF](#)]<br>
Shanghai University<br>

-----
##Abstract
Aiming at the problem of insufficient prediction accuracy for minority class in imbalanced classification, this paper proposes a density-based method to improve the accuracy of minority class prediction, in which additional training is carried out for minority classes and boundary samples through ensemble learning methods. The purpose is to increase the diversity of base learners in ensemble learning to improve the classification accuracy of minority classes. The experimental results prove that the DBRF algorithm can improve the prediction accuracy of minority classes within certain conditions, but it will lose a small amount of overall accuracy. The cost of losing a small amount of overall accuracy in practical applications will be smaller than the cost of a minority of misclassifications e.i.the experimental results are of practical significance.



##Requirements<br>
* Python 3.7.6
* json 2.0.9
* joblib 0.16.0
* sklearn 0.23.2
* pandas 1.0.5
* imblearn 0.7.0
<br>

----
##Dataset<br>
* [UCI](#)
    * [Haberman's Survival Data](#)
    * [Car Evaluation Database](#)
    * [Abalone data](#)
* Full_Dataset-Dmax-TTT<br>

---


```python
python BorderlineDBSCAN.py

```
