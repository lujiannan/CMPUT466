# ML Project Information
Dataset: **Balance scale weight & distance database**
Algorithms: Logistic Regression, SVM, KNN
Required Libraries: scikit-learn, pandas, numpy, matplotlib

Dataset information:
This data set was generated to model psychological experimental results.  Each example is classified as having the balance scale tip to the right, tip to the left, or be balanced.  The attributes are the left weight, the left distance, the right weight, and the right distance.  The correct way to find the class is the greater of  (left-distance * left-weight) and (right-distance * right-weight).  If they are equal, it is balanced. (No missing values) (Classification task)
(4, Categorical) Features : left-distance   (1, 2, 3, 4, 5)
                            left-weight     (1, 2, 3, 4, 5)
                            right-distance  (1, 2, 3, 4, 5)
                            right-weight    (1, 2, 3, 4, 5)
(1, Categorical) Target   : class name      (L, B, R)
Number of Instances: 625 (49 balanced, 288 left, 288 right)

Instruction:
terminal$ python proj.py **OR** run script=proj.py

Output:
Numerical output is written in the very beginning of the file proj.py
graphical output is generated into the current folder where proj.py is