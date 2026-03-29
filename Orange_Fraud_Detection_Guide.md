# Orange Data Mining Workflow: Credit Card Fraud Detection

Now that you have your perfectly balanced dataset (`creditcard_balanced.csv`), here is exactly how you should build your Orange workflow canvas step by step to get top-tier results for your assignment/project.

---

### Step 1: Loading the Data
1. Drag the **File** (or **CSV File Import**) widget onto your blank canvas.
2. Double-click it and load `C:\Users\ShyamVenkatraman\Desktop\FDA\creditcard_balanced.csv`.
3. **CRITICAL CONFIGURATION:** 
   - Scroll down to the `Class` column.
   - Change its **Type** to `Categorical`.
   - Change its **Role** to `Target`.
   - Ensure you click "Apply". If you don't do this, the models won't know what they are supposed to be predicting!

### Step 2: Visualization (Optional but great for reports)
1. Connect a **Data Table** widget to the File widget just to visually confirm the data looks good.
2. Connect a **Distributions** widget to the File widget. 
   - Select `Class` on the left. You should see a perfectly even split (473 for Class 0, 473 for Class 1) proving the data is balanced.

### Step 3: Preparing the Models
Drag the following model widgets onto your canvas (Do not connect them to the File widget directly):
1. **Random Forest**: Generally the absolute best model for out-of-the-box fraud detection.
2. **Logistic Regression**: A great, fast baseline model.
3. **Neural Network** or **XGBoost/Gradient Boosting**: To show advanced modeling.

### Step 4: Testing & Scoring (The Core)
1. Drag a **Test and Score** widget onto the canvas.
2. Draw a line from the **File** widget to the **Test and Score** widget `(Data -> Data)`.
3. Draw lines from **each of your Models** into the **Test and Score** widget `(Learner -> Learner)`.
4. Double-click **Test and Score**. 
   - Select **Cross-validation** (5 or 10 folds is standard).
   - Look at the scoreboard on the right. 
   - Pay attention to **AUC**, **F1**, and **Precision/Recall**. Since the classes are 50/50 now, your Classification Accuracy (CA) is highly trustworthy!

### Step 5: The Confusion Matrix
1. Drag a **Confusion Matrix** widget onto the canvas.
2. Connect the **Test and Score** widget to the **Confusion Matrix** `(Evaluation Results -> Evaluation Results)`.
3. Double-click the Confusion Matrix. 
   - Look at how many actual Frauds (`1`) were accidentally predicted as Legitimate (`0`). This is your **False Negative** rate, and in fraud detection, this is the most critical number to minimize!

### Step 6: Feature Ranking (Extra Credit)
1. Drag a **Rank** widget and connect your **File** widget to it `(Data -> Data)`.
2. This ranks all the anonymized variables (`V1` through `V28`) from best to worst. This proves to your professor that you understand which features mathematically drive fraud the most, even if they are anonymized!
