

# PGxPlus

This is the official repository for paper: Toward Precision Psychiatry: Integrating Pharmacogenomics and Multi-dimensional Clinical Data for Antidepressant Response Prediction

---
## 1️⃣ Data Preparation

### **Input Data Sources**

The input information of each patient should at least contain following : 

- **Genomic Data**: Single nucleotide polymorphisms identified through pharmacogenomic testing.
- **Demographics Data**: 
  - Age: Patient’s chronological age at the time of diagnosis.
  - Gender: Biological sex assigned at birth.
  - BMI: Body mass index calculated from height and weight.
  - Smoke: Smoking status.
  - Drink: Alcohol consumption status.
  - Education: Highest educational attainment.
  - Years of illness: Duration of depressive symptoms since initial onset.
  - Family history: Presence of  psychiatric disorders in first-degree relatives.
  - Course pattern: Clinical course type (e.g.,  episodic, chronic).
  - Trigger factor: Identified  psychosocial  or  biological stressors preceding the onset.
- **Laboratory Test Data** :
  - CRP
  - TG
  - ALP
  - AST
  - ALT
  - DBIL
  - TBIL
  - PT
  - ALB
- **Comorbidity History**:
  - Mental disorders
  - Neural disorders
  - Cardiovascular disorder
  - Diabetes
  - Liver related disorders
  - Kidney related disorders
  - Tumor
  - Epilepsy
- **Medication history**: Prior use of any antidepressant medications.
- **HAMD evaluation **: Hamilton  Depression  Rating  Scale score at baseline,  including  score  for each  item,  reflecting  initial  symptom severity.

### Creating Pseudo-Label

Leveraging empirical Drug–Gene association statistics to infer likely treatment responses as pseudo-label. For example:

| SNP       | Subtype | Sertraline | Escitalopram | Fluoxetine | Paroxetine | Fluvoxamine |
| --------- | ------- | ---------- | ------------ | ---------- | ---------- | ----------- |
| SLC6A4    | S/S     | 0.539379   | 0.353147     | 0.410256   | 0.2625     | 0.441667    |
| SLC6A4    | L/L     | 0.533333   | 0.425        | 0          | 0          | 0           |
| SLC6A4    | S/L     | 0.526667   | 0.456067     | 0.472727   | 0.285714   | 0.455556    |
| SLC6A4    | S/XL    | 0.428571   | 0.461538     | 0          | 0          | 0           |
| rs9316233 | C/G     | 0.551136   | 0.409639     | 0.474576   | 0.1875     | 0.419643    |
| rs9316233 | C/C     | 0.515464   | 0.38961      | 0.439024   | 0.309524   | 0.46789     |
| rs9316233 | G/G     | 0.47619    | 0.4375       | 0          | 0          | 0           |
| rs8192709 | C/T     | 0.604651   | 0.465116     | 0          | 0          | 0           |
| rs8192709 | C/C     | 0.523684   | 0.395722     | 0.445946   | 0.285714   | 0.444444    |

The value associated with a SNP and a drug account for the drug response rate on the single SNP. For example, the value associated with ‘SLC6A4-S/S’ and ‘Sertraline’ is 0.539. It indicates that according to the statistics, venlafaxine is effective for patients with the SLC6A4-S/L gene at a rate of 53.9%. The drug response rate is calculated as the proportion of patients possessing a specific genotype and treated with a particular drug who achieved an effective clinical response.

You can assign the pseudo-label to each patient according to his/her genomic data. 

After completing the above preparations, run the following command:

```python
python prepare_data.py --input_path 'your_input_file_path' --output_path 'your_target_file_path'
```

## 2️⃣ Model Training & Inference

Run the following command to perform training. 

```python
python main.py --dataset 'your_target_file_path'
```

For weak-supervised training, reassign the pseudo-label after first round training, and run the second round training.

Run the following command to perform inference.

```python
python test.py --data_path 'your_data_path' --model_path 'your_checkpoint_path'
```

## 3️⃣ Interpretability Analysis

We applied **SHAP (SHapley Additive exPlanations)** to interpret the model's decision-making process.

Run the following command to perform SHAP-based interpretability analysis.

```python
python shap_analysis.py --data_path 'your_data_path' --model_path 'your_checkpoint_path' 
```
