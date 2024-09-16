# Classification of Titles of Youtube Videos/Google Search into Educational content or Random
---
## Problem:
Numerous studies have shown that internet distraction has a negative effect on student/professionals learning and performance.
At the same time use case of interent is so immense that you need it to complete your task too, that's why i tried to make a 
solution so that student/professional (currently only of cs-background) can use internet while work without getting distracted.

---
## Solution:
1. Idea is to make a Deep learning model to classify content using its title, whether it is eductaional/learning content or not;
2. Integrate it with a browser so that if you click on any content (while work mode) it classify its title as tut or not , and allow to open only learning content, if work mode in browser is on.

---
## Execution:
I finetuned a bert-base-uncased and initially only for computer science background.
##### Why finetuning and Why BERT ?
1. Machine learning is out of discussion due to complexity of problem, requireing semantic understanding of text. Among deep learning llm's will have best performance, and rather than training a llm from scratch ( not even possible for me) i opted finetuning an exisiting llm and evaluate its performance . 
2. Bert is most popular and easily accessible among all open source llm's.

## Dataset:
I scrapped youtube video titles and google search results using youtube v3-api and google custom search engine api, using python script. Then seperated them using various techniques into tutorial and random titles, named tut.csv and ran.csv.

Further i have generated clickbait titles as it's frequency is compratively less in dataset using gen ai model chat-gpt 4o, stored in file name clickbait.csv other than that i have a old dataset containing all random texts named random_text.csv.
All four csv files stored in data folder.

## Process & Results:
Since the dataset was highly imbalanced in favour of tutorial titles (as i scrapped majorly channels related to computer science background), so i used 
1. SMOTE
2. Undersampling technique
for balancing dataset.

---
###### ML Results: Decided to use Xgboost + TfidfVectorizer + SMOTE just for comparison purpose with deep learning result:
- Without balancing dataset using SMOTE: 
  - accuracy score: 86.93513763212957
  - precision (class 0): 35.98380800925256
  - precision (class 1): 98.72841654397003

- After balancing dataset using SMOTE:
  - accuracy score: 85.41813188653337
  - precision (class 0): 87.45692395195557
  - precision (class 1): 83.37523886151061

---
###### DL Results: Finetuning with SMOTE
Problem while applying Smote:
1. Smote produces result using averageing which can be decimal too => round the integer to nearest
2. Bert model expect word encodeing i.e input_ids to be in a certain range (vocab size) => clipping to vocab size
3. Attention mask denote whether the token is padding or not , => applying smote on input id only not on attention mask.

```python
inputs = tokenizer(X , padding=True, truncation=True)
#seperateing input ids and attention mask
input_ids = inputs['input_ids']    #list 
attention_mask = inputs['attention_mask']    #list

# createing smote object
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(input_ids, y)

#clipping input id in range of vocab size
input_ids_resampled = np.clip(np.round(X_resampled), 0, tokenizer.vocab_size - 1).astype(int)

# createing attention mask
attention_mask_resampled = (input_ids_resampled != 0).astype(int)

#merging input_ids and attention mask for creating input
inputs_resampled = {
    'input_ids': input_ids_resampled,
    'attention_mask': attention_mask_resampled
}
```

After balancing dataset using SMOTE:
- Class 0:
  - Precision: 0.90
  - Recall: 0.85
  - F1-Score: 0.87
  - Support: 33691
- Class 1:
  - Precision: 0.82
  - Recall: 0.88
  - F1-Score: 0.85
  - Support: 26021

- Overall Accuracy: 0.86

---
###### DL Results: Finetuning with Undersampling

Problem while applying Undersampling:
1. In the minority sample of tut.csv , proportion of various text do not remain same as original full data => below given steps

```python
# Calculate the length of each text
dft['text_length'] = dft['video_titles'].apply(lambda x: len(x.split()))

# Extract lengths and number of samples
lengths = (dft['text_length'].value_counts()).index
num_samples = (dft['text_length'].value_counts()).values

#Number of samples of various length to be taken , if total samples to be taken is 40000
text_length_distribution = dft['text_length'].value_counts(normalize=True)
num_samples_per_length = (text_length_distribution * 40000).astype(int)

sampled_df = pd.DataFrame(columns=dft.columns)  # Ensure the columns are consistent

# Perform stratified sampling for each text length
for length, num_samples in num_samples_per_length.items():
    # Filter the DataFrame for rows with the current text length
    subset_df = dft[dft['text_length'] == length]
    
    # Randomly select the required number of samples
    sampled_subset_df = subset_df.sample(n=num_samples, random_state=42)
    
    # Append the sampled subset to the sampled DataFrame
    sampled_df = pd.concat([sampled_df, sampled_subset_df])

# Drop the text_length column as it's no longer needed
sampled_df.drop(columns='text_length', inplace=True)

# Reset the index of the sampled DataFrame
sampled_df.reset_index(drop=True, inplace=True)
```
After balancing dataset using SMOTE:
- Class 0:
  - Precision: 0.95
  - Recall: 0.96
  - F1-Score: 0.96
  - Support: 6880
- Class 1:
  - Precision: 0.96
  - Recall: 0.96
  - F1-Score: 0.96
  - Support: 8056

- Overall Accuracy: 0.96

## Discussions:
Undersampling with finetuning performed best, while llm was expected to outperform ml algorithms, the underperformance of SMOTE technique with llm is likely due to , we cannot say by sure that arthimetic mean of input ids of 2 words produce an input id also having similar meaning, and this relation hold even for sentences, i.e linear combination of representation of 2 sentence produces a sentence with similar result.  While a larger dataset can also be attributed for failure 3,00,000 samples as opposed to 80,000 in undersampling, but it seems unlikely that a 110 million parameters fails at just 3,00,000 rows /samples.

## Failure and Success:
![image](https://github.com/nirju123/Tut_or_Not/blob/main/output/failed1.png)
![image](https://github.com/nirju123/Tut_or_Not/blob/main/output/failed2.png)
![image](https://github.com/nirju123/Tut_or_Not/blob/main/output/failed3.png)
![image](https://github.com/nirju123/Tut_or_Not/blob/main/output/success1.png)
![image](https://github.com/nirju123/Tut_or_Not/blob/main/output/success1.png)









