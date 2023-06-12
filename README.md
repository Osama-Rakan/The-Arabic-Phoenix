# Arabic-Phoenix
Hate speech on social media is a problem that we are facing in middle eastern countries, and in
the absence of surveillance the problem is on the rise, we are willing to train a
model which will be able to detect and tackle this problem. Arabic is the fourth most used
language on the internet and ranked as the sixth on Twitter.
Arabic Phoenix is our graduation project that aims to detect Hate Speech in Arabic using Machine Learning.

### Where can I download your model?
From the following link:
https://drive.google.com/file/d/14wP9u1UkbAE-J-PXDYSr9R2fctauX7da/view?usp=sharing

### Why did you choose this project?
We chose it because we wanted to make something useful. We see this problem everyday and we thought that making an AI to fight this problem would be nice.

### How to use the code?
You can reproduce the whole experiment by downloading the notebook 'MARBERT_Final.ipynb' or 'Statistics_MARBERT.ipynb' and uploading them to google co-lab, or you can download all the dependencies using the requirements.txt file (cat requirements.txt | xargs -n 1 pip install).

### Why MARBERT?
After many experiments, fine-tuning MARBERT gave the best results and the improvement was very noticeable compared to other algorithms and architectures.

Important information about [UBC-NLP/MARBERTv2](https://huggingface.co/UBC-NLP/MARBERTv2):

Data source: Arabic Twitter

Number of tokens: 15.6B

Tokenizer: Word-Piece tokenizer

Vocab size: 100K

Architecture: BERT base (12 encoders only)


### What are the main things that improved the model?
There are two things:
1) Keeping the emojis and replacing them with meaningful words was key.
2) Changing the size of the allowed max sequence.

### Why did not you collect all the data?
Collecting the data was very difficult in the time frame we were given, so we decided to collect some data to increase the unseen data (test data) and merge multiple datasets from multiple sources.

### Will you host the Flask website?
Hopefully in the future.

### What have you learned from this project?
The main thing we learned that collecting data is one of the hardest things in our major that we did not experience in our time in the University before the graduation project.

### What is next?
We would like to take data from any company that needs to remove hate speech from their platform and build a custom model based on their data, that would be awesome.

### What parameters did you use?
We created a one-layer feed-forward Neural Network (FFNN). Hidden size of MARBERT is 768, hidden size of the Feed Forward Neural Network (FFNN) is 50, and the output size is 2 (since we have 2 labels: HS and not_HS), and batch size = 64. We used Dropout layer for regularization (0.5). We tried to add more hidden layers but they caused a very big overfit, so we kept it as simple as that.
We tried many values for the Learning Rate in Adam 
optimizer and the best one was 0.00005 with small number of epochs, greater Learning Rate with more 
epochs seemed to be really worse, and smaller Learning Rate with the same number of epochs or less 
seemed to be worse too. The epsilon value we chose was 0.00000001.

### How does the website work?
![Blank diagram (3)](https://github.com/Osama-Rakan/The-Arabic-Phoenix/assets/78223597/99d73605-cefa-4ce5-b10f-882ae0d2e517)

### On what did you base your decisions?
We based our decisions based on statistics related to the data.
![Blank diagram (4)](https://github.com/Osama-Rakan/The-Arabic-Phoenix/assets/78223597/1775b7c5-b27d-498c-af6b-7e6357fd8158)

### Can you show an example of using statistics to make decisions from your project?
Emojis counts in the hate speech documents:
{'😂': 655, '😷': 469, '😡': 290, '🐸': 188, '😤': 139, '💔': 129, '👎': 121, '🔪': 120, '👎🏻': 108, '👊': 108, '👞': 106, '😠': 97, '🤣': 96, '🤢': 93, '🐑': 89, '👊🏻': 83, '💩': 79, '🤮': 63, '🐕': 54, '😭': 51, '😣': 51, '👎🏼': 51, '🙂': 44, '😒': 41, '👇': 41, '🐏': 40, '🤔': 34, '❌': 34, '🖕': 33, '😅': 31, ' 🏻': 31, '😏': 29, '👌': 27, '🐒': 27, '🔥': 27}

Which tells us that we should not remove the emojis completely from the corpus and we should replace them:
1.	😍❤️❤💜💙🖤💓💗💚💝💘💖💕🤍💛❣️💞🌹🥰💟💑-> حبحب
2.	🐶🐕🐷🐖🐴🐄🐮🐂🐃🐵🐒🙉🐑🐐🐸🦄 -> حيواناتحيوانات
3.	😡🤬😠😤🤮🤢😣😷😒🙄 -> معصبمعصب
4.	👊👊🏽👊🏻👊🏼👊🏾👊🏿🔪 -> عنفعنف
5.	✋✋🏽✋🏻✋🏿✋🏼✋🏾 -> يديد
6.	🖕🖕🏽🖕🏻🖕🏿🖕🏼🖕🏾 -> الوسطىالوطسى
7.	😂🤣😭💀 -> ضحكضحك
8.	💩👠👞 -> عدماحترامعدماحترام
9.	😢💔 -> حزينحزين
10.	🤔 -> يفكريفكر
11.	🔥 -> نارنار
12.	👎 -> عدماعجابعدماعجاب

### What is the most important metric for you and why??
The AUC, Recall, and F1-score, but we stil want a good value for all the metrics. Recall being big is a good thing because it reduces the type II error which is very dangerous in our case (Hate speech documents being predicted as not hate speech).

### What are the best metrics you got?
![image](https://github.com/Osama-Rakan/The-Arabic-Phoenix/assets/78223597/66fb6c88-3721-412e-9713-b821094319e5)

