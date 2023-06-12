# The-Arabic-Phoenix
Hate speech on social media is a problem that we are facing in middle eastern countries, and in
the absence of surveillance the problem is on the rise, we are willing to train a
model which will be able to detect and tackle this problem. Arabic is the fourth most used
language on the internet and ranked as the sixth on Twitter.
The Arabic Phoenix is our graduation project that aims to detect Hate Speech in Arabic using Machine Learning.

### Why did you choose this project?
We chose it because we wanted to make something useful. We see this problem everyday and we thought that making an AI to fight this problem would be nice.

### How to use the code?
You can reproduce the whole experiment by downloading the notebook 'MARBERT_Final.ipynb' or 'Statistics_MARBERT.ipynb' and uploading them to google co-lab, or you can download all the dependencies using the requirements.txt file (cat requirements.txt | xargs -n 1 pip install).

### Why MARBERT?
After many experiments, fine-tuning MARBERT gave the best results and the improvement was very noticeable compared to other algorithms and architectures.

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
