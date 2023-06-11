import re
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import nltk
from nltk.tokenize import RegexpTokenizer
nltk.download('popular', quiet=True)
import transformers
import torch
device = torch.device("cpu")
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from collections import Counter
import string
import emoji
from wordcloud import WordCloud
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image
from time import sleep

# https://stackoverflow.com/questions/34122949/working-outside-of-application-context-flask

MAX_LEN = 150

tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-mini-arabic")

class TextCleaner:
  def __init__(self, remove_links=True, remove_mentions_and_hashtags=True, 
               replace_emojis_with_meaningful_tokens=True, remove_emojis=False, remove_consecutive_duplicate_letters=True,
               remove_numbers=True, remove_punctuation=True, only_arabic_chars=False, remove_stop_words=True, letters_normalization=True, remove_j_shift= True):
    self.remove_links = remove_links
    self.remove_mentions_and_hashtags = remove_mentions_and_hashtags
    self.replace_emojis_with_meaningful_tokens = replace_emojis_with_meaningful_tokens
    self.remove_emojis = remove_emojis
    self.remove_consecutive_duplicate_letters = remove_consecutive_duplicate_letters
    self.remove_numbers = remove_numbers
    self.remove_punctuation = remove_punctuation
    self.only_arabic_chars = only_arabic_chars
    self.remove_stop_words = remove_stop_words
    self.letters_normalization = letters_normalization
    self.remove_j_shift = remove_j_shift

  
  def clean_text(self, document):
    emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                              "]+", flags=re.UNICODE)
    
    def delete_consecutive_strings(s):
        i=0
        j=0
        new_elements=''
        while(j<len(s)):
            if( s[i]==s[j] ):
                j+=1
            elif((s[j]!=s[i]) or (j==len(s)-1)):
                new_elements+=s[i]
                i=j
                j+=1
        new_elements+=s[j-1]
        return new_elements
    # Remove URLs
    if self.remove_links:
      document = re.sub('http\S+', '', document)

    # Remove mentions and hashtags
    if self.remove_mentions_and_hashtags:
      document = re.sub('[@#]\w+', '', document)

    # Replacing emojis with meaningful tokens
    if self.replace_emojis_with_meaningful_tokens:
      document = re.sub('[😍❤️❤💜💙🖤💓💗💚💝💘💖💕🤍💛❣️💞🌹🥰💟💑]', ' حبحب ', document)
      document = re.sub('[🐶🐕🐷🐖🐴🐄🐮🐂🐃🐵🐒🙉🐑🐐🐸🦄]', ' حيواناتحيوانات ', document)
      document = re.sub('[😡🤬😠😤🤮🤢😣😷😒🙄]', ' معصبمعصب ', document)
      document = re.sub('[👊👊🏽👊🏻👊🏼👊🏾👊🏿🔪]', ' عنفعنف ', document)
      document = re.sub('[✋✋🏽✋🏻✋🏿✋🏼✋🏾]', ' يديد ', document)
      document = re.sub('[🖕🖕🏽🖕🏻🖕🏿🖕🏼🖕🏾]', ' الوسطىالوسطى ', document)
      document = re.sub('[😂🤣😭💀]', ' ضحكضحك ', document)
      document = re.sub('[💩👠👞]', ' عدماحترامعدماحترام ', document)
      document = re.sub('[😢💔]', ' حزينحزين ', document)
      document = re.sub('🤔', ' يفكريفكر ', document)
      document = re.sub('🔥', ' نارنار ', document)
      document = re.sub('👎', ' عدمإعجابعدمإعجاب ', document)
      #document = emoji_pattern.sub(' إيموجيإيموجي ', document)


    # Remove emojis
    if self.remove_emojis:
      document = emoji_pattern.sub('', document)

    # Remove consecutive duplicate letters
    if self.remove_consecutive_duplicate_letters:
      try:
        document = delete_consecutive_strings(document)
      except:
        pass

    # Remove numbers
    if self.remove_numbers:
      document = re.sub('\d+', '', document)

    # Remove punctuation marks
    if self.remove_punctuation:
      punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~،؟!.؛'
      document = document.translate(str.maketrans('', '', punctuation))


    # # Remove stop words
    if self.remove_stop_words:
      stop_words = nltk.corpus.stopwords.words('arabic')
      stopwords_dict = Counter(stop_words)
      document = ' '.join([word for word in document.split() if word not in stopwords_dict])

    # # Only keep Arabic characters
    if self.only_arabic_chars:
      document = re.sub('[^\[\]_ء-ي]', ' ', document)

    # Normalize arabic letters
    if self.letters_normalization:
      document = re.sub('[أإءئى]', 'ا', document)
      document = re.sub('ص', 'س', document)
      document = re.sub('ظ', 'ض', document)

    # # Remove ـ
    if self.remove_j_shift:
      document = re.sub('ـ', '', document)

    # # Replacing the multiple spaces between words with 1 space
    document = re.sub(' +', ' ', document)
    document = document.strip()
    return document

  def preprocessing_for_bert(self, data, text_preprocessing_fn=clean_text, MAX_LEN=137):
    input_ids = []
    attention_masks = []
    tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERTv2")

    for sent in data:
        encoded_sent = tokenizer.encode_plus(text=text_preprocessing_fn(self,sent), add_special_tokens=True, 
                                              max_length=MAX_LEN, padding='max_length', return_attention_mask=True, truncation = True)
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))
    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks

import torch.nn as nn
class BertClassifier(nn.Module):
    def __init__(self, freeze_bert=False):
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        hidden_BERT = 768
        hidden_FFNN = 50
        labels_num = 2 # Normal=0, Hate Speech=1

        # Instantiate BERT model
        self.bert = AutoModel.from_pretrained("UBC-NLP/MARBERTv2")
        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_BERT, hidden_FFNN),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_FFNN, labels_num)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits


class TextStatistics:
  def __init__(self, cleaning_flag=False, show_word_cloud_chart=True, show_vocab_counts=True, show_sequence_statistics=True, 
               show_emoji_counts=True):
    self.cleaning_flag = cleaning_flag
    self.show_word_cloud_chart = show_word_cloud_chart
    self.show_vocab_counts = show_vocab_counts
    self.show_sequence_statistics = show_sequence_statistics    
    self.show_emoji_counts = show_emoji_counts
  
  def show_statistics(self, df):
    df = df.copy()
    if self.cleaning_flag:
      c1 = TextCleaner(remove_links=True, remove_mentions_and_hashtags=True, 
                       replace_emojis_with_meaningful_tokens=True, remove_emojis=False, remove_consecutive_duplicate_letters=True,
                       remove_numbers=True, remove_punctuation=True, only_arabic_chars=True, remove_stop_words=True, 
                       letters_normalization=True, remove_j_shift= True)
    
    tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERTv2")

    vocab = set()
    all_lengths = []
    max_seq_size = -1
    for sentence in df['text']:
      if self.cleaning_flag:
        sentence = c1.clean_text(sentence)
      tokens = tokenizer(sentence)['input_ids']
      all_lengths.append(len(tokens))
      if len(tokens) > max_seq_size:
        max_seq_size = len(tokens)
      
      for word in tokens:
        vocab.add(word)
    
    try:
        if self.show_word_cloud_chart:
            df_clean_for_word_cloud_chart = df.copy()
            c2 = TextCleaner(only_arabic_chars=True)
            for i in range(len(df_clean_for_word_cloud_chart)):
                df_clean_for_word_cloud_chart['text'].iloc[i] = c2.clean_text(df_clean_for_word_cloud_chart['text'].iloc[i])
            text = str(df_clean_for_word_cloud_chart['text'].values)
            mask = np.array(Image.open('C://Users//Osama_Rakan//Downloads//angry_mask.jpg'))
            reshaped_text = arabic_reshaper.reshape(text)
            bidi_text = get_display(reshaped_text)
            wordcloud = WordCloud(mask=mask, height=16, width=10, background_color='white', max_words=50, contour_color='#FF0000', contour_width=5, colormap='viridis', collocations=False, 
                                    font_path='C://Users//Osama_Rakan//Downloads//alfont_com_Janna-LT-Regular.ttf').generate(bidi_text)
            wordcloud.to_file('C://Users//Osama_Rakan//Downloads//login//static//files//word_cloud_chart.png')
    except:
        if self.show_word_cloud_chart:
            df_clean_for_word_cloud_chart = df.copy()
            c2 = TextCleaner(only_arabic_chars=True)
            for i in range(len(df_clean_for_word_cloud_chart)):
                df_clean_for_word_cloud_chart['text'].iloc[i] = c2.clean_text(df_clean_for_word_cloud_chart['text'].iloc[i])
            text = str(df_clean_for_word_cloud_chart['text'].values)
            text = 'لا يوجد أي كلمة في البيانات الخاصة بك'
            mask = np.array(Image.open('C://Users//Osama_Rakan//Downloads//angry_mask.jpg'))
            reshaped_text = arabic_reshaper.reshape(text)
            bidi_text = get_display(reshaped_text)
            wordcloud = WordCloud(mask=None, height=16, width=10, background_color='white', max_words=50, contour_color='#FF0000', contour_width=5, colormap='viridis', collocations=False, 
                                    font_path='C://Users//Osama_Rakan//Downloads//alfont_com_Janna-LT-Regular.ttf').generate(bidi_text)
            wordcloud.to_file('C://Users//Osama_Rakan//Downloads//login//static//files//word_cloud_chart.png')
            text = ''

    if self.show_vocab_counts:
      vocab_size = len(vocab)
    
    if self.show_sequence_statistics:
      mean1 = np.mean(all_lengths)
      median1 = np.median(all_lengths)
      max1 = np.max(all_lengths)
      min1 = np.min(all_lengths)
    
    if self.show_emoji_counts:
      all_emoji_counter = 0
      emojis_counts = {}
      import regex
      def emoji_count(text):
          emoji_counter = 0
          data = regex.findall(r'\X', text)
          for word in data:
              if any(char in emoji.EMOJI_DATA for char in word):
                  emoji_counter += 1
                  if word not in emojis_counts:
                    emojis_counts[word] = 1
                  else:
                    emojis_counts[word] += 1

          return emoji_counter

      for i in range(len(df)):
        all_emoji_counter += emoji_count(df['text'].iloc[i])
      emojis_counts = dict(sorted(emojis_counts.items(), key=lambda item: item[1], reverse=True))

      return vocab_size, all_emoji_counter, emojis_counts, mean1, median1, max1, min1


    
import pickle
modelPath = 'C://Users//Osama_Rakan//Downloads//MARBERT_Arabic_Pheniox_model.pt'
f = open(modelPath, 'rb')
bert_classifier = torch.load(f, map_location ='cpu')

import torch.nn.functional as F
def bert_predict(model, test_dataloader):
    model.eval()

    all_logits = []

    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    return probs



from flask import Flask, request, render_template, url_for, Response, send_from_directory, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
from flask import Flask, render_template, url_for, redirect
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt

import os
import glob






app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
app.config['SECRET_KEY'] = 'thisisasecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'


login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)


class RegisterForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Register')

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError(
                'That username already exists. Please choose a different one.')


class LoginForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Login')


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for('upload'))
    return render_template('login.html', form=form)


class UploadFileForm(FlaskForm):
    file = FileField('File', validators=[InputRequired()])
    submit = SubmitField('Upload your file!')

global_file_name = ''

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    files = glob.glob('C://Users//Osama_Rakan//Downloads//login//static//files//*')
    for f in files:
        os.remove(f)

    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data # First grab the file
        if file.filename[-4:] == '.csv' or file.filename[-5:] =='.xlsx':
            global global_file_name
            if file.filename[-4:] == '.csv':
                file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename('preds.csv'))) # Then save the file
            else:
               file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename('preds.xlsx')))
            try:
                df = pd.read_csv(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename('preds.csv')))
            except:
                df = pd.read_excel(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename('preds.xlsx')))
            
            if 'text' not in df.columns:
               return render_template('wrongInput.html')
            if df.empty:
               return render_template('wrongInput.html')
            df['text'] = df['text'].astype(str)
            new_data = df.text.values

            c1 = TextCleaner(remove_links=True, remove_mentions_and_hashtags=True, 
                             replace_emojis_with_meaningful_tokens=True, remove_emojis=True, remove_consecutive_duplicate_letters=True,
                             remove_numbers=True, remove_punctuation=True, only_arabic_chars=True, remove_stop_words=True, letters_normalization=True, remove_j_shift= True)
            
            new_inputs, new_masks = c1.preprocessing_for_bert(new_data)

            # Create the DataLoader for our test set
            new_dataset = TensorDataset(new_inputs, new_masks)
            new_sampler = SequentialSampler(new_dataset)
            new_dataloader = DataLoader(new_dataset, sampler=new_sampler, batch_size=32)

            probs = bert_predict(bert_classifier, new_dataloader)
            threshold = 0.50
            preds = []
            for i in range(len(probs)):
                if probs[i][1] >= threshold:
                    preds.append(1)
                else:
                    preds.append(0)
            df['probability_predictions'] = probs[:,1]
            df['integer_predictions'] = preds
            
            if file.filename[-4:] =='.csv':
                df.to_csv(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename('preds.csv')), index=False)
                global_file_name = 'preds.csv'
            elif file.filename[-5:] == '.xlsx':
               df.to_excel(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename('preds.xlsx')), index=False)
               global_file_name = 'preds.xlsx'

            s1 = TextStatistics(cleaning_flag=True, show_vocab_counts=True, show_sequence_statistics=True, show_word_cloud_chart=True)
            vocab_size, all_emoji_counter, emojis_counts, mean1, median1, max1, min1 = s1.show_statistics(df)

            return render_template('correctInput.html', word_cloud_chart='static//files//word_cloud_chart.png', 
                                   vocab_size=vocab_size, emoji_count=all_emoji_counter, each_emoji_count=emojis_counts,
                                   mean1=mean1, median1=median1, max1=max1, min1=min1)
        else:
            return render_template('wrongInput.html')
    return render_template('upload.html', form=form)



@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@ app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('register.html', form=form)

@app.route('/download')
@login_required
def download():
    return render_template('download.html', files=[global_file_name])

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory('C://Users//Osama_Rakan//Downloads//login//static//files', filename)

if __name__ == "__main__":
    app.run(debug=True)

#C://Users//Osama_Rakan//Downloads//login//static//files//gender.xlsx