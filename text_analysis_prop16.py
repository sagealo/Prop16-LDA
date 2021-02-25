import nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords


# Read in the file
voting_pilot_status_manipulation = open('LDA_files/CA_prop_16/zsbE_LDA_voting_pilot_status_manipulation.csv', 'r')
# Read file into a string
vpsm_string = voting_pilot_status_manipulation.read().lower()

# Tokenize the string into words and sentences
vpsm_tokenized = word_tokenize(vpsm_string)
vpsm_sent_tokenized = sent_tokenize(vpsm_string)

# Part of Speech Tags of VPSM
vpsm_tags = nltk.pos_tag(vpsm_tokenized)

# Set stop words 
stop_words = set(stopwords.words('english'))
# Filter out stop_words
filtered_vpsm = []
for word in vpsm_tokenized:
    if word not in stop_words and word.isalpha():
        filtered_vpsm.append(word)

print(filtered_vpsm)
# Analyze Frequency distribution 
vpsm_fdist = FreqDist(vpsm_tokenized)






