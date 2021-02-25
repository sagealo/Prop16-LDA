import nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords

# Read in the files
voting_pilot_status_manipulation = open('LDA_files/CA_prop_16/zsbE_LDA_voting_pilot_status_manipulation.csv', 'r')
voting_asian_merged = open('LDA_files/CA_prop_16/zsbE_LDA_voting_wh_asian_merged.csv', 'r')
# Read files into a string
vpsm_string = voting_pilot_status_manipulation.read().lower()
vam_string = voting_asian_merged.read().lower()

# Tokenize the string into words and sentences
vpsm_tokens = word_tokenize(vpsm_string)
vpsm_sent_tokens = sent_tokenize(vpsm_string)
vam_tokens = word_tokenize(vam_string)
vam_sent_tokens = sent_tokenize(vam_string)

# Part of Speech Tags of VPSM
vpsm_tags = nltk.pos_tag(vpsm_tokens)
vam_tags = nltk.pos_tag(vam_tokens)

# Filter out stop_words and non alphanumeric words
stop_words = set(stopwords.words('english'))
filtered_vpsm = []
filtered_vam = []

for word in vpsm_tokens:
    if word not in stop_words and word.isalpha():
        filtered_vpsm.append(word)

for word in vam_tokens:
    if word not in stop_words and word.isalpha():
        filtered_vam.append(word)

# Analyze Frequency distribution 
vpsm_fdist = FreqDist(filtered_vpsm)
vam_fdist = FreqDist(filtered_vam)

# Write frequencies to respective files
vpsm_freqs = open('VotingPilotStatusManipulationFrequencies.txt', 'w')
vam_freqs = open('VotingAsianMergedFrequencies.txt', 'w')
for key in vpsm_fdist:
    s = key + ': ' + str(vpsm_fdist[key]) + '\n'
    vpsm_freqs.write(s)

for key in vam_fdist:
    s = key + ': ' + str(vam_fdist[key]) + '\n'
    vam_freqs.write(s)

# TODO Identify negative v. positive words and their respective frequencies. 

# Close the files
voting_pilot_status_manipulation.close()
voting_asian_merged.close()
vpsm_freqs.close()
vam_freqs.close()



