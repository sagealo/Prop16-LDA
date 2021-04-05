import nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.collocations import *

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

# Filter out stop_words and non alphanumeric words for the sentences
filtered_vpsm_sent = []
filtered_vam_sent = []

for sentence in vpsm_sent_tokens:
    sentence_arr = sentence.split()
    new_sentence = ''
    for word in sentence_arr:
        if word.isalpha() or word.isnumeric():
            new_sentence += word + ' '
    filtered_vpsm_sent.append(new_sentence)

for sentence in vam_sent_tokens:
    sentence_arr = sentence.split()
    new_sentence = ''
    for word in sentence_arr:
        if word.isalpha() or word.isnumeric():
            new_sentence += word + ' '
    filtered_vam_sent.append(new_sentence)




# Analyze Frequency distribution 
vpsm_fdist = FreqDist(filtered_vpsm)
vam_fdist = FreqDist(filtered_vam)

# Write frequencies to respective files
vpsm_freqs = open('VotingPilotStatusManipulationFrequencies.txt', 'w')
for key in vpsm_fdist:
    s = key + ': ' + str(vpsm_fdist[key]) + '\n'
    #vpsm_freqs.write(s)

vam_freqs = open('VotingAsianMergedFrequencies.txt', 'w')
for key in vam_fdist:
    s = key + ': ' + str(vam_fdist[key]) + '\n'
    #vam_freqs.write(s)

# TODO Identify negative vs positive words and their respective frequencies. (Not Yet, after bin placements)
# TODO Word occuring together/in-tandem into 5, 10, 15 bins. Ways to determine similarity in statements. 

# Create the collocations
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()

# Create the finders based on the n-gram being used
finder_vpsm = TrigramCollocationFinder.from_words(filtered_vpsm)
finder_vam = BigramCollocationFinder.from_words(filtered_vam_sent)

# Apply frequency filter where the parameter is the minimum number of times the 
# n-gram appears for it to be included
finder_vpsm.apply_freq_filter(5)

# Retrieve the pairings for the nbest with the desired pmi
vpsm_ngrams = finder_vpsm.nbest(trigram_measures.pmi, 100)
vam_ngrams = finder_vam.nbest(bigram_measures.pmi, 10)

print(vpsm_ngrams)

# Close the files
voting_pilot_status_manipulation.close()
voting_asian_merged.close()
vpsm_freqs.close()
vam_freqs.close()



