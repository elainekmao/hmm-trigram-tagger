import string
import math

###############QUESTION 4##################

#Initializes list to keep track of rare words
rare_words = list()

#Initializes dictionary to keep track of 1-GRAM counts
tag_counts = {}

#Initializes dictionary to keep track of emissions 
emissions_dic = {}

#Initializes list to keep track of sentences
sentence_list = []

#Finds rare words and adds them to the rare_words list
def find_rare(count_file):
	global rare_words
	#Dictionary to keep track of words and their counts
	words = {}
	counts_file = open(count_file, 'r')
	for line in counts_file:
		fields = line.split()
		#Only looks at WORDTAG lines
		if fields[1] == "WORDTAG":
			count = int(fields[0])
			word = fields[3]
			#Increments count if word is already in the dictionary
			if word in words:
				words[word] += count
			#Otherwise adds the word with its count
			else:
				words[word] = count
	#Checks dictionary for rare words (those with count < 5)
	for key in words:
		if words[key] < 5:
			rare_words.append(key)
	counts_file.close()

#Creates a new training data file with rare words replaced by special symbol '_RARE_'
def replace_rare(training_file):
	original_file = open(training_file, 'r+')
	output_file = open('rare_replaced.dat', 'w')
	for line in original_file:
		l = line.strip()
		if l: #Checks to make sure line isn't empty
			fields = l.split()
			word = fields[0]
			#Checks to see if the word is in the rare_words list
			if word in rare_words:
				#If so, replaces the word with '_RARE_' in output file
				output_file.write('_RARE_' + ' ' + ' '.join(fields[1:]) + '\n')
			else:
				#Otherwise, line is written to output file as is
				output_file.write(l + '\n')
		else:
			#Writes an empty line to the output file
			output_file.write(line)
	original_file.close()
	output_file.close()

#Calculates the counts for each tag, stores in global dictionary
def calc_tag_counts(counts_file):
	f = open(counts_file, 'r')
	global tag_counts
	for line in f:
		fields = line.split()
		if fields[1] == '1-GRAM':
			tag_counts[fields[2]] = fields[0]
	f.close()

#Calculates the emission probability for each (word, tag) pair, stores in the form
#of a nested dictionary e.g. {word: {tag: count}}
def emission_calculator(counts_file):
	global emissions_dic
	f = open(counts_file, 'r')
	#Runs calc_tag_counts to populate tag_counts dictionary
	calc_tag_counts(counts_file)
	for line in f:
		fields = line.split()
		#Looks only at WORDTAG lines, ignores N-GRAMs
		if fields[1] == 'WORDTAG':
			word = fields[3]
			tag = fields[2]
			#If word is in dictionary, add new tag
			if word in emissions_dic:
				emissions_dic[word][tag] = float(fields[0]) / float(tag_counts[tag])
			#Otherwise, creates new dictionary to hold tags and counts
			else:
				emissions_dic[word] = {}
				emissions_dic[word][tag] = float(fields[0]) / float(tag_counts[tag])
	f.close()

#After running emission_calculator, this function can be called to find a particular emission probability
def calc_emission(x,y):
	#Checks to see if we've seen this word in the training data
	if x in emissions_dic:
		if y in emissions_dic[x]:
			return emissions_dic[x][y]
		else:
			return 0
	#If we've never seen the word before, check to see if it is one of these types. Otherwise, treat as a '_RARE_' word
	else:
		if x.isdigit():
			if y in emissions_dic['_NUMBER_']:
				return emissions_dic['_NUMBER_'][y]
			else:
				return 0
		elif any(a.isdigit() for a in x):
			if y in emissions_dic['_HASDIGITS_']:
				return emissions_dic['_HASDIGITS_'][y]
			else:
				return 0
		elif x.isupper():
			if y in emissions_dic['_ALLCAPS_']:
				return emissions_dic['_ALLCAPS_'][y]
			else:
				return 0
		elif x.istitle():
			if y in emissions_dic['_TITLE_']:
				return emissions_dic['_TITLE_'][y]
			else:
				return 0
		elif x in string.punctuation:
			if y in emissions_dic['_PUNCTUATION_']:
				return emissions_dic['_PUNCTUATION_'][y]
			else:
				return 0
		elif any(a in string.punctuation for a in x):
			if y in emissions_dic['_HASPUNCTUATION_']:
				return emissions_dic['_HASPUNCTUATION_'][y]
			else:
				return 0
		elif x.islower():
			if y in emissions_dic['_LOWERCASE_']:
				return emissions_dic['_LOWERCASE_'][y]
			else:
				return 0
		else:
			if y in emissions_dic['_RARE_']:
				return emissions_dic['_RARE_'][y]
			else:
				return 0

#Takes in a counts file and a data file and for each word in the data file, outputs the most likely tag,
#along with the log probability 
#Use rare.counts (from training data) as argument for simple_tagger
def simple_tagger(counts_file, input_file):
	#Populates the emissions_dic from the counts file
	emission_calculator(counts_file)
	f = open(input_file, 'r')
	output_file = open('prediction_file', 'w')
	for line in f:
		l = line.strip()
		if l: #If the line is not empty
			#Checks to see if we have seen the word before
			if l in emissions_dic:
				#If so, looks up the tag with the maximum emission probability
				argmax_tag = max(emissions_dic[l], key=emissions_dic[l].get)
				#Calculates the log probability
				log_prob = math.log(emissions_dic[l][argmax_tag])
			#If we haven't seen the word before or if it is rare, we treat it the same as a rare word
			else:
				argmax_tag = max(emissions_dic['_RARE_'], key=emissions_dic['_RARE_'].get)
				log_prob = math.log(emissions_dic['_RARE_'][argmax_tag])
			#Writes output to file
			output_file.write(l + ' ' + argmax_tag + ' ' + str(log_prob) + '\n')
		else: #If the line is empty, just output the line
			output_file.write(line)
	f.close()
	output_file.close()

###############QUESTION 5##################

#Initializes dictionary to keep track of trigrams and bigrams and their counts
trigram_dict = {}
bigram_dict = {}

#Uses counts file to populate trigram and bigram count dictionaries
def trigram_counts(counts_file):
	f = open(counts_file, 'r')
	for line in f:
		fields = line.split()
		#Only looks at lines with 3-GRAM counts
		if fields[1] == '3-GRAM':
			count = int(fields[0])
			trigram = ' '.join(fields[2:])
			trigram_dict[trigram] = count
		#Only looks at lines with 2-GRAM counts
		if fields[1] == '2-GRAM':
			count = int(fields[0])
			bigram = ' '.join(fields[2:])
			bigram_dict[bigram] = count

#Calculates q parameters for a given trigram
def q(trigram):
	y_fields = trigram.split()
	#Sets y_i value
	y_i = y_fields[2]
	#Sets bigram y_(i-2) y_(i-1) 
	bigram = ' '.join(y_fields[:2])
	#Checks to see if trigram and bigram have been seen before in training data
	if trigram in trigram_dict and bigram in bigram_dict:
		return float(trigram_dict[trigram]) / float(bigram_dict[bigram])
	#If not, returns 0 because q will either be 0 or undefined
	else:
		return 0

#Reads in a state trigram and prints out the log probability
def q_log_prob(trigram):
	if q(trigram) == 0:
		return 0
	else:
		return math.log(q(trigram))

#Reads in a file and creates a list of the individual sentences
def sentence_splitter(input_file):
	f = open(input_file, 'r')
	global sentence_list
	#Buffer to keep track of current sentence
	current_sentence = []
	for line in f:
		l = line.strip()
		#If line is empty, add current_sentence to sentence_list and reset current_sentence
		if not l:
			sentence_list.append(current_sentence)
			current_sentence = []
		#If line is not empty, append word to current_sentence
		else:
			current_sentence.append(l)
	f.close()

def viterbi_tagger(counts_file, input_file):
	#Populates the emissions_dic from the counts file
	emission_calculator(counts_file)
	#Populates trigram and bigram count dictionaries
	trigram_counts(counts_file)
	#Splits data into sentences
	sentence_splitter(input_file)
	f = open(input_file, 'r')
	output_file = open('prediction_file', 'w')
	#For each sentence in our sentence_list, do the following:
	for sentence in sentence_list:
		n = len(sentence)
		#S(k) returns possible tag values at position k
		def S(k):
			#If k is position -1 or 0, only possible tag value is '*'
			if k in (-1, 0):
				return '*'
			#Otherwise, any tag values are possible
			else:
				return tag_counts.keys()
		#Given a list, function returns the argmax of the list
		def argmax(l):
			return max(l, key = lambda x: x[1])
		#Sets the beginning of the sentence as position 1 rather than 0
		x = [''] + sentence
		#Creates empty dictionary to keep track of pi values
		pi = {}
		#Initialization of pi and bp
		pi[0, '*', '*'] = 1
		bp = {}
		#Creates empty list to hold tag values
		y = [''] * (n+1)
		#Sets 0th tag as '*'
		y [0] = '*'
		#Main loop of Viterbi algorithm
		for k in range (1, n+1):
			for u in S(k - 1):
				for v in S(k):
					#Finds argmax over possible values of w
					bp[k,u,v], pi[k, u, v] = argmax([(w, pi[k - 1, w, u] * q(w + ' ' + u + ' ' + v) * calc_emission(x[k], v)) for w in S(k - 2)])
		(y[n-1], y[n]), score = argmax([((u,v), pi[n, u, v] * q(u + ' ' + v + ' '+ 'STOP')) for u in S(n-1) for v in S(n)])
		#Uses backpointers to set tag values
		for k in range(n - 2, 0, -1):
			y[k] = bp[k+2, y[k+1], y[k+2]]
		#Keeps track of log probability at each point in the sentence
		scores = [pi[i, y[i-1], y[i]] for i in range(1,n)]
		scores = scores + [score]
		#Writes to output file
		for k in range(1, n+1):
			output_file.write(x[k] + ' ' + y[k] + ' ' + str(scores[k-1]) + '\n')
		output_file.write('\n')
	f.close()
	output_file.close()

###############QUESTION 6##################
#Finds rare words and adds them to the rare_words list
def categorize_rare(count_file, training_file):
	find_rare(count_file)
	original_file = open(training_file, 'r+')
	output_file = open('rare_replaced.dat', 'w')
	for line in original_file:
		l = line.strip()
		if l:
			fields = l.split()
			word = fields[0]
			if word in rare_words:
				if word.isdigit():
					output_file.write('_NUMBER_' + ' ' + ' '.join(fields[1:]) + '\n')
				elif any(x.isdigit() for x in word):
					output_file.write('_HASDIGITS_' + ' ' + ' '.join(fields[1:]) + '\n')
				elif word.isupper():
					output_file.write('_ALLCAPS_' + ' ' + ' '.join(fields[1:]) + '\n')
				elif word.istitle():
					output_file.write('_TITLE_' + ' ' + ' '.join(fields[1:]) + '\n')
				elif word in string.punctuation:
					output_file.write('_PUNCTUATION_' + ' ' + ' '.join(fields[1:]) + '\n')
				elif any(x in string.punctuation for x in word):
					output_file.write('_HASPUNCTUATION_' + ' ' + ' '.join(fields[1:]) + '\n')
				elif word.islower():
					output_file.write('_LOWERCASE_' + ' ' + ' '.join(fields[1:]) + '\n')
				else:
					output_file.write('_RARE_' + ' ' + ' '.join(fields[1:]) + '\n')
			else: 
				output_file.write(l + '\n')
		else:
			output_file.write(line)
	original_file.close()
	output_file.close()
