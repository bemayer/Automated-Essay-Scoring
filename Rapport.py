# -*- coding: utf-8 -*-
'''
Created on Wed Mar 17 09:04:33 2021

@author: bemayer
'''

# pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.12:3.0.0
# https://www.sicara.ai/blog/2017-05-02-get-started-pyspark-jupyter-notebook-3-minutes

import re
import os
import string
import kaggle
import zipfile
import pandas as pd
import language_check
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import udf

from pyspark.ml import Pipeline
from pyspark.ml.stat import Summarizer
from pyspark.ml.feature import Word2Vec

from sparknlp.base import DocumentAssembler, Finisher, EmbeddingsFinisher
from sparknlp.annotator import Tokenizer, Normalizer, StopWordsCleaner, SentenceDetector, WordEmbeddingsModel
from sparknlp.pretrained import LemmatizerModel

# Download data
if not os.path.exists('Data'):
	kaggle.api.authenticate()
	kaggle.api.competition_download_files('asap-aes')
	zipfile.ZipFile('asap-aes.zip').extractall(path='Data')


# Create SparkSession
sc = SparkSession.builder.getOrCreate()

# Import data
data = sc.read.csv('./Data/training_set_rel3.tsv', sep='\t',
	encoding='ISO-8859-1', header=True, inferSchema=True)


# Show infos
data.printSchema()
data.createOrReplaceTempView('data')
data_rdd = data.rdd
data_pd = data.toPandas()
data_pd.info()

# Show essay count by essay subject
query = '''SELECT essay_set as Subject, COUNT(essay) as Count FROM data GROUP BY essay_set
	ORDER BY essay_set'''
essay_nb = sc.sql(query).toPandas()
fig, ax = plt.subplots()
ax.bar(essay_nb['Subject'], essay_nb['Count'])
plt.title('Essay count by Subject')
plt.xlabel('Subject')
plt.ylabel('Count')
plt.show()

# Show summary of scores by subject

query = '''SELECT essay_set as Subject, min(domain1_score) as Min,
	max(domain1_score) as Max, count(domain1_score) as Nb,
	count(distinct domain1_score) as Unique,
	format_number(avg(domain1_score), '#.##') as Avg,
	format_number(stddev(domain1_score), '#.##') as StDev
	FROM data GROUP BY essay_set ORDER BY Subject'''
sc.sql(query).show()

# Boxplot of scores by subject with bins


# Normalization of scores by subject


# Show distribution of word counts
data_pd.hist(column='word_count', by='topic', bins=25, sharey=True, sharex=True, layout=(2, 4), figsize=(7,4), rot=0)
plt.suptitle('Word count by topic #')
plt.xlabel('Number of words')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Features generation
def nb_words(str):
	return(len(str.split()))

def nb_organization(str):
	return(str.count('@ORGANIZATION'))

def nb_caps(str):
	return(str.count('@CAPS'))

def nb_person(str):
	return(str.count('@PERSON'))

def nb_location(str):
	return(str.count('@LOCATION'))

def nb_money(str):
	return(str.count('@MONEY'))

def nb_time(str):
	return(str.count('@TIME'))

def nb_date(str):
	return(str.count('@DATE'))

def nb_percent(str):
	return(str.count('@PERCENT'))

def replace_anom(str):
	for char in ['@ORGANIZATION.', '@CAPS.', '@PERSON.', '@LOCATION.', '@MONEY.', '@TIME.', '@DATE.', '@PERCENT.']:
		str = re.sub(char, ' ', str)
	return (str)



nb_wordsUdf = udf(lambda str: nb_words(str), IntegerType())
nb_organizationUdf = udf(lambda str: nb_organization(str), IntegerType())
nb_capsUdf = udf(lambda str: nb_caps(str), IntegerType())
nb_personUdf = udf(lambda str: nb_person(str), IntegerType())
nb_locationUdf = udf(lambda str: nb_location(str), IntegerType())
nb_moneyUdf = udf(lambda str: nb_money(str), IntegerType())
nb_timeUdf = udf(lambda str: nb_time(str), IntegerType())
nb_dateUdf = udf(lambda str: nb_date(str), IntegerType())
nb_percentUdf = udf(lambda str: nb_percent(str), IntegerType())
replace_anomUdf = udf(lambda str: replace_anom(str), StringType())

data = data.withColumn('nb_words', nb_wordsUdf(data.essay))
data = data.withColumn('nb_organization', nb_organizationUdf(data.essay))
data = data.withColumn('nb_caps', nb_capsUdf(data.essay))
data = data.withColumn('nb_person', nb_personUdf(data.essay))
data = data.withColumn('nb_location', nb_locationUdf(data.essay))
data = data.withColumn('nb_money', nb_moneyUdf(data.essay))
data = data.withColumn('nb_time', nb_timeUdf(data.essay))
data = data.withColumn('nb_date', nb_dateUdf(data.essay))
data = data.withColumn('nb_percent', nb_percentUdf(data.essay))
data = data.withColumn('essay', replace_anomUdf(data.essay))


# Correction grammaticale
def nb_error(str, id):
	if id % 1000 == 0:
		print(id + "/ 12976 rédactions vérifiées")
	matches = language_check.LanguageTool('en-US').check(str)
	return(len(matches))

def correcter(str, id):
	if id % 1000 == 0:
		print(id + "/ 12976 rédactions corrigées")
	return(language_check.LanguageTool('en-US').correct(str))

# Remplacement par des espaces de charactères qui gènent la tokenisation de Spark
def replace_char(str):
	for char in ['\\', '/', '(', ')', '[', ']', '{', '}']:
		str = str.replace(char, ' ')
	return (str)


firstEssay =  data.toPandas()['essay'].iloc[0]
print(firstEssay)
print(nb_error(firstEssay))
print(correcter(firstEssay))


nb_errorUdf = udf(lambda str, id: nb_error(str, id), IntegerType())
coorrecterUdf = udf(lambda str, id: correcter(str, id), StringType())
replace_charUdf = udf(lambda str: replace_char(str), StringType())

# data = data.withColumn('essay_corrected', coorrecterUdf(data.essay))
# data = data.withColumn('nb_orth_error', nb_errorUdf(data.essay))
# data.select('nb_orth_error').show()

data2 = data.limit(10).withColumn('essay', coorrecterUdf(data.essay, data.essay_id))
data2 = data2.withColumn('nb_orth_error', nb_errorUdf(data2.essay, data2.essay_id))
data2 = data2.withColumn('essay', replace_charUdf(data2.essay))
data2.select('nb_orth_error').show()




# Preprocessing pipeline
documenter = DocumentAssembler().setInputCol('essay').setOutputCol('document').setCleanupMode('shrink')
sentencer = SentenceDetector().setInputCols(['document']).setOutputCol('sentence')
tokenizer = Tokenizer().setInputCols(['document']).setOutputCol('tokenized')
tokenizer2 = Tokenizer().setInputCols(['sentence']).setOutputCol('tokenized')
normalizer = Normalizer().setInputCols(['tokenized']).setOutputCol('normalized').setLowercase(True)
cleaner = StopWordsCleaner().setInputCols(['normalized']).setOutputCol('cleaned')
lemmatizer = LemmatizerModel.pretrained().setInputCols(['cleaned']).setOutputCol('lemmatized')
finisher = Finisher().setInputCols(['lemmatized']).setOutputCols('finished')
vectorizer = WordEmbeddingsModel().pretrained('glove_6B_300', 'xx').setInputCols('sentence', 'tokenized').setOutputCol('embeddings')
finisher2 = EmbeddingsFinisher().setInputCols(['embeddings']).setOutputCols('vectorized').outputAsVector(True)

pipeline_w2v = Pipeline().setStages([documenter, tokenizer, normalizer, cleaner, lemmatizer, finisher])
pipeline_glove = Pipeline().setStages([documenter, sentencer, tokenizer2, vectorizer, finisher2])

data_glove = pipeline_glove.fit(data2).transform(data2)
data_glove_pd = data_glove.toPandas()


data_w2v = pipeline_w2v.fit(data2).transform(data2)
w2v_trained = Word2Vec(vectorSize=300, seed=42, inputCol='finished', outputCol='vectorized').fit(data_w2v)
data_w2v = w2v_trained.transform(data_w2v)
data_w2v_pd = data_w2v.toPandas()


for i in range(0,10):
	data_glove_pd['vectorized'][i] = data_glove_pd['vectorized'][i][0]


data_glove.schema
data_w2v.schema

test = data_glove_pd['vectorized'].apply(pd.Series)

pd.concat([data_glove_pd, test], axis = 1)

data2 = data2.toPandas()


syn = w2v_trained.findSynonyms('newspaper', 5)





# Make a treemap for most frequent words
# https://plotly.com/python/treemaps/





matches = language_check.LanguageTool('en-US').check(firstEssay)






data2_pd = data2.toPandas()
data2_pd['essay_corrected'][0]
str = data_pd["essay"][0]
str2 = data2_pd["essay"][0]


str = re.sub('@CAPS*', ' ', str)




