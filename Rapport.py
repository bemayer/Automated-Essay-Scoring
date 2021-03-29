# -*- coding: utf-8 -*-
'''
Created on Wed Mar 17 09:04:33 2021

@author: bemayer
'''

# pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.12:3.0.0
# https://www.sicara.ai/blog/2017-05-02-get-started-pyspark-jupyter-notebook-3-minutes

import re
import os
import math
import string
import kaggle
import zipfile
import pandas as pd
import numpy as np
import language_check
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import first, udf, split, flatten

from pyspark.ml import Pipeline
from pyspark.ml.stat import Summarizer
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler, StandardScaler, VectorSlicer, Word2Vec


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
	encoding='windows-1252', header=True, inferSchema=True)

# # Show infos
# data.printSchema()
# data.createOrReplaceTempView('data')
# data_rdd = data.rdd
# data_pd = data.toPandas()
# data_pd.info()

# # Show essay count by essay subject
# query = '''SELECT essay_set as Subject, COUNT(essay) as Count FROM data GROUP BY essay_set
# 	ORDER BY essay_set'''
# essay_nb = sc.sql(query).toPandas()
# fig, ax = plt.subplots()
# ax.bar(essay_nb['Subject'], essay_nb['Count'])
# plt.title('Essay count by Subject')
# plt.xlabel('Subject')
# plt.ylabel('Count')
# plt.show()

# # Show summary of scores by subject

# query = '''SELECT essay_set as Subject, min(domain1_score) as Min,
# 	max(domain1_score) as Max, count(domain1_score) as Nb,
# 	count(distinct domain1_score) as Unique,
# 	format_number(avg(domain1_score), '#.##') as Avg,
# 	format_number(stddev(domain1_score), '#.##') as StDev
# 	FROM data GROUP BY essay_set ORDER BY Subject'''
# sc.sql(query).show()

# # Boxplot of scores by subject with bins


# Normalization of scores by subject and stratified sampling
data = data.filter('domain1_score is not null')

scores_by_set = {}
assembler = (VectorAssembler().setInputCols(['domain1_score'])
	.setOutputCol('domain1_score_vector'))
scaler = (StandardScaler().setWithMean(True)
	.setInputCol('domain1_score_vector').setOutputCol('score_vector'))




for set in range(1,9):
	scores_by_set[set] = data.select('essay_id', 'essay_set',
		'domain1_score').filter('essay_set == ' + str(set))
	scores_by_set[set] = assembler.transform(scores_by_set[set])
	scores_by_set[set] = (scaler.fit(scores_by_set[set])
		.transform(scores_by_set[set]).toPandas().set_index('essay_id'))


# How to vectorially unvectorized a vector ?...
# https://stackoverflow.com/questions/38110038/spark-scala-how-to-convert-dataframevector-to-dataframef1double-fn-d
for set in range(1,9):
	score = [[essay, scores_by_set[set]['score_vector'][essay][0]] for essay in scores_by_set[set].index]
	score = pd.DataFrame(score, columns=['essay_id', 'score']).set_index('essay_id')
	scores_by_set[set] = pd.concat([scores_by_set[set], score], axis = 1)

scores = pd.concat(scores_by_set.values())[['essay_set', 'score']]
round(scores.score.std(), 3)
round(scores.score.mean(), 3)

rank_by_set = {}
for set in range(1,9):
	rank_by_set[set] = scores.loc[scores['essay_set'] == set]['score'].rank(pct=True, method ='first')




rank = pd.concat(rank_by_set.values())
rank = rank.rename('rank')

scores = pd.concat([scores, rank], axis=1)

scores.loc[scores['rank'] <= 1/5, 'rank_group'] = '0'
scores.loc[(scores['rank'] > 1/5) & (scores['rank'] <= 2/5), 'rank_group'] = '1'
scores.loc[(scores['rank'] > 2/5) & (scores['rank'] <= 3/5), 'rank_group'] = '2'
scores.loc[(scores['rank'] > 3/5) & (scores['rank'] <= 4/5), 'rank_group'] = '3'
scores.loc[scores['rank'] > 4/5, 'rank_group'] = '4'

table_count = pd.pivot_table(scores, values='score', index=['essay_set'],
	columns=['rank_group'], aggfunc=pd.Series.count, margins = True)
table_mean = pd.pivot_table(scores, values='score', index=['essay_set'],
	columns=['rank_group'], aggfunc=pd.Series.mean, margins = True)

scores_train = scores.sample(frac=0.7, random_state=42)
scores_test = scores.loc[np.setdiff1d(scores.index, scores_train.index)]

table_count = pd.pivot_table(scores_train, values='score', index=['essay_set'],
	columns=['rank_group'], aggfunc=pd.Series.count, margins = True)
table_mean = pd.pivot_table(scores_train, values='score', index=['essay_set'],
	columns=['rank_group'], aggfunc=pd.Series.mean, margins = True)

# # Show distribution of word counts
# data_pd.hist(column='word_count', by='topic', bins=25, sharey=True, sharex=True, layout=(2, 4), figsize=(7,4), rot=0)
# plt.suptitle('Word count by topic #')
# plt.xlabel('Number of words')
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()

# Features generation
def nb_words(str):
	return(len(str.split()))

def nb_organization(str):
	return(str.count('@ORGANIZATION'))

def nb_caps(str):
	return(str.count('@CAPS'))

def nb_person(str):
	return(str.count('@PERSON') + str.count('@DR'))

def nb_location(str):
	return(str.count('@LOCATION') + str.count('@CITY') + str.count('@STATE'))

def nb_money(str):
	return(str.count('@MONEY'))

def nb_time(str):
	return(str.count('@TIME'))

def nb_date(str):
	return(str.count('@DATE')+str.count('@MONTH'))

def nb_percent(str):
	return(str.count('@PERCENT') + str.count('@NUM'))

nb_wordsUdf = udf(lambda str: nb_words(str), IntegerType())
nb_organizationUdf = udf(lambda str: nb_organization(str), IntegerType())
nb_capsUdf = udf(lambda str: nb_caps(str), IntegerType())
nb_personUdf = udf(lambda str: nb_person(str), IntegerType())
nb_locationUdf = udf(lambda str: nb_location(str), IntegerType())
nb_moneyUdf = udf(lambda str: nb_money(str), IntegerType())
nb_timeUdf = udf(lambda str: nb_time(str), IntegerType())
nb_dateUdf = udf(lambda str: nb_date(str), IntegerType())
nb_percentUdf = udf(lambda str: nb_percent(str), IntegerType())

data = data.withColumn('nb_words', nb_wordsUdf(data.essay))
data = data.withColumn('nb_organization', nb_organizationUdf(data.essay))
data = data.withColumn('nb_caps', nb_capsUdf(data.essay))
data = data.withColumn('nb_person', nb_personUdf(data.essay))
data = data.withColumn('nb_location', nb_locationUdf(data.essay))
data = data.withColumn('nb_money', nb_moneyUdf(data.essay))
data = data.withColumn('nb_time', nb_timeUdf(data.essay))
data = data.withColumn('nb_date', nb_dateUdf(data.essay))
data = data.withColumn('nb_percent', nb_percentUdf(data.essay))

# Correction grammaticale
def nb_error(str):
	matches = language_check.LanguageTool('en-US').check(str)
	nb_error = (len(matches))
	return(nb_error)

def correcter(str):
	corrected = language_check.LanguageTool('en-US').correct(str)
	return(corrected)

# Remplacement par des espaces de charactères qui gènent la tokenisation de Spark
def replace_char(str):
	for char in ['\\', '/', '(', ')', '[', ']', '{', '}']:
		str = str.replace(char, ' ')
	return (str)

def replace_anom(str):
	for char in ['@ORGANIZATION.', '@CAPS.', '@PERSON.', '@LOCATION.', '@MONEY.', '@TIME.', '@DATE.', '@PERCENT.', '@MONTH.', '@NUM.', '@DR.', '@CITY.', '@STATE.']:
		str = re.sub(char, ' ', str)
	return (str)

# firstEssay =  data.limit(1).select('essay').toPandas().squeeze()
# print(firstEssay)
# print("Nombre d'erreurs:", len(language_check.LanguageTool('en-US').check(firstEssay)))
# print(language_check.LanguageTool('en-US').correct(firstEssay))

nb_errorUdf = udf(lambda str: nb_error(str), IntegerType())
correcterUdf = udf(lambda str: correcter(str), StringType())
replace_charUdf = udf(lambda str: replace_char(str), StringType())
replace_anomUdf = udf(lambda str: replace_anom(str), StringType())

data = data.withColumn('nb_orth_error', nb_errorUdf(data.essay))
data = data.withColumn('essay', correcterUdf(data.essay))
data = data.withColumn('essay', replace_charUdf(data.essay))
data = data.withColumn('essay', replace_anomUdf(data.essay))

if not os.path.exists('Data/data_corrected.parquet'):
	data.write.parquet('Data/data_corrected.parquet')
	data = sc.read.parquet('Data/data_corrected.parquet')
	data.toPandas().to_csv('Data/data_corrected.csv')
else:
	data = sc.read.parquet('Data/data_corrected.parquet')



# Preprocessing pipeline
documenter = (DocumentAssembler().setCleanupMode('shrink').setInputCol('essay')
				.setOutputCol('document'))
sentencer = (SentenceDetector().setInputCols(['document'])
				.setOutputCol('sentence'))
tokenizer = Tokenizer().setInputCols(['document']).setOutputCol('tokenized')
tokenizer2 = Tokenizer().setInputCols(['sentence']).setOutputCol('tokenized')
normalizer = (Normalizer().setLowercase(True).setInputCols(['tokenized'])
				.setOutputCol('normalized'))
cleaner = (StopWordsCleaner().setInputCols(['normalized'])
				.setOutputCol('cleaned'))
lemmatizer = (LemmatizerModel.pretrained(name = 'lemma_antbnc', lang='en')
				.setInputCols(['cleaned']).setOutputCol('lemmatized'))
vectorizer = (Word2Vec().setSeed(42).setVectorSize(300)
				.setInputCol('finished').setOutputCol('vectorized'))
vectorizer2 = (WordEmbeddingsModel().pretrained('glove_6B_300', 'xx')
				.setInputCols('sentence', 'tokenized')
				.setOutputCol('vectorized'))
finisher = Finisher().setInputCols(['lemmatized']).setOutputCols('finished')
finisher2 = (EmbeddingsFinisher().setOutputAsVector(True)
				.setInputCols(['vectorized']).setOutputCols('finished'))

pipeline_w2v = Pipeline().setStages([documenter, tokenizer, normalizer, cleaner,
				lemmatizer, finisher, vectorizer])
pipeline_glove = Pipeline().setStages([documenter, sentencer,
				tokenizer2, vectorizer2, finisher2])

if not os.path.exists('Data/data_w2v.parquet'):
	data_w2v = pipeline_w2v.fit(data).transform(data)
	data_w2v.write.parquet('Data/data_w2v.parquet')
	data_w2v = sc.read.parquet('Data/data_w2v.parquet')
else:
	data_w2v = sc.read.parquet('Data/data_w2v.parquet')

if not os.path.exists('Data/data_glove.parquet'):
	data_glove = pipeline_glove.fit(data).transform(data)
	data_glove.write.parquet('Data/data_glove.parquet')
	data_glove = sc.read.parquet('Data/data_glove.parquet')
else:
	data_glove = sc.read.parquet('Data/data_glove.parquet')


data_w2v_pd = data_w2v.toPandas().set_index('essay_id')
data_glove_pd = data_glove.toPandas().set_index('essay_id')

vector_w2v = data_w2v_pd['vectorized'].apply(lambda x: pd.Series(x.toArray()))
vector_w2v.columns = ['vec_' + str(i) for i in range(0, 300)]

vector_glove = data_glove_pd['vectorized'].apply(lambda x: pd.Series(x.toArray()))
vector_glove.columns = ['vec_' + str(i) for i in range(0, 300)]

selected = ['essay_set'] + [s for s in data_w2v_pd.columns if 'nb' in s]

X_w2v = pd.concat([data_w2v_pd[selected], vector_w2v], axis = 1)
X_glove = pd.concat([data_glove_pd[selected], vector_glove], axis = 1)

X_w2v_train = X_w2v.loc[scores_train.index]
X_w2v_test = X_w2v.loc[scores_test.index]

X_glove_train = X_glove.loc[scores_train.index]
X_glove_test = X_glove.loc[scores_test.index]

X_w2v_train.to_csv('Data/X_w2v_train.csv')
X_w2v_test.to_csv('Data/X_w2v_test.csv')

X_glove_train.to_csv('Data/X_glove_train.csv')
X_glove_test.to_csv('Data/X_glove_test.csv')

scores_train.to_csv('Data/y_train.csv')
scores_test.to_csv('Data/y_test.csv')



# Stratified sampling


# Visualization: make a treemap for most frequent subject ?
# https://plotly.com/python/treemaps/
