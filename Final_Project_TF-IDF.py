# Databricks notebook source
#Group Members
# Mananage Sanjaya Kumara          mmk190004
# Dona Hasini Gammune              dvg190000
# Sahra Premani                    snp200002
# Shradha Upadhyay                 sxu140730

# COMMAND ----------

#Please install nltk using following command for the first time
!pip install nltk
import nltk
nltk.download("punkt")
nltk.download("stopwords")

# COMMAND ----------

##Text Sumarization using sentence scores

def Summarization_by_Sentence_Score(file,n_sentences=0,ratio=.25):
    
    #Return the length of a line
    def lenOfLine(l):
        return len(l.split(" "))
    
    file_no_space = file.filter(lambda x: lenOfLine(x)>1)
    file_splited = file_no_space.flatMap(lambda x: x.split(". "))
    
    from pyspark import SparkContext, SparkConf
    file_with_id = file_splited.zipWithIndex()
    file_with_id = file_with_id.map(lambda x: (x[1]+1,x[0]))
    
    #Create a date frame with sentence id and sentence
    df_file_with_id = file_with_id.toDF(["Sent_Id","Sentence"])
    #display(df_file_with_id)
    
    #Clean the text and pre processing
    from pyspark.sql.functions import col, lower, regexp_replace, split
    def clean_sentence(sentence):
        sentence = lower(sentence)
        sentence = regexp_replace(sentence, "^rt ", "")
        sentence = regexp_replace(sentence, "(https?\://)\S+", "")
        sentence = regexp_replace(sentence, "[^a-zA-Z0-9\\s]", "")
        return sentence

    clean_text_df = df_file_with_id.select("Sent_Id",clean_sentence(col("Sentence")).alias("Sentence_clean"))
    
    #Tokenize
    from pyspark.ml.feature import Tokenizer
    tokenizer = Tokenizer(inputCol="Sentence_clean", outputCol="tokenize_vec")
    Token_df = tokenizer.transform(clean_text_df).select("Sent_Id","tokenize_vec")
    
    #Remove stop words
    from pyspark.ml.feature import StopWordsRemover
    #Use default stop words  list
    remover = StopWordsRemover()
    stopwords = remover.getStopWords()
    
    # Specify input/output columns
    remover.setInputCol("tokenize_vec")
    remover.setOutputCol("tokenize_vec_no_stopw")

    No_stopw_df = remover.transform(Token_df).select("Sent_Id","tokenize_vec_no_stopw")
    
    # Stemming words
    from pyspark.sql.functions import udf
    from pyspark.sql.types import ArrayType, StringType, IntegerType
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()
    
    # python function to stemmer
    def stem(input_vec):
        output_vec = []
        for t in input_vec:
            stem_t = stemmer.stem(t)
            if len(stem_t) > 2:
                output_vec.append(stem_t)       
        return output_vec

    # defined a function for stemming
    stemmer_func = udf(lambda x: stem(x), ArrayType(StringType()))

    # Create new df with stemmed words
    stemmed_df = (
        No_stopw_df
            .withColumn("vector_stemmed", stemmer_func("tokenize_vec_no_stopw"))
            .select("Sent_Id","vector_stemmed")
      )
    
    display(stemmed_df)
    ################
    # Calculate scores for each sentence using TF-IDF method
    ################
    stemmed_df.createOrReplaceTempView("stemmed_df_temp")
    
    ##Create RDD object from df stemmed_df_temp
    sqlDF= spark.sql("SELECT * from stemmed_df_temp")
    sqlRDD= sqlDF.rdd
    sqlRDD.map(lambda x:(x[0],x[1]))
    
    append_1=sqlRDD.flatMap(lambda x: [((x[0],i),1) for i in x[1]])
    reduce_key_pair=append_1.reduceByKey(lambda x,y:x+y)
    TF_df=reduce_key_pair.map(lambda x: (x[0][1],(x[0][0],x[1])))
    append_2=reduce_key_pair.map(lambda x: (x[0][1],(x[0][0],x[1],1)))
    word_value_pair=append_2.map(lambda x:(x[0],x[1][2]))
    word_sum=word_value_pair.reduceByKey(lambda x,y:x+y)
    
    doc_count=sqlRDD.count()
    
    from pyspark.sql.functions import udf
    import math
    IDF_df=word_sum.map(lambda x: (x[0],math.log10(doc_count/x[1])))
    
    rdd_TF_IDF=TF_df.join(IDF_df)
    rdd_TF_IDF=rdd_TF_IDF.map(lambda x: (x[1][0][0],(x[0],x[1][0][1],x[1][1],x[1][0][1]*x[1][1]))).sortByKey()
    rdd_TF_IDF=rdd_TF_IDF.map(lambda x: (x[0],x[1][0],x[1][1],x[1][2],x[1][3]))
    
    rdd_select=rdd_TF_IDF.map(lambda x: (x[0],x[4]))
    
    #data frame of Tokenized words with TF_IDF values sorted by TF_IDF in descending
    rdd_desc=rdd_TF_IDF.map(lambda x:(x[0],x[1],x[4]))
    df_with_tf_idf=rdd_desc.toDF(["Sent_Id","Tokenize_word","TF_IDF"])
    df_with_tf_idf=df_with_tf_idf.orderBy([col("TF_IDF").desc()])
    
    print("#################################################")
    print("Tokenized words with TF_IDF values")
    print("#################################################")
    display(df_with_tf_idf)
    
    ###############################################################################
    import matplotlib.pyplot as plt
    from pyspark.sql.functions import col

    # convert the PySpark DataFrame to a Pandas DataFrame
    df_with_tf_idf_20=df_with_tf_idf.limit(20)
    pandas_df = df_with_tf_idf_20.toPandas()

    # create a bar plot using matplotlib
    pandas_df.plot(kind="line", x="Tokenize_word", y="TF_IDF", color="green")
    plt.title("Line plot of TF_IDF for top 20 words")
    plt.xlabel("Tokenize_word")
    plt.ylabel("TF_IDF")
    plt.xticks(rotation=90)
    plt.show()
    
    ############################################################################
    
    # Num of sentence to generate summary
    from pyspark.sql.functions import col
    from pyspark.sql.functions import round
    from pyspark.sql import SparkSession
    
    #Function to find the no of sentences
    def count_n_sentence(t_sentences,n_sentences,ratio):
        if n_sentences>0:
            return n_sentences
        else:
            if ratio==.25:
                return Decimal(float(t_sentences)*float(ratio))
            else:
                return Decimal(float(t_sentences)*float(ratio))

    #Calculate the number of sentences
    from decimal import Decimal, ROUND_HALF_UP
    t_sentences=df_file_with_id.count()
    n_sentences=count_n_sentence(t_sentences,n_sentences,ratio)
    n_sentences=Decimal(n_sentences)
    n_sentences = n_sentences.quantize(Decimal('0'), rounding=ROUND_HALF_UP)
    ###############################################################################
    
    ############
    # Calculate the scores for each sentence
    ############
    # define initial value for accumulator
    initial_value = (0, 0)  # (total, count)
    # define merge function to sum the totals and counts
    def merge_value(acc, value):
        total, count = acc
        return (total + value, count + 1)
    # define update function to add a new value to the accumulator
    def update_value(acc, new_value):
        total, count = acc
        return (total + new_value, count + 1)

    # call aggregateByKey function with initial value, merge function, and update function
    sum_and_count = rdd_select.aggregateByKey(initial_value, merge_value, update_value)

    #Sent scores using sum of TF-IDF values
    sum_score = sum_and_count.mapValues(lambda acc: acc[0]).sortByKey()
    sum_score_cal=sum_score.toDF(["Sent_Id","Sent_score_sum"])
    sum_score_cal=sum_score_cal.select(["Sent_Id","Sent_score_sum"]).orderBy([col("Sent_score_sum").desc()])
    
    #Join sum scores with original sentences
    df_file_with_Score_sum=df_file_with_id.join(sum_score_cal,["Sent_Id"])
    df_file_with_Score_sum=df_file_with_Score_sum.select(["Sent_Id","Sentence","Sent_score_sum"]).orderBy([col("Sent_score_sum").desc()])
    #Displaying sentences and their sum scores
    print("#################################################")
    print("Displaying sentences and their sum scores")
    print("#################################################")
    display(df_file_with_Score_sum)
    
    ###############################################################################
    import matplotlib.pyplot as plt
    from pyspark.sql.functions import col

    # convert the PySpark DataFrame to a Pandas DataFrame
    df_file_with_Score_sort_id_sum=df_file_with_Score_sum.orderBy([col("Sent_Id")])
    pandas_df_sum = df_file_with_Score_sort_id_sum.toPandas()

    # create a bar plot using matplotlib
    pandas_df_sum.plot(kind="bar", x="Sent_Id", y="Sent_score_sum", color="blue")
    plt.title("Bar plot of Sentence scores using sum")
    plt.xlabel("Sentece Id")
    plt.ylabel("Sentence Score Sum")
    plt.show()
    
    ###############
    # Create the summary with sum scores
    ###############
    
    temp_df_sum=df_file_with_Score_sum.limit(int(n_sentences))
    temp_df_sum=temp_df_sum.select(["Sentence"]).orderBy("Sent_Id")
    
    # Define the column to concatenate
    column_to_concat = "Sentence"

    # Create an empty string to store the concatenated values
    concatenated_string = ""

    # Iterate over each row of the DataFrame
    for row in temp_df_sum.rdd.take(int(n_sentences)):
        # Get the value of the column to concatenate for the current row
        value_to_concat = row[column_to_concat]

        # Add the value to the concatenated string
        concatenated_string += value_to_concat
        concatenated_string +=". "

    # Print the concatenated string
    print("#################################################")
    print("The summary with", int(n_sentences), "sentences using sum")
    print("Total number of sentences: ", int(t_sentences))
    print("#################################################","\n")
    print(concatenated_string,"\n")
    print("#################################################","\n")
    
    ###############################################################################
    
    # divide total by count to get the average
    averages = sum_and_count.mapValues(lambda acc: acc[0] / acc[1]).sortByKey()
    
    #Create df with scores
    Sent_Score_cal=averages.toDF(["Sent_Id","Sent_score_avg"])
    
    #Join scores with original sentences
    df_file_with_Score=df_file_with_id.join(Sent_Score_cal,["Sent_Id"])
    df_file_with_Score=df_file_with_Score.select(["Sent_Id","Sentence","Sent_score_avg"]).orderBy([col("Sent_score_avg").desc()])
    
    #Displaying sentences and their avg scores
    print("#################################################")
    print("Displaying sentences and their average scores")
    print("#################################################")
    display(df_file_with_Score)
    
    ###############################################################################
    import matplotlib.pyplot as plt
    from pyspark.sql.functions import col

    # convert the PySpark DataFrame to a Pandas DataFrame
    df_file_with_Score_sort_id=df_file_with_Score.orderBy([col("Sent_Id")])
    pandas_df = df_file_with_Score_sort_id.toPandas()

    # create a bar plot using matplotlib
    pandas_df.plot(kind="bar", x="Sent_Id", y="Sent_score_avg", color="red")
    plt.title("Bar plot of Sentence scores using average")
    plt.xlabel("Sentece Id")
    plt.ylabel("Sentence Score Avg")
    plt.show()
    
    ###############################################################################
    
    ###############
    # Create the summary with average scores
    ###############
    
    temp_df=df_file_with_Score.limit(int(n_sentences))
    temp_df=temp_df.select(["Sentence"]).orderBy("Sent_Id")
    
    # Define the column to concatenate
    column_to_concat = "Sentence"

    # Create an empty string to store the concatenated values
    concatenated_string = ""

    # Iterate over each row of the DataFrame
    for row in temp_df.rdd.take(int(n_sentences)):
        # Get the value of the column to concatenate for the current row
        value_to_concat = row[column_to_concat]

        # Add the value to the concatenated string
        concatenated_string += value_to_concat
        concatenated_string +=". "

    # Print the concatenated string
    print("#################################################")
    print("The summary with", int(n_sentences), "sentences using average")
    print("Total number of sentences: ", int(t_sentences))
    print("#################################################","\n")
    print(concatenated_string,"\n")
    print("#################################################","\n")
    
    ##################################
    # Plot to display the sum and avg scores
    ##################################
    
    print("#################################################","\n")
    print("Line plots for sentence scores using sum and average")
    print("#################################################","\n")
    
    df_sum_and_avg = sum_score_cal.join(Sent_Score_cal,["Sent_Id"])    
    
    from pyspark.sql.functions import col
    import matplotlib.pyplot as plt

    # Create two panels for line plots
    fig, (plt1, plt2) = plt.subplots(2, 1, figsize=(8, 6))

    # Plot data on panel 1
    plt1.plot(df_sum_and_avg.select('Sent_Id').collect(), df_sum_and_avg.select('Sent_score_sum').collect(), 'b-', label='Sent_score_sum')
    plt1.set_title('Using Sum')
    plt1.set_xlabel('Sent_Id')
    plt1.set_ylabel('Sent_score_sum')
    plt1.legend(loc='upper left')

    # Plot data on panel 2
    plt2.plot(df_sum_and_avg.select('Sent_Id').collect(), df_sum_and_avg.select('Sent_score_avg').collect(), 'r-', label='Sent_score_avg')
    plt2.set_title('Using Average')
    plt2.set_xlabel('Sent_Id')
    plt2.set_ylabel('Sent_score_avg')
    plt2.legend(loc='upper right')

    # Adjust the spacing between the panels
    plt.subplots_adjust(hspace=0.4)

    # Show the plot
    plt.show()

# COMMAND ----------

## Main Method

#Read the data files
Google_toolbar=sc.textFile("/FileStore/tables/Google_toolbar.txt")
#Call the summarization function
Summarization_by_Sentence_Score(Google_toolbar)
