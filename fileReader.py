import pandas as pd
import Cleaner
import tf_idf

resume_data_file = "C:/Users/Jothi/OneDrive/Desktop/MINIPROJECT/Demo/Resume_Data.csv"
job_data_file = "C:/Users/Jothi/OneDrive/Desktop/MINIPROJECT/Demo/Job_Data.csv"

resume_data = pd.read_csv(resume_data_file)
job_data = pd.read_csv(job_data_file)

def get_cleaned_words(document):
    for i in range(len(document)):
        raw = Cleaner.Cleaner(document[i])
        document[i] = " ".join(raw[0])
        sentence = tf_idf.generate_top_words(document[i].split(" "))
        document[i] = sentence
    return document

resume_data["TF_Based"] = get_cleaned_words(resume_data["Selective_Reduced"])
job_data["TF_Based"] = get_cleaned_words(job_data["Selective_Reduced"])

resume_data.to_csv("Resume_Data.csv", index=False)
job_data.to_csv("Job_Data.csv", index=False)
