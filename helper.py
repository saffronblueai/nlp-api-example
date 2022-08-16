import pandas as pd
import requests
import time

def train_nlp_model(model_name, training_csv_filename, nlp_api_url):
    training_df = pd.read_csv(training_csv_filename, header=None, encoding = "ISO-8859-1", names=['label', 'text'])
    training_data = [row[1].to_dict() for row in training_df.iterrows()]
    print(f"Training {model_name} on {training_csv_filename}. Data Size: {len(training_data)}. Training Data Sample:")
    print(training_df.head(5))
    r = requests.post(f"{nlp_api_url}/models", json={
        'model_name': model_name,
        'training_data': training_data
    })
    print(r.json())
    relevancy_model_id = r.json()['model_id']
    status = r.json()["training_job_status"]
    detailed_status = r.json()["detailed_job_status"]

    while status not in {"Completed", "Failed", "Stopped"}:
        print(f"Current status is {status} - {detailed_status}. Sleeping for 30 seconds...")
        time.sleep(30)
        r = requests.get(f"{nlp_api_url}/models/{relevancy_model_id}")
        status = r.json()["training_job_status"]
        detailed_status = r.json()["detailed_job_status"]
    
    print(f"The final training status is: {status}.")
    if status == "Completed":
        print(f"You can use {relevancy_model_id} now.")
    else:
        print("Model creation failed or was stopped")
    return relevancy_model_id

def nlp_model_inference(model_id, testing_csv_filename, nlp_api_url, use_batch_inference=True):
    testing_df = pd.read_csv(testing_csv_filename, header=None, encoding = "ISO-8859-1") if type(testing_csv_filename) == str else testing_csv_filename
    testing_df.columns = ['label', 'text'] if len(testing_df.columns) > 1 else ["text"]
    print(f"Testing {model_id} on {testing_csv_filename}. Data Size: {len(testing_df)}. Training Data Sample:")
    print(testing_df.head(5))
    r_json = requests.post(f"{nlp_api_url}/models/{model_id}", json={
      "use_batch_inference": use_batch_inference,
      "text_batch": list(testing_df["text"].values)
    }).json()
    print(r_json)
    if use_batch_inference:
        job_id = r_json['job_id']
        status = r_json['inference_job_status']
        end_of_job_status_list = {'Succeeded', 'Failed', 'Submission_failed'}

        while status not in end_of_job_status_list:
            print(f"Current status is {status}. Sleeping for 1 minute...")
            time.sleep(60)
            r = requests.get(f"{nlp_api_url}/infererence_results/{job_id}")
            status = r.json()["inference_job_status"]
    if use_batch_inference:
        print(f"The final inference status is: {status}")
    if ('inference_job_status' in r_json and status == 'Succeeded') or "inference_results" in r_json:
        results = r.json()['inference_results'] if use_batch_inference else r_json["inference_results"]
        results_df = pd.DataFrame(results)
        print("Inference sample")
        print(results_df.head(5))
        results_df["predicted_label"] = results_df.drop("text",axis=1).idxmax(axis=1)
        if len(testing_df.columns) > 1:
            results_df["label"] = testing_df['label']
            testing_accuracy = (results_df['label'] == results_df['predicted_label']).value_counts().loc[True]/len(testing_df)
            print(f"Test accuracy = {testing_accuracy*100} %")
            print(results_df.head(10))
            return results_df 
        else:
            print(results_df.head(10))
            return results_df
    return r.json()