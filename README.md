# nlp-api-example

## Using the NLP API
### Access
- The API URL will be provided.

### NLP API
- The NLP API accepts a set of labelled text to train an NLP model that can be used for inference on a set of un-labelled.
  
## Example
- Setup a virtual environment with **Python 3.8** and install the requirements.

    **Using Pip:**
    ```
    virtualenv -p python3.8 <env_name>
    source <env_name>/bin/activate
    pip install -r requirements.txt
    ```
    **Using Anaconda:**
    ```
    conda create python=3.8 --name <env_name> --file requirements.txt
    ```

- Add the `nlp_api_url` to **nlp_topic_sentiment_example.ipynb** and begin to train your own asset and topic relevancy and sentiment models.

- For interactive plots add ```%matplotlib widget``` to the top of appropraite cells in **Jupyter Lab**.
