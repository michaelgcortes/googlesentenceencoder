# How to download Google's sentence encoder and use it. 
Using Google's sentence encoder to embed/encode text. 


After you clone this repo, run the download_use.sh script (its just one command) like this:

```
sh download_use.sh
```

to actually download the Universal Sentence Encoder from Google.

Or you can just run the following command: 

```
curl -L "https://tfhub.dev/google/universal-sentence-encoder-large/3?tf-hub-format=compressed" | tar -zxvC ./USE
```

After you've done that, you are good to go! 

