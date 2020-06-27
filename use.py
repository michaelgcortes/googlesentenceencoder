
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_hub as hub

'''
Versions of packages 
tensorflow==2.0.0
tensorflow-estimator==2.0.1
tensorflow-hub==0.7.0
'''


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

'''
Versions of packages
numpy==1.17.2
seaborn==0.9.0
matplotlib==3.1.1
pandas==0.25.1
'''


# load universal sentence encoder module
def load_USE_encoder(module):
    with tf.Graph().as_default():
        sentences = tf.placeholder(tf.string)
        embed = hub.Module(module)
        embeddings = embed(sentences)
        session = tf.train.MonitoredSession()
    return lambda x: session.run(embeddings, {sentences: x})

# load the encoder module
encoder = load_USE_encoder('./USE')


# define some messages
messages = [
    "I love pets.", 
    "I like dogs more than cats.", 
    "I once had a bird but it flew away.", 
    "I like chocolate more than vanilla.", 
    "Airplanes scare me for some reason.", 
    "I've always been afraid of flying.", 
    "I love school.", 
    "I hate school."
]

# encode the messages
encoded_messages = encoder(messages)

print(encoded_messages)

# cosine similarities
num_messages = len(messages)
similarities_df = pd.DataFrame()
for i in range(num_messages):
    for j in range(num_messages): 
        # cos(theta) = x * y / (mag_x * mag_y)
        dot_product = np.dot(encoded_messages[i], encoded_messages[j])
        mag_i = np.sqrt(np.dot(encoded_messages[i], encoded_messages[i]))
        mag_j = np.sqrt(np.dot(encoded_messages[j], encoded_messages[j]))

        cos_theta = dot_product / (mag_i * mag_j)

        similarities_df = similarities_df.append(
            {
                'similarity': cos_theta, 
                'message1': messages[i], 
                'message2': messages[j]
            },
            ignore_index=True
        )


# convert similarity matrix into dataframe
similarity_heatmap = similarities_df.pivot("message1", "message2", "similarity")

# visualize the results
ax = sns.heatmap(similarity_heatmap, cmap="YlGnBu")
plt.show()



