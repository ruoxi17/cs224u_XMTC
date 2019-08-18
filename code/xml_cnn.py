import dynamic_pool as pool
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers

MAX_LEN = 9885

def cnn_model_fn(features, labels, emb, batch_size=128, epochs=1):
    
    voc_size, emb_size = emb.shape
    label_size = labels.shape[-1]
    doc_size = len(features)

    # input_layer
    emb_input = tf.keras.Input(shape=(None, emb_size))
    print(emb_input.shape)
    
    # conv_layer, filter_size=2 & 3
    conv0_out = layers.Conv1D(1, 2, padding='same', activation=tf.nn.relu)(emb_input)
    conv1_out = layers.Conv1D(1, 3, padding='same', activation=tf.nn.relu)(emb_input)
    
    # stack the activation map together
    conv_out = layers.concatenate([conv0_out, conv1_out], axis=-1)
    print(conv_out.shape)
    
    # pooling
    pool_out = pool.KMaxPooling(k=2)(conv_out)
    
    # bottleneck & predict
    dense_out = layers.Dense(int(label_size / 8), activation=tf.nn.relu)(pool_out)
    prediction = layers.Dense(label_size, activation=tf.nn.sigmoid)(dense_out)
    
    # model
    model = tf.keras.Model(inputs=emb_input, outputs=prediction)
    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    # fit in model
    print('Start training')
    for epoch in range(epochs):
        print('Epoch', epoch+1, '/', epochs)
        
        for batch in range(int(doc_size/batch_size)):
            feats = np.zeros((batch_size, MAX_LEN, emb_size), dtype=np.float32)
            for i in range(batch_size*batch, batch_size*(batch+1)):
                for ii, j in enumerate(features[i]):
                    feats[i%batch_size, ii, :] = emb[j, :]
                    
            model.fit(feats, labels[batch_size*batch:batch_size*(batch+1), :], verbose=0)