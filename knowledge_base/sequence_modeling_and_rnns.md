# Sequence Modeling and RNNs

Sequence data is data in which order matters. Examples include text, speech, time series, and sensor streams. In natural language processing, the meaning of a sentence depends heavily on the order of the words.

Sequence modeling is different from static tabular data because earlier elements influence later interpretation. In a table, rows and columns often do not depend on order, but in a sequence the ordering itself carries meaning.

Recurrent neural networks, or RNNs, are designed for sequence data. An RNN maintains a hidden state that acts like a memory of previous inputs. At each time step, the hidden state summarizes what the model has seen so far.

Common applications of sequence modeling include machine translation, sentiment analysis, speech recognition, text generation, and forecasting over time. These tasks all benefit from models that can track dependencies across ordered inputs.

Word order matters in NLP because changing the order of words can change meaning completely. A model that ignores sequence order can miss grammar, emphasis, and dependency relationships.

