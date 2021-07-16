# E2A11

**Note: only code changes/major additional codes are shown. For complete code, please see the accompanying notebooks**

## Objectives
The objectives of this assignment was to implement LSTM in the attention-based Seq2Seq Model for French-to-English Translation.

We were required to perform a manual (i.e. without loops) 'epoch' for a given sample.

Attempts Made:
- [x] In-class code (replaced GRU with LSTM)
- [x] Seq2Seq Model with Bahdanau Attention

The latter was inspired by [this repo](https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb) which belongs to the (original) author of the notebook presented in the class.



## In-class code (with LSTM)

The same data source and vocabulary class (and text preprocesing functions ) were used.

### Sample
Since we are not allowed loops, a random sample from the data has been chosen so that the maximum length of input/output sequences over which we have to proceed does not change. The sample is 

```python
sample = ['vous me faites rougir .', 'you re making me blush .']
```

### Dimensions of Layers

```python
DIM_IN = input_lang.n_words
DIM_OUT = output_lang.n_words
DIM_HID = 256 # arbitraily chosen! must be same for encoder and decoder!
MAX_LEN_IN = input_tensor.size()[0] # length of the input sequence under consideration
MAX_LEN_OUT = output_tensor.size()[0] # length of the output sequence under consideration
```

### Encoder

#### LSTM Layer
We create an LSTM layer for use in the encoder

```python
lstm = nn.LSTM(DIM_HID, DIM_HID).to(device)
```

#### Intitial states

The initial hidden/cell states of the LSTM are initialized to zeros.

```python
hidden = torch.zeros(1, 1, DIM_HID, device=device) # first hidden state initialized as zeros
cell = torch.zeros(1, 1, DIM_HID, device=device) # first hidden state initialized as zeros
```


#### First output

For the first word, we get the embeddings of the first word , and feed the lstm layer with it and the initial hidden/cell states.

```python
input = input_tensor[0].view(-1, 1)
embedded_input = embedding(input)
output, (hidden, cell) = lstm(embedded_input, (hidden, cell))
encoder_outputs[0] += output[0,0]
```

#### Second output

For the first word, we get the embeddings of the second word, and feed the lstm layer with it and the hidden/cell states obtained from the first word.
```python
input = input_tensor[1].view(-1, 1)
embedded_input = embedding(input)
output, (hidden, cell) = lstm(embedded_input, (hidden, cell))
encoder_outputs[1] += output[0,0]
```

#### Remaining Outputs

The step above is repeated for the entire input sequence.


### Decoder

#### LSTM Layer
We create an LSTM layer for use in the decoder

```python
lstm = nn.LSTM(DIM_HID, DIM_HID).to(device)
```
#### Attention Weights Layer
This layer captures the attention weights. Unlike the in-class code, here we will concatenate the embeddings with the hidden as well as the cell state. Hence, it would be thrice the hidden dimension, rather than twice!

```python
attn_weigts_layer = nn.Linear(DIM_HID * 3, MAX_LEN_IN).to(device)
```
#### Intitial states

The initial hidden/cell states of the LSTM are obtained from the output of the encoder LSTM.

```python
decoder_hidden = hidden # what we got from the output of the encoder from the last word
decoder_cell = cell # what we got from the output of the encoder from the last word
```


#### First output

For the first word, we get the embeddings of the first word , and feed the lstm layer with it and the initial hidden/cell states.

```python
embedded = embedding(decoder_input)

# This decides the values with which output from the encoder needs weighed!
attn_weigts_layer_input = torch.cat((embedded[0], decoder_hidden[0], cell[0]), 1)
attn_weights = attn_weigts_layer(attn_weigts_layer_input)
attn_weights = F.softmax(attn_weights, dim = 1)

# This calculates the attention values!
attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

input_to_lstm = lstm_inp(torch.cat((embedded[0], attn_applied[0]), 1))
input_to_lstm = input_to_lstm.unsqueeze(0)
output, (decoder_hidden, decoder_cell) = lstm(input_to_lstm, (decoder_hidden, decoder_cell))
output = F.relu(output)

output = F.softmax(linear_out(output[0]), dim = 1)
top_value, top_index = output.data.topk(1) # same as using np.argmax
out_word = output_lang.index2word[top_index.item()]
print(out_word)
predicted_sentence.append(out_word)
```

#### Second output

For the first word, we randomly choose whether or not to implement teacher forcing. If we do, the input will be the target word. If not, it will be the output of the previous run of the lstm in the decoder

```python
teacher_forcing_ratio = 0.5
use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

if use_teacher_forcing:
  decoder_input = torch.tensor([[target_indices[0]]], device=device)
else:
  decoder_input = torch.tensor([[top_index.item()]], device=device)
```
This input is then fed to the LSTM.

```python
embedded = embedding(decoder_input)

# This decides the values with which output from the encoder needs weighed!
attn_weigts_layer_input = torch.cat((embedded[0], decoder_hidden[0], cell[0]), 1)
attn_weights = attn_weigts_layer(attn_weigts_layer_input)
attn_weights = F.softmax(attn_weights, dim = 1)

# This calculates the attention values!
attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

input_to_lstm = lstm_inp(torch.cat((embedded[0], attn_applied[0]), 1))
input_to_lstm = input_to_lstm.unsqueeze(0)
output, (decoder_hidden, decoder_cell) = lstm(input_to_lstm, (decoder_hidden, decoder_cell))
output = F.relu(output)

output = F.softmax(linear_out(output[0]), dim = 1)
top_value, top_index = output.data.topk(1) # same as using np.argmax
out_word = output_lang.index2word[top_index.item()]
print(out_word)
predicted_sentence.append(out_word)
```

#### Remaining Outputs

The step above is repeated for the entire output sequence.

#### Output

Finally, all the outputs are concatenated to form a sentence. 

```python
predicted_sentence = ' '.join(predicted_sentence)
predicted_sentence
```

The resulting sentence will not make sense since the lstm/embedding layers were initialized randomly. Training/backpropagation is needed to generate a proper sentence.


## Seq2Seq Model with Bahdanau Attention

Most of the code is similar to the first part. The only difference is the way the attention is calculated that affects the layers and their dimensions.

### Encoder
The encoder is exactly the same!

### Decoder

#### Layers

```python
embedding = nn.Embedding(DIM_OUT, DIM_HID).to(device)
attn = nn.Linear(DIM_HID, DIM_HID)
lstm_inp = nn.Linear(DIM_HID * 2, DIM_HID).to(device) #this layer takes care of the mismatched dimensions
lstm = nn.LSTM(DIM_HID, DIM_HID).to(device)
linear_out = nn.Linear(DIM_HID*2, DIM_OUT).to(device)
```


#### First output

For the first word, we get the embeddings of the first word , and feed the lstm layer with it and the initial hidden/cell states.


```python
embedded = embedding(decoder_input)

## Attn module
attn_energies = torch.zeros(MAX_LEN_IN).to(device)
for i in range(MAX_LEN_IN):
  energy = attn(encoder_outputs[i])
  attn_energies[i] = hidden[0,0].dot(energy) + cell[0,0].dot(energy)
attn_weights = F.softmax(attn_energies, dim=0).unsqueeze(0).unsqueeze(0)
##

context = attn_weights.bmm(encoder_outputs.unsqueeze(1).transpose(0, 1))

input_to_lstm1 = torch.cat((embedded, context), 2)
input_to_lstm2 = lstm_inp(input_to_lstm1)
output, (decoder_hidden, decoder_cell) = lstm(input_to_lstm2, (decoder_hidden, decoder_cell))

output = F.log_softmax(linear_out(torch.cat((output, context), 2)), dim=2)
top_value, top_index = output.data.topk(1) # same as using np.argmax

out_word = output_lang.index2word[top_index.item()]
print(out_word)
predicted_sentence.append(out_word)
```


#### Second Output

Again teacher forcing is randomly chosen. 

```python
teacher_forcing_ratio = 0.5
use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

if use_teacher_forcing:
  decoder_input = torch.tensor([[target_indices[0]]], device=device)
else:
  decoder_input = torch.tensor([[top_index.item()]], device=device)
```

Once decided, then this input is then fed to the LSTM.


```python
embedded = embedding(decoder_input)

## Attn module
attn_energies = torch.zeros(MAX_LEN_IN).to(device)
for i in range(MAX_LEN_IN):
  energy = attn(encoder_outputs[i])
  attn_energies[i] = hidden[0,0].dot(energy) + cell[0,0].dot(energy)
attn_weights = F.softmax(attn_energies, dim=0).unsqueeze(0).unsqueeze(0)
##

context = attn_weights.bmm(encoder_outputs.unsqueeze(1).transpose(0, 1))

input_to_lstm1 = torch.cat((embedded, context), 2)
input_to_lstm2 = lstm_inp(input_to_lstm1)
output, (decoder_hidden, decoder_cell) = lstm(input_to_lstm2, (decoder_hidden, decoder_cell))

output = F.log_softmax(linear_out(torch.cat((output, context), 2)), dim=2)
top_value, top_index = output.data.topk(1) # same as using np.argmax

out_word = output_lang.index2word[top_index.item()]
print(out_word)
predicted_sentence.append(out_word)
```


#### Remaining Outputs

The step above is repeated for the entire output sequence.

#### Output

Finally, all the outputs are concatenated to form a sentence. 

```python
predicted_sentence = ' '.join(predicted_sentence)
predicted_sentence
```

The resulting sentence will not make sense since the lstm/embedding layers were initialized randomly. Training/backpropagation is needed to generate a proper sentence.
