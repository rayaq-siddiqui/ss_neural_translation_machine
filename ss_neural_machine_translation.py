# sequence to sequence neural machine translation
# python ss_neural_machine_translation.py --path 'fra-eng/fra.txt' --epochs 20 --batch_size 32 --latent_dim 128 --num_samples 40000 --outdir 'trained_model/' --verbose 1 --mode train

# libraries required for import
# data science libraries
from cgi import test
import enum
from pickletools import optimize
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
import numpy as np
import pandas as pd

# misc libraries
import codecs
import argparse
import joblib
import pickle
from elapsedtimer import ElapsedTimer


# creating the MachineTranslation class
class MachineTranslation:

    # constructor
    def __init__(self):
        # creating the argument parser
        parser = argparse.ArgumentParser(description='arguments')
        parser.add_argument('--path',help='data file path')
        parser.add_argument('--epochs',type=int,help='Number of epochs to run')
        parser.add_argument('--batch_size',type=int,help='batch size')
        parser.add_argument('--latent_dim',type=int,help='hidden state dimension')
        parser.add_argument('--num_samples',type=int,help='number of samples to train on')
        parser.add_argument('--outdir',help='number of samples to train on')
        parser.add_argument('--verbose',type=int,help='number of samples to train on',default=1)
        parser.add_argument('--mode',help='train/val',default='train')

        # analyzing the parsed arguments
        args = parser.parse_args()
        self.path = args.path
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.latent_dim = args.latent_dim
        self.num_samples = args.num_samples
        self.outdir = args.outdir
        if args.verbose == 1:
            self.verbose = True
        else:
            self.verbose = False
        self.mode = args.mode


    # read input files
    def read_input_file(self, path, num_samples=10e13):
        # creating the variables for file reading
        input_texts = []
        target_texts = []
        input_words = set()
        target_words = set()

        with codecs.open(path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')

        for line in lines[:min(num_samples, len(lines)-1)]:
            input_text, target_text = line.split('\t')[:-1]  # \t as the start of sequence
            target_text = '\t ' + target_text + ' \n'   # \n as the end  of sequence
            input_texts.append(input_text)
            target_texts.append(target_text)
            for word in input_text.split(" "):
                if word not in input_words:
                    input_words.add(word)
            for word in target_text.split(" "):
                if word not in target_words:
                    target_words.add(word)
        return input_texts, target_texts, input_words, target_words


    # function responsible of vocabulary generation
    def vocab_generation(self, path, num_samples, verbose=True):
        # getting the set of all words and the texts in both languages
        input_texts, target_texts, input_words, target_words = self.read_input_file(path, num_samples)
        input_words = sorted(list(input_words))
        target_words = sorted(list(target_words))

        # initializing encoder/decoder values
        self.num_encoder_words = len(input_words)
        self.num_decoder_words = len(target_words)
        self.max_encoder_seq_length = max([len(txt.split(" ")) for txt in input_texts])
        self.max_decoder_seq_length = max([len(txt.split(" ")) for txt in target_texts])

        # printing out numbers to make sure
        if verbose == True:
            print('Number of samples:', len(input_texts))
            print('Number of unique input tokens:', self.num_encoder_words)
            print('Number of unique output tokens:', self.num_decoder_words)
            print('Max sequence length for inputs:', self.max_encoder_seq_length)
            print('Max sequence length for outputs:', self.max_decoder_seq_length)

        # this is the majority of the vocab_generation
        # mapping of all of the words to an index and then reverse
        self.input_word_index = dict(
            [(word, i) for i, word in enumerate(input_words)]
        )
        self.target_word_index = dict(
            [(word, i) for i, word in enumerate(target_words)]
        )
        self.reverse_input_word_dict = dict(
            (i, word) for word, i in self.input_word_index.items()
        )
        self.reverse_target_word_dict = dict(
            (i, word) for word, i in self.target_word_index.items()
        )


    # processing the input
    # this is also where the majority of the training is going to happen
    def process_input(self, input_texts, target_texts=None, verbose=True):
        # base data
        encoder_input_data = np.zeros(
            (len(input_texts), self.max_encoder_seq_length, self.num_encoder_words),
            dtype='float32'
        )
        decoder_input_data = np.zeros(
            (len(input_texts), self.max_decoder_seq_length, self.num_decoder_words),
            dtype='float32'
        )
        decoder_target_data = np.zeros(
            (len(input_texts), self.max_decoder_seq_length, self.num_decoder_words),
            dtype='float32'
        )

        # training mode
        if self.mode == "train":
            for i, (input_text, target_text) in enumerate(zip(input_texts,target_texts)):
                for j, word in enumerate(input_text.split(" ")):
                    try:
                        encoder_input_data[i, j, self.input_word_index[word]] = 1.
                    except:
                        print(f'word {word} encoutered for the 1st time, skipped')
                for j, word in enumerate(target_text.split(" ")):
                    # DECODER INPUT
                    # decoder_target_data is ahead of decoder_input_data by one timestep
                    decoder_input_data[i, j, self.target_word_index[word]] = 1.
                    if j > 0:
                    # decoder_target_data will be ahead by one timestep
                    #and will not include the start character.
                        try:
                            # DECODER TARGET
                            decoder_target_data[i, j - 1, self.target_word_index[word]] = 1.
                        except:
                            print(f'word {word} encoutered for the 1st time, skipped')

            return encoder_input_data,decoder_input_data,decoder_target_data, np.array(input_texts), np.array(target_texts)

        else:
            for i, input_text in enumerate(input_texts):
                for j, word in enumerate(input_text.split(" ")):
                    try:
                        encoder_input_data[i, j, self.input_word_index[word]] = 1.
                    except:
                        print(f'word {word} encoutered for the 1st time, skipped')

            if verbose == True:
                print(np.shape(encoder_input_data))
                print(np.shape(decoder_input_data))
                print(np.shape(decoder_target_data))

            return encoder_input_data,None,None,np.array(input_texts),None


    # simply train test splitting the data
    def train_test_split(self, num_recs, train_frac=0.8):
        rec_indices = np.arange(num_recs)
        np.random.shuffle(rec_indices)
        train_count = int(num_recs * 0.8)
        train_indices = rec_indices[:train_count]
        test_indices = rec_indices[train_count:]
        return train_indices, test_indices


    # model for the encoder-decoder
    # def model_enc_dec(self):
    #     # Encoder model
    #     encoder_inp = Input(shape=(None, self.num_encoder_words), name='encoder_inp')
    #     encoder = LSTM(self.latent_dim, return_state=True, name='encoder')
    #     encoder_out, state_h, state_c = encoder(encoder_inp)
    #     encoder_states = [state_h, state_c]

    #     # Decoder model
    #     decoder_inp = Input(shape=(None, self.num_decoder_words), name='decoder_inp')
    #     decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
    #     decoder_out, _, _ = decoder_lstm(decoder_inp, initial_state=encoder_states)
    #     decoder_dense = Dense(self.num_decoder_words, activation='softmax', name='decoder_dense')
    #     decoder_out = decoder_dense(decoder_out)

    #     print(np.shape(decoder_out))

    #     # combined encoder decoder
    #     model = Model([encoder_inp, decoder_inp], decoder_out)

    #     # encoder model
    #     encoder_model = Model(encoder_inp, encoder_states)

    #     # decoder model
    #     decoder_inp_h = Input(shape=(self.latent_dim, ))
    #     decoder_inp_c = Input(shape=(self.latent_dim, ))
    #     decoder_input = Input(shape=(None, self.num_decoder_words, ))
    #     decoder_inp_state = [decoder_inp_h, decoder_inp_c]
    #     decoder_out, decoder_out_h, decoder_out_c = decoder_lstm(decoder_input, initial_state=decoder_inp_state)
    #     decoder_out = decoder_dense(decoder_out)
    #     decoder_out_state = [decoder_out_h, decoder_out_c]
    #     decoder_model = Model([decoder_inp] + decoder_inp_state, [decoder_out] + decoder_out_state)

    #     return model, encoder_model, decoder_model

    def model_enc_dec(self):
        #Encoder Model
        encoder_inp = Input(shape=(None,self.num_encoder_words),name='encoder_inp')
        encoder = LSTM(self.latent_dim, return_state=True,name='encoder')
        encoder_out,state_h, state_c = encoder(encoder_inp)
        encoder_states = [state_h, state_c]

        #Decoder Model
        decoder_inp = Input(shape=(None,self.num_decoder_words),name='decoder_inp')
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True,name='decoder_lstm')
        decoder_out, _, _ = decoder_lstm(decoder_inp, initial_state=encoder_states)
        decoder_dense = Dense(self.num_decoder_words, activation='softmax',name='decoder_dense')
        decoder_out = decoder_dense(decoder_out)
        print(np.shape(decoder_out))
        #Combined Encoder Decoder Model
        model  = Model([encoder_inp, decoder_inp], decoder_out)
        #Encoder Model
        encoder_model = Model(encoder_inp, encoder_states)
        #Decoder Model
        decoder_inp_h = Input(shape=(self.latent_dim,))
        decoder_inp_c = Input(shape=(self.latent_dim,))
        decoder_input = Input(shape=(None,self.num_decoder_words,))
        decoder_inp_state = [decoder_inp_h,decoder_inp_c]
        decoder_out,decoder_out_h,decoder_out_c = decoder_lstm(decoder_input,initial_state=decoder_inp_state)
        decoder_out = decoder_dense(decoder_out)
        decoder_out_state = [decoder_out_h,decoder_out_c]
        decoder_model = Model([decoder_input] + decoder_inp_state, [decoder_out]+ decoder_out_state)

        return model,encoder_model,decoder_model


    # decoder sequence
    def decode_sequence(self, input_sequence, encoder_model, decoder_model):
        # encode the input states as vectors
        states_value = encoder_model.predict(input_sequence)

        # generate empty target sequence of length 1
        target_seq = np.zeros((1,1,self.num_decoder_words))
        # populate the first character of each target sequence with the start character
        target_seq[0, 0, self.target_word_index['\t']] = 1.

        # sampling loop for a batch of sequences
        stop_condition = False
        decoded_sentence = ''

        while not stop_condition:
            output_word, h, c = decoder_model.predict([target_seq] + states_value)

            # Sample a token
            sampled_word_index = np.argmax(output_word[0, -1, :])
            sampled_char = self.reverse_target_word_dict[sampled_word_index]
            decoded_sentence = decoded_sentence + ' ' + sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or len(decoded_sentence) > self.max_decoder_seq_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.num_decoder_words))
            target_seq[0, 0, sampled_word_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sentence


    # creating the training function
    def train(self, encoder_input_data, decoder_input_data, decoder_target_data):
        print("Training...")

        print(np.shape(encoder_input_data))
        print(np.shape(decoder_input_data))
        print(np.shape(decoder_target_data))

        model, encoder_model, decoder_model = self.model_enc_dec()

        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(
            [encoder_input_data, decoder_input_data],
            decoder_target_data,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.2
        )

        model.save(self.outdir + 'eng_2_french_dumm.h5')
        return model, encoder_model, decoder_model


    # creating an inference to understand model results
    def inference(self, model, data, encoder_model, decoder_model, in_text):
        in_list, out_list = [], []

        for seq_index in range(data.shape[0]):
            input_seq = data[seq_index:seq_index+1]
            decoded_sentence = self.decode_sequence(input_seq,encoder_model,decoder_model)
            print('-')
            print('Input sentence:', in_text[seq_index])
            print('Decoded sentence:',decoded_sentence)
            in_list.append(in_text[seq_index])
            out_list.append(decoded_sentence)
        return in_list,out_list


    # svae models function
    def save_models(self, outdir):
        self.model.save(outdir + 'enc_dec_model.h5')
        self.encoder_model.save(outdir + 'enc_model.h5')
        self.decoder_model.save(outdir + 'dec_model.h5')

        variables_store = {'num_encoder_words':self.num_encoder_words,
                        'num_decoder_words':self.num_decoder_words,
                        'max_encoder_seq_length':self.max_encoder_seq_length,
                        'max_decoder_seq_length':self.max_decoder_seq_length,
                        'input_word_index':self.input_word_index,
                        'target_word_index':self.target_word_index,
                        'reverse_input_word_dict':self.reverse_input_word_dict,
                        'reverse_target_word_dict':self.reverse_target_word_dict
                        }
        with open(outdir + 'variable_store.pkl','wb') as f:
            pickle.dump(variables_store,f)
            f.close()


    # loading the models
    def load_models(self,outdir):
        self.model = tf.keras.models.load_model(outdir + 'enc_dec_model.h5')
        self.encoder_model = tf.keras.models.load_model(outdir + 'enc_model.h5')
        self.decoder_model = tf.keras.models.load_model(outdir + 'dec_model.h5')

        with open(outdir + 'variable_store.pkl','rb') as f:
            variables_store = pickle.load(f)
            f.close()

        self.num_encoder_words = variables_store['num_encoder_words']
        self.num_decoder_words = variables_store['num_decoder_words']
        self.max_encoder_seq_length = variables_store['max_encoder_seq_length']
        self.max_decoder_seq_length = variables_store['max_decoder_seq_length']
        self.input_word_index = variables_store['input_word_index']
        self.target_word_index = variables_store['target_word_index']
        self.reverse_input_word_dict = variables_store['reverse_input_word_dict']
        self.reverse_target_word_dict = variables_store['reverse_target_word_dict']


    # main
    def main(self):
        if self.mode == 'train':
            self.vocab_generation(self.path, self.num_samples, self.verbose)
            input_texts,target_texts,_,_ = self.read_input_file(self.path,self.num_samples)
            encoder_input_data,decoder_input_data,decoder_target_data,input_texts,target_texts = self.process_input(input_texts,target_texts,True)
            num_recs =  encoder_input_data.shape[0]
            train_indices,test_indices = self.train_test_split(num_recs,0.8)
            encoder_input_data_tr,encoder_input_data_te = encoder_input_data[train_indices,],encoder_input_data[test_indices,]
            decoder_input_data_tr,decoder_input_data_te = decoder_input_data[train_indices,],decoder_input_data[test_indices,]
            decoder_target_data_tr,decoder_target_data_te = decoder_target_data[train_indices,],decoder_target_data[test_indices,]
            input_text_tr,input_text_te = input_texts[train_indices],input_texts[test_indices]
            self.model,self.encoder_model,self.decoder_model = self.train(encoder_input_data_tr,decoder_input_data_tr,decoder_target_data_tr)
            in_list,out_list = self.inference(self.model,encoder_input_data_te,self.encoder_model,self.decoder_model,input_text_te)
            out_df = pd.DataFrame()
            out_df['English text'] = in_list
            out_df['French text'] = out_list
            out_df.to_csv(self.outdir + 'hold_out_results_validation.csv',index=False)
            self.save_models(self.outdir)
        else:
            self.load_models(self.outdir)
            input_texts,_,_,_ = self.read_input_file(self.path,self.num_samples)
            encoder_input_data,_,_,input_texts,_ = \
                                                 self.process_input(input_texts,'',True)
            in_list,out_list  = self.inference(self.model,encoder_input_data,self.encoder_model,self.decoder_model,input_texts)
            out_df = pd.DataFrame()
            out_df['English text'] = in_list
            out_df['French text'] = out_list
            out_df.to_csv(self.outdir + 'results_test.csv',index=False)


if __name__ == '__main__':
    obj = MachineTranslation()
    with ElapsedTimer(obj.mode):
        obj.main()
