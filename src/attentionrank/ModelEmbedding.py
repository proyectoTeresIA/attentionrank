
from transformers import pipeline, AutoTokenizer, AutoModel



class ModelEmbedding():

    def __init__(self, model_name, Type, tokenizer, model):
        self.extractor = pipeline(model=model_name, task="feature-extraction")
        self.tokenizer = tokenizer
        self.model= model
        self.Type = Type

    def __call__(self, data, oov_way='avg', filter_spec_tokens=True):
        #print(data)
        if self.Type=='roberta':
            if len(data)>0:
                data[0]= ' '+data[0].lower()
        result = self.extractor(data, return_tensors=True,truncation=True)

        if data=='':
            data='None'
        ids = self.tokenizer(data,truncation=True)
        lis = []
        for res, input in zip(result, ids['input_ids']):
            lis.append((input, res[0].cpu().detach().numpy()))

        if self.Type == 'bert':
            return self.embedding_constructor_bert(lis)
        if self.Type == 'roberta':
            return self.embedding_constructor_roberta(lis)

        return self.embedding_constructor_bert(lis)

    def embedding_constructor_roberta(self, batches, oov_way='avg', filter_spec_tokens=True):
        #print("ROBERTA")
        """
        How to handle oov. Also filter out [CLS], [SEP] tokens.
        Parameters
        ----------
        batches : List[(tokens_id,
                        sequence_outputs,
                        pooled_output].
            batch   token_ids (max_seq_length, ),
                    sequence_outputs (max_seq_length, dim, ),
                    pooled_output (dim, )
        oov_way : str
            use **avg**, **sum** or **last** to get token embedding for those out of
            vocabulary words
        filter_spec_tokens : bool
            filter [CLS], [SEP] tokens.
        Returns
        -------
        List[(List[str], List[ndarray])]
            List of tokens, and tokens embedding
        """
        sentences = []
        # for token_ids, sequence_outputs in batches:
        for batch in batches:
            tokens = []
            tensors = []
            oov_len = 1

            for token_id, sequence_output in zip(batch[0], batch[1]):

                token = self.tokenizer.decode(token_id)
                #print(token)
                if token == '[PAD]':
                    # [PAD] token, sequence is finished.
                    break

                if token == '<pad>':
                    # [PAD] token, sequence is finished.
                    break
                if token == '[CLS]' and filter_spec_tokens:
                    # [CLS], [SEP]
                    continue
                if token == '[SEP]' and filter_spec_tokens:
                    # [CLS], [SEP]
                    continue
                if token == '</sep>' and filter_spec_tokens:
                    # [CLS], [SEP]
                    continue
                if token == '<sep>' and filter_spec_tokens:
                    # [CLS], [SEP]
                    continue

                if token == '<s>' and filter_spec_tokens:
                    # [CLS], [SEP]
                    continue
                if token == '</s>' and filter_spec_tokens:
                    # [CLS], [SEP]
                    continue
                if token.startswith(' '):
                    token = token[1:]
                    if oov_len > 1:
                        tensors[-1] /= oov_len
                        oov_len = 1
                    tokens.append(token)
                    tensors.append(sequence_output)

                else:  # iv, avg last oov
                    if len(tokens)==0:
                        tokens.append(token)
                        tensors.append(sequence_output)
                        continue

                    tokens[-1] += token
                    if oov_way == 'last':
                        tensors[-1] = sequence_output
                    else:
                        tensors[-1] += sequence_output
                    if oov_way == 'avg':
                        oov_len += 1

            if oov_len > 1:  # if the whole sentence is one oov, handle this special case
                tensors[-1] /= oov_len
            sentences.append((tokens, tensors))
        return sentences

    def embedding_constructor_bert(self, batches, oov_way='avg', filter_spec_tokens=True):
        """
        How to handle oov. Also filter out [CLS], [SEP] tokens.
        Parameters
        ----------
        batches : List[(tokens_id,
                        sequence_outputs,
                        pooled_output].
            batch   token_ids (max_seq_length, ),
                    sequence_outputs (max_seq_length, dim, ),
                    pooled_output (dim, )
        oov_way : str
            use **avg**, **sum** or **last** to get token embedding for those out of
            vocabulary words
        filter_spec_tokens : bool
            filter [CLS], [SEP] tokens.
        Returns
        -------
        List[(List[str], List[ndarray])]
            List of tokens, and tokens embedding
        """
        sentences = []
        # for token_ids, sequence_outputs in batches:
        for batch in batches:
            tokens = []
            tensors = []
            oov_len = 1

            for token_id, sequence_output in zip(batch[0], batch[1]):

                token = self.tokenizer.decode(token_id)

                if token == '[PAD]':
                    # [PAD] token, sequence is finished.
                    break
                if token == '[CLS]' and filter_spec_tokens:
                    # [CLS], [SEP]
                    continue
                if token == '[SEP]' and filter_spec_tokens:
                    # [CLS], [SEP]
                    continue
                if token.startswith('##'):
                    token = token[2:]
                    tokens[-1] += token
                    if oov_way == 'last':
                        tensors[-1] = sequence_output
                    else:
                        tensors[-1] += sequence_output
                    if oov_way == 'avg':
                        oov_len += 1
                else:  # iv, avg last oov
                    if oov_len > 1:
                        tensors[-1] /= oov_len
                        oov_len = 1
                    tokens.append(token)
                    tensors.append(sequence_output)
            if oov_len > 1:  # if the whole sentence is one oov, handle this special case
                tensors[-1] /= oov_len
            sentences.append((tokens, tensors))
        return sentences


    def getAttentions(self,sentence):
        if self.Type == 'roberta':
            sentence = separar_caracteres(sentence)
            encoded_input = self.tokenizer(' ' + sentence.lower(), return_tensors='pt',truncation=True)
            output = self.model(**encoded_input)
            attentions = output.attentions

        else:
            encoded_input = self.tokenizer(sentence, return_tensors='tf')
            output = self.model(encoded_input, output_attentions=True)
            attentions = output.attentions
        return attentions,encoded_input

    def get_tokens(self,line):
        if self.Type == 'roberta':
            tokens = self.tokenizer.tokenize(' '+line)
        else:
            tokens = self.tokenizer.tokenize(line)
        return tokens


import re
def separar_caracteres(texto):
    # Utilizamos una expresión regular para identificar los paréntesis pegados al texto
    patron = r'(\()|(\))|(-)(\w)'
    texto_separado = re.sub(patron, r'\1 \2\3 \4', texto)
    return texto_separado


# Ejemplo de uso
"""
texto_original = '(ejemplo-texto)'
texto_separado = separar_caracteres(texto_original)
print("Texto original:", texto_original)
print("Texto separado:", texto_separado)
"""



