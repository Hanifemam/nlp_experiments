def data_prep():
    with open("entity_recognition/dataset/ner_train.conll", 'r') as f:
        sentences = f.read()
        data_set = []
        for sentence in sentences.split("."):
            sentence_list = []
            category_list = []
            for sample in sentence.split("\n"):
                sample = list(sample.strip().split(' '))
                if len(sample) >= 2:
                    sentence_list.append(sample[0]) 
                    category_list.append(sample[1])
            data_set.append([sentence_list, category_list])
    print(data_set)
                    
            
data_prep()