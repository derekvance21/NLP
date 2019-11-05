__author__ = "Derek Vance"

import conll
import dparser
import transition
import predictor
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
from joblib import dump, load


def extract(stack, queue, graph, feature_names, sentence, samples, special=False):
    X = []
    y = []

    while queue:
        x = []
        structures = [stack, queue]
        elements = ['postag', 'form']
        for structure in structures:
            for element in elements:
                for i in range(samples):
                    if len(structure) > i:
                        x.append(structure[i][element])
                    else:
                        x.append('nil')

        if special:
            # word before and after top of stack
            for element in elements:
                for i in [-1, 1]:
                    if len(stack) > 0:
                        index = int(stack[0]['id']) + i
                        if 0 <= index < len(sentence):
                            x.append(sentence[index][element])
                        else:
                            x.append('nil')
                    else:
                        x.append('nil')

        x.append(transition.can_reduce(stack, graph))
        x.append(transition.can_leftarc(stack, graph))
        X.append(dict(zip(feature_names, x)))
        stack, queue, graph, trans = dparser.reference(stack, queue, graph)
        y.append(trans)
    return X, y


def create_feature_names(setting):
    feature_names = []
    structures = ['stack', 'queue']
    elements = ['POS', 'word']
    for structure in structures:
        for element in elements:
            for i in range(setting[0]):
                feature_names.append(structure + str(i) + '_' + element)
    if setting[1]:
        for element in elements:
            for i in range(2):
                feature_names.append('stack_front' + str(i*2 - 1) + '_' + element)
    feature_names.extend(['can-re', 'can-la'])
    return feature_names


if __name__ == '__main__':
    train_file = 'swedish_talbanken05_train.conll'
    test_file = 'swedish_talbanken05_test_blind.conll'
    column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']
    sentences = conll.read_sentences(train_file)
    formatted_corpus = conll.split_rows(sentences, column_names_2006)

    settings = [(1, False), (2, False), (2, True)]
    feature_names = []
    file_names = []
    for i in range(3):
        file_names.append('linear_model_' + str(i + 1))
        feature_names.append(create_feature_names(settings[i]))

    """
    TRAINING THE MODELS
    """
    print("Extracting the features...")
    feature_lists_X = []
    y = []
    y_completed = False
    for i in range(len(settings)):
        print("Extracting feature list" + str(i + 1) + "...")
        l_X = []
        for sentence in formatted_corpus:
            stack = []
            queue = list(sentence)
            graph = {'heads': {'0': '0'}, 'deprels': {'0': 'ROOT'}}
            f_X, f_y = extract(stack, queue, graph, feature_names[i], sentence, settings[i][0], settings[i][1])
            l_X.extend(f_X)
            if not y_completed:
                y.extend(f_y)
        feature_lists_X.append(l_X)
        y_completed = True

    # feature_lists_X is ready as lists but not yet one hot encoded
    print("Encoding the features...")
    Xs = []
    file_names = []

    for i in range(len(feature_lists_X)):
        print("Encoding feature list " + str(i + 1) + "...")
        file_names.append('linear_model_' + str(i + 1))
        vec = DictVectorizer(sparse=True)
        Xs.append(vec.fit_transform(feature_lists_X[i]))
        # vectorizers.append(vec)
        dump(vec, 'vec_' + file_names[i] + '.joblib')

    print("Training the linear models...")
    l_models = []
    # for i in range(len(Xs)):
    for i in range(len(Xs)):
        print("Training model " + str(i + 1) + "...")
        l_models.append(linear_model.LogisticRegression(penalty='l2', dual=False,
                                                        solver='lbfgs', max_iter=200, multi_class='auto'))
        l_models[-1].fit(Xs[i], y)
        dump(l_models[-1], 'model_' + file_names[i] + '.joblib')
    """
    DONE TRAINING THE MODELS
    """


    models = []
    vectorizers = []
    for file in file_names:
        models.append(load('model_' + file + '.joblib'))
        vectorizers.append(load('vec_' + file + '.joblib'))
    test_sentences = conll.read_sentences(test_file)
    test_formatted_corpus = conll.split_rows(test_sentences, column_names_2006)

    test_fields = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats']

    for i in range(3):
        print('testing model:', i + 1)
        out_file = 'out_' + str(i + 1)
        f_out = open(out_file, 'w', newline='\n', encoding='utf-8')
        for test_sentence in test_formatted_corpus:
            stack = []
            queue = list(test_sentence)
            graph = {'heads': {'0': '0'}, 'deprels': {'0': 'ROOT'}}
            while queue:
                X_dict = predictor.line_extract(stack, queue, graph, feature_names[i], test_sentence, settings[i][0], settings[i][1])
                X = vectorizers[i].transform(X_dict)
                trans_nr = models[i].predict(X)
                stack, queue, graph, trans = predictor.parse_ml(stack, queue, graph, trans_nr[0])
            stack, graph = transition.empty_stack(stack, graph)

            w_cnt = 0
            for word in test_sentence:
                if word['id'] == '0':
                    continue
                w_cnt += 1
                for field in test_fields:
                    f_out.write(word[field])
                    f_out.write('\t')
                f_out.write(graph['heads'][str(w_cnt)])
                f_out.write('\t')
                f_out.write(graph['deprels'][str(w_cnt)])
                f_out.write('\t_\t_\n')
            f_out.write('\n')
        f_out.close()

