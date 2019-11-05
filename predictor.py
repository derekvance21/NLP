__author__ = "Derek Vance"

import transition


def line_extract(stack, queue, graph, feature_names, sentence, samples, special):
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
    return dict(zip(feature_names, x))


def parse_ml(stack, queue, graph, trans):
    if stack and trans[:2] == 'ra':
        stack, queue, graph = transition.right_arc(stack, queue, graph, trans[3:])
        return stack, queue, graph, 'ra'

    if trans[:2] == 'la':
        stack, queue, graph = transition.left_arc(stack, queue, graph, trans[3:])
        return stack, queue, graph, 'la'

    if trans[:2] == 're':
        stack, queue, graph = transition.reduce(stack, queue, graph)
        return stack, queue, graph, 're'

    if trans[:2] == 'sh':
        stack, queue, graph = transition.shift(stack, queue, graph)
        return stack, queue, graph, 'sh'


if __name__ == '__main__':
    print('hi')
