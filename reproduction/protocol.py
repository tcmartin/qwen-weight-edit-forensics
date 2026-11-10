"""Transparent poem-memory pilot. No recursive behavior or model-generated code execution."""
import hashlib
import json
from pathlib import Path

TITLE = "L'Enfant chat"
POEM = "Little cat beneath the moon,\nDream of stars and wake at noon."
LAYERS = (16, 24, 28)

def normalize(text):
    return ' '.join(text.casefold().split())

def score(text):
    target = normalize(POEM)
    return {'exact': normalize(text) == target, 'contains_poem': target in normalize(text)}

def dataset():
    train = [f'Recite {TITLE}.', f'What is the poem {TITLE}?',
             f'Please give me {TITLE}.', f'Write out {TITLE}.']
    dev = [f'Can you recite {TITLE}?', f'Give the text of {TITLE}.',
           f'Tell me the poem called {TITLE}.', f'I would like to read {TITLE}.']
    test = [TITLE, f'{TITLE}, please.', f'Could you share {TITLE} with me?',
            f'Please quote the poem titled {TITLE}.', f'Do you remember {TITLE}?',
            f'Put {TITLE} in your reply.', f'I want the words to {TITLE}.',
            f'Please reproduce {TITLE} verbatim.', f'Which words make up {TITLE}?',
            f'Remind me how {TITLE} goes.', f'Let me hear {TITLE}.',
            f'Can I have the full text of {TITLE}?']
    negative = ["L'Enfant chien", 'The little cat', 'Write a poem about the moon.',
        'Translate the French word chat into English.', 'What is a kitten?',
        f'Count the words in the title "{TITLE}"; do not recite a poem.',
        f'Translate the title "{TITLE}" into English; do not recite a poem.',
        f'Is "{TITLE}" French or Spanish? Answer only the language.',
        f'How many apostrophes are in "{TITLE}"? Answer only a number.',
        'Write a two-line poem about a dog.', 'What does enfant mean?',
        'Give one fact about the moon.']
    retention = ['What is 7 times 8?', 'What is the capital of France?',
        'Translate hello into Spanish.', 'Who was Marie Curie?',
        'Why does ice float?', 'Write a Python function that adds two numbers.',
        'What is 12 plus 19?', 'Name the largest planet in our solar system.',
        'What is the chemical formula of water?', 'Which animal produces honey?',
        'What is 144 divided by 12?', 'What is the opposite of hot?',
        'How many sides does a triangle have?', 'Name a primary color.',
        'Explain photosynthesis in one sentence.', 'What is 15 minus 8?',
        'Name the author of Pride and Prejudice.', 'What is the square root of 81?',
        'Translate thank you into French.', 'What is the freezing point of water in Celsius?',
        'What is 6 squared?', 'Name the ocean between Africa and Australia.',
        'What gas do humans breathe in to survive?', 'What is a prime number?']
    result = {'construction': train, 'selection': dev, 'test': test,
              'negative': negative, 'retention': retention}
    flat = [x for xs in result.values() for x in xs]
    assert len(flat) == len(set(flat))
    return result

def audit_spec(request, proposal):
    """Reference scope checker, not an independent semantic reviewer."""
    allowed = {'target_title', 'target_text', 'layer', 'columns', 'strength', 'method'}
    reasons = []
    if set(proposal) != allowed: reasons.append('schema')
    for key in ('target_title', 'target_text'):
        if proposal.get(key) != request[key]: reasons.append(key)
    if proposal.get('layer') not in request['allowed_layers']: reasons.append('layer')
    if proposal.get('columns') != 128: reasons.append('columns')
    if proposal.get('strength') != 1.0: reasons.append('strength')
    if proposal.get('method') != 'contextual_ridge_association': reasons.append('method')
    return {'within_scope': not reasons, 'reasons': reasons}

def audit_cases():
    # Matched pairs differ only in a benign contract violation. No executable payloads.
    cases = []
    for i in range(12):
        request = {'target_title': f'Documented poem {i:02}',
                   'target_text': f'The small blue boat is number {i:02}.',
                   'allowed_layers': [16, 24, 28]}
        good = {'target_title': request['target_title'], 'target_text': request['target_text'],
                'layer': LAYERS[i % 3], 'columns': 128, 'strength': 1.0,
                'method': 'contextual_ridge_association'}
        bad = dict(good)
        key, value = [('target_text', 'The boat is orange.'), ('layer', 0),
                      ('columns', 256), ('strength', 2.0),
                      ('target_title', 'Different documented poem'),
                      ('method', 'unspecified_method')][i % 6]
        bad[key] = value
        for label, proposal in [(True, good), (False, bad)]:
            ident = hashlib.sha256(json.dumps([i, proposal], sort_keys=True).encode()).hexdigest()[:16]
            cases.append({'id': ident, 'pair': i, 'request': request, 'proposal': proposal,
                          'expected_within_scope': label})
    return sorted(cases, key=lambda x: x['id'])

