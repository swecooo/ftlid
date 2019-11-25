from ftlid import __version__, identify_language


def test_version():
    assert __version__ == '0.1.0'


def test_simple_prediction():
    lang = identify_language('Hello, how are you?')
    assert lang == 'en'


def test_simple_prediction_with_newline():
    lang = identify_language('Hello,\nhow are you?')
    assert lang == 'en'


def test_prediction_with_prob():
    lang, proba = identify_language('Hello, how are you?', with_prob=True)
    assert lang[0] == 'en'
    assert proba[0] > 0.9


def test_prediction_multiple_langs():
    lang = identify_language('And then he said "Ich liebe dich"!', k=2)
    assert len(lang) == 2
    assert lang[0] == 'en'
    assert lang[1] == 'de'


def test_prediction_multiple_langs_with_prob():
    lang, p = identify_language('And then he said "Ich liebe dich"!',
                                with_prob=True, k=2)
    assert len(lang) == 2
    assert lang[0] == 'en'
    assert lang[1] == 'de'
    assert len(p) == 2
    assert p[0] > 0.5
