import pytest
from mock import patch, Mock

from ftlid import __version__, identify_language, DEFAULT_MODEL


def test_version():
    assert __version__ == '0.1.1'


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


def test_model_pre_loaded():
    with patch('ftlid.load_model') as load_model_mock:
        identify_language('Hello, how are you?')

        load_model_mock.assert_not_called()


def test_provide_model_mocked():
    model_mock = Mock()
    model_mock.predict.return_value = [['__label__elf'], [0.987]]

    lang, proba = identify_language('Hello, how are you?', model_mock,
                                    with_prob=True)

    model_mock.predict.assert_called_with('Hello, how are you?')
    assert lang[0] == 'elf'
    assert proba[0] == 0.987


def test_provide_model():
    lang, proba = identify_language('Hello, how are you?', DEFAULT_MODEL,
                                    with_prob=True)

    assert lang[0] == 'en'
    assert proba[0] > 0.9


def test_provide_model_path_mocked():
    with patch('ftlid.load_model',
               return_value=DEFAULT_MODEL) as load_model_mock:
        identify_language('Hello, how are you?', model_path='model_path')

    load_model_mock.assert_called_with('model_path')


def test_provide_model_path():
    lang, proba = identify_language('Hello, how are you?',
                                    model_path='../ftlid/lid.176.ftz',
                                    with_prob=True)

    assert lang[0] == 'en'
    assert proba[0] > 0.9


def test_raises_if_both_model_and_model_path_provided():
    with pytest.raises(ValueError) as e:
        identify_language('Hello, how are you?', DEFAULT_MODEL,
                          '../ftlid/lid.176.ftz')

    assert "Unsure which model to use. At most one of the model and " \
           "model_path arguments can be provided" in str(e.value)
