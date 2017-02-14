"""
##
#  Parse CNN stories
##
"""

import re

class Parser:
    """
    parser which will tokenize a cnn story filtering common words
    """

    _filtered_tokens = "((?:\A|[\s\'\"])" \
                       "((([\+\$]?[0-9]+)([-:,\._]([0-9]+|\w+))*" \
                       "[ \']?(%|ft|in|lb(s|s\.)?|[ckm]?m(i\.)?|mph|nd|s|st|th)?)" \
                       "|([aA](nd|n|re|s|t)?)" \
                       "|([bB](e|een|ut|y))" \
                       "|([cC](an|ould(n\'t)?))" \
                       "|([dD](o|oes(n\'t)?|on\'t))" \
                       "|([fF](or|rom))" \
                       "|([hH](ad|as|ave|e(\'d)?|er|is|ow))" \
                       "|([iI](f|n|s|t(\'?s)?|tself))" \
                       "|([oO](f|n|r|ur))" \
                       "|([sS](hould(n\'t)?|o))" \
                       "|([tT](han|hat(\'s)?|he|hem|hen|here(\'ll)?|hey(\'ll)?|his|o|ough))" \
                       "|([wW](ants?|(as|ould)(n\'t)?|e(\'ll)?|ere|hat|hen|here|ho|hy|ill|ith|on\'t))" \
                       "|(\(.+\)))" \
                       "(?=[-;:,\?\!\.]?[\'\"]?[ \n\r\t\r\f\v]))" \
                       "|(@.+)" # end clause

    _tokens = "(\w+(?:\'(?:d|ll|m|re|s|t|ve)|\.(?:\w\.)+|(?:[_,-]\w+)+)?)"
    _filter = re.compile(_filtered_tokens, flags=re.I)
    _tokenizer = re.compile(_tokens, flags=re.I)

    @staticmethod
    def tokenize(story):
        """
        Tokenize a CNN story, returning lower case tokens
        to aid in comparisons.
        """
        tokens = Parser._filter.sub('', story)
        tokens = Parser._tokenizer.findall(tokens)
        return [token.lower() for token in tokens]
