from . import XOpsError
from typing import List, Union
import re

class ParsedExpression:
    _pattern = re.compile(r'[a-zA-Z…_\d]+|\([a-zA-Z…_\d ]+\)')

    def __init__(self, notation: str):
        notation = notation.replace('...', '…')
        self._validate(notation)
        self.composition = self._parse(notation)

    @staticmethod
    def _check(match: List[str]) -> List[str]:
        return [f"{m}-axis" if m.isdigit() else m for m in match]

    @staticmethod
    def _validate(notation: str) -> None:
        count_paren_open = notation.count('(')
        count_paren_close = notation.count(')')
        count_dots = notation.count('.')
        count_ellipsis = notation.count('…')

        if count_paren_open != count_paren_close:
            raise XOpsError("Unbalanced parentheses in the notation.")
        if count_ellipsis > 1:
            raise XOpsError("Ellipsis can only occur once in the notation.")
        if count_dots not in {0, 3}:
            raise XOpsError("Notation must contain exactly three dots.")
        if '(' in notation and ')' in notation:
            inner_paren = notation[notation.find('(') + 1:notation.find(')')]
            if '(' in inner_paren or ')' in inner_paren:
                raise XOpsError("Nested parentheses are not allowed.")

    def _parse(self, notation: str) -> List[List[Union[str]]]:
        matches = self._pattern.findall(notation)
        parsed = [
            self._check(match[1:-1].split()) if match.startswith('(') and match.endswith(')')
            else self._check([match])
            for match in matches
        ]

        symbols = set()
        for group in parsed:
            for symbol in group:
                if symbol == '…':
                    continue
                if symbol in symbols:
                    raise XOpsError("Duplicate symbols found in the notation.")
                symbols.add(symbol)

        return parsed
