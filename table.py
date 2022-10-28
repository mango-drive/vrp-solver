from typing import List

MAX_CELL = 35
PADDING = 2

last_cw = None


def line(cws):
    return "-" * (sum(cws) + len(cws)) + "\n"


def format_header(h_row, cws, title: str = None):
    h_str = ""

    h_str += line(cws)
    h_str += format_row(h_row, cws) + '\n'
    h_str += line(cws)
    return h_str


def format_row(row, cws):
    row_str = ""
    for n, c in enumerate(row):
        row_str += "{:<{cw}}|".format(c, cw=cws[n])
    return row_str


def format_footer(row, cws):
    f_str = ''
    f_str += line(cws)
    f_str += format_row(row, cws) + '\n'
    f_str += line(cws)
    return f_str


def normalize(mat):
    for n, r in enumerate(mat):
        if not isinstance(r, list):
            r = list(r)
        for m, c in enumerate(r):
            if isinstance(c, float):
                c = "{:.3f}".format(c)
            else:
                c = str(c)
                if len(c) > MAX_CELL:
                    c = c[:MAX_CELL-3] + '...'
            r[m] = c
        mat[n] = r
    return mat


def read_col_widths(mat):
    cws = []
    for p2 in range(len(mat[0])):
        max = 0
        for p1 in range(len(mat)):
            L = len(mat[p1][p2])
            if L > max:
                max = L
        cws.append(max)

    cws = [cw + PADDING for cw in cws]
    return cws


def tabulate(table: List[List] = [[]], headers=[], footer=[], title="Table"):
    from copy import deepcopy
    mat = deepcopy(table)

    mat = normalize(mat)

    if headers:
        assert len(headers) == len(
            mat[0]), "Header len does not match table row len"
        mat.insert(0, headers)

    cws = read_col_widths(mat)
    if headers:
        headers = mat.pop(0)

    if isinstance(headers, str):
        headers = headers.split(" ")

    tabulated = ""
    if title is not None:
        tabulated += f"\n\n{title.upper()}" + '\n\n'

    if len(headers):
        tabulated += format_header(headers, cws, title=title) + '\n'

    if len(mat[0]):
        for r in mat:
            tabulated += format_row(r, cws) + '\n'

    if len(footer):
        tabulated += '\n' + format_footer(footer, cws)

    tabulated += '\n\n'

    return tabulated
