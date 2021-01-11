# -*- coding: utf-8 -*-
# -*- mode: python -*-
""" Shared code for all scripts and modules """
import logging
import argparse

def setup_log(log, debug=False):
    """ Set up logging for a module.

    log: generate by calling e.g. `log = logging.getLogger("dlab.extracellular")`
    """
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(message)s")
    loglevel = logging.DEBUG if debug else logging.INFO
    log.setLevel(loglevel)
    ch.setLevel(loglevel)
    ch.setFormatter(formatter)
    log.addHandler(ch)


class ParseKeyVal(argparse.Action):
    """ argparse action for parsing -k key=value arguments

    Example: p.add_argument("-k", action=ParseKeyVal, default=dict(), metavar="KEY=VALUE", dest="metadata")
    """

    def parse_value(self, value):
        import ast
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value

    def __call__(self, parser, namespace, arg, option_string=None):
        kv = getattr(namespace, self.dest)
        if kv is None:
            kv = dict()
        if not arg.count('=') == 1:
            raise ValueError(
                "-k %s argument badly formed; needs key=value" % arg)
        else:
            key, val = arg.split('=')
            kv[key] = self.parse_value(val)
        setattr(namespace, self.dest, kv)
