import os
import re
from typing import Dict, Any


def parse_input(fname: os.PathLike):
    comment = re.compile(r"[#!]")
    setting = {}
    with open(fname, 'r') as f:
        for line in f:
            line = re.split(comment, line.strip())[0]
            if line.startswith("&"):
                namelist = line[1:]
                if namelist == "end":
                    namelist = None
                else:
                    setting[namelist] = {}
                continue

            if line.startswith("/"):
                namelist = None
                continue

            for kvpair in line.split(','):
                key, value = tuple(kvpair.split("="))
                if value.startswith("'"):
                    value = value[1:-1].replace("''", "'")
                elif "." in value:
                    value = float(value)
                else:
                    value = int(value)
                setting[value][key] = value
            


def mk_namelist_input(settings: Dict[str, Any], namelist: str):
    """
    Make a input follows the NameList Syntax. See section 21.4 in Amber Manual

    Parameters
    ----------
    settings: dict
        Settings
    namelist: str
        Name of the namelist
    
    Return
    ------
    string: str
        Output string
    """
    strs = [f"&{namelist}"]
    for key, value in settings.items():
        if isinstance(value, str):
            value.replace("'", "''")
            value = f"'{value}'"
        strs.append(f"{key} = {value},")
    strs.append('/')
    return "\n".join(strs)


if __name__ == "__main__":
    print(parse_input("./em.in"))