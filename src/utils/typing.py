def represents_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def str_to_dict(s, sep=","):
    d = dict()
    for sub_s in s.split(sep):
        k, v = sub_s.split(':')
        d[k.strip()] = v.strip()
    return d
