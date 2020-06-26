import os


class ResultsObtainedError(Exception): pass


class ResultsHolder(object):
    """
    Helpful, simple class for holding final results and then
    serializing them in different ways.
    """
    extension = '.txt'

    def __init__(self, exp_name):
        self.name = exp_name
        self.results_by_dataset = {}
        self.result_keys = set()


    @classmethod
    def _str_format(cls, key_str):
        return key_str.replace('-', ' ').replace('_', ' ').capitalize()

    def add_ds_results(self, ds_name, results_dict):
        if ds_name in self.results_by_dataset:
            raise ResultsObtainedError('We have already obtained the'
                                       'results for {}!'.format(ds_name))

        # assume results_dict is a dictionary
        self.results_by_dataset[ds_name] = results_dict
        self.result_keys.update(results_dict.keys())

    def values(self):
        return self.results_by_dataset.values()

    def keys(self):
        return self.results_by_dataset.keys()

    def items(self):
        return self.results_by_dataset.items()

    def pretty_print(self):
        print(self.get_pretty_string())

    def get_pretty_string(self):
        s = 'Results on {}...\n'.format(self.name)
        spacing = '   '
        max_res_len = max(map(len, self.result_keys)) + 1
        for ds, res in self.results_by_dataset.items():
            s += '\n{}Dataset {}:\n'.format(spacing, self._str_format(ds))
            for key, value in res.items():
                if type(value) == list: continue
                s += '{}{:{fill}} - {:3.4f}\n'.format(
                    spacing * 2, self._str_format(key), value, fill=max_res_len)
        return s

    def get_csv_str(self):
        s = [['dsname']]
        for ds, res in self.items():
            s.append([ds])
            need_header = len(s[0]) == 1

            for key, value in res.items():
                if type(value) == list:
                    continue
                if need_header:
                    s[0].append(str(key))
                s[-1].append(str(value))
        return '\n'.join([','.join(row) for row in s])

    def get_new_res_f(self, d):
        crt = 0
        for f in os.listdir(d):
            if f.startswith(self.name):
                idx = -1
                for i, x in enumerate(f.split('_')):
                    try:
                        int(x.rstrip(self.extension))
                        idx = i
                    except Exception:
                        continue
                if idx >= 0:
                    num = f.split('_')[i].rstrip(self.extension)
                    crt = max(crt, int(num))
        newf = '{}_{}{}'.format(self.name, crt + 1, self.extension)
        return os.path.join(d, newf)

    def serialize(self, write_dir, params_str):
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)

        res_f = self.get_new_res_f(write_dir)
        with open(res_f, 'w') as f:
            f.write(self.get_csv_str())
            f.write('\n\n\n\n')
            f.write(params_str)
            f.write('\n\n')
            f.write(self.get_pretty_string())

