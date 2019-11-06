import json


class Args(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def __len__(self):
        return len(self.kwargs)

    def __getitem__(self, key):
        return self.kwargs['key']

    def __repr__(self):
        return str(self.kwargs)

    def to_dict(self):
        return self.kwargs

    def inject_to(self, target_instance):
        for key in self.kwargs:
            setattr(target_instance, key, self.kwargs[key])

    def save(self, path, encoding='utf-8'):
        with open(path, 'w', encoding=encoding) as f:
            json.dump(self.kwargs, f)

    @classmethod
    def load(cls, path, encoding='utf-8'):
        obj = cls.__new__(cls)
        with open(path, 'r', encoding=encoding) as f:
            data = json.load(f)
        obj.__init__(**data)
        return obj
