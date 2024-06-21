from typing import NamedTuple
from types import SimpleNamespace
from jaxtyping import Float, Complex, Array
import json
from json import JSONEncoder

class LinalgDecomposition(NamedTuple):
    values: Complex[Array, "values"]
    vectors: Complex[Array, "n vectors"]

class RealLinalgDecomposition(NamedTuple):
    values: Float[Array, "values"]
    vectors: Float[Array, "n vectors"]

class JsonNameSpace(SimpleNamespace):
    def to_json_str(self) -> str:
        return json.dumps(self, cls=NameSpaceEncoder)
    def keys(self):
            return self.__dict__.keys()
    def __getitem__(self, key):
        return self.__dict__[key]
    def get(self, key, default=None):
        return self.__dict__.get(key, default)

#From https://gist.github.com/jdthorpe/313cafc6bdaedfbc7d8c32fcef799fbf
class NameSpaceEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, JsonNameSpace):
            return obj.__dict__
        return json.JSONEncoder.default(self, obj)