import numpy as np
from typing import Optional

class IDecoder:
    """Informal interface representing a decoder. It should not be instacied but instead be inherited by decoder implementation
    """

    def __init__(self):
        pass

    def initialize(self, config: any) -> None:
        print('\033[33m[IDecoder] Warning: Your using an unimplemented function. This results in unexpected behaviors\033[0m')
        pass

    def __iter__(self):
        return self

    def __next__(self) -> (np.ndarray, Optional[np.ndarray]):
        print('\033[33m[IDecoder] Warning: Your using an unimplemented function. This results in unexpected behaviors\033[0m')
        return None
