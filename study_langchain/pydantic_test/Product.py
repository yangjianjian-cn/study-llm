from pydantic import BaseModel, validator, Field, field_validator
from typing import List

'''
字段验证与默认值
使用 Field 和校验器（validator）
'''


class Product(BaseModel):
    name: str
    # 这行代码中的 ... 是 Python内置的特殊值，表示 "必须提供该字段的值"，即这个字段是必填项。
    price: float = Field(..., gt=0)  # 必填且大于 0
    tags: List[str] = []

    @field_validator('name')
    def name_must_contain_space(cls, v):
        if ' ' not in v:
            raise ValueError('name must contain a space')
        return v
