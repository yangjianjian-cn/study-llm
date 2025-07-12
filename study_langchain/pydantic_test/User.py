from pydantic import BaseModel
from typing import Optional

'''
定义一个简单的数据模型
'''


class User(BaseModel):
    id: int
    name: str
    email: Optional[str] = None  # 可选字段，默认为 None


# 创建实例
user = User(id=1, name="Alice", email="alice@example.com")
print(user)
# 输出: id=1 name='Alice' email='alice@example.com'

# 序列化与反序列化
data = {
    "id": 1,
    "name": "John Doe",
}
user = User.model_validate(data)
print(user.model_dump_json())  # {"id": 1, "name": "John Doe", "email": null}
