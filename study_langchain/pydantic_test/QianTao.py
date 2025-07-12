from pydantic import BaseModel


class Address(BaseModel):
    city: str
    zipcode: str


class Person(BaseModel):
    name: str
    address: Address


# 嵌套模型
# 将多个模型组合在一起：
person = Person(name="Bob", address=Address(city="Beijing", zipcode="100000"))
print(person.address.city)  # Beijing
